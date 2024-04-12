import random

import torch
from typing import Dict

from datasets import get_dataset
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from copy import deepcopy
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.optim import SGD
import numpy as np


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Complementary Learning Systems Based Experience Replay')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    # Consistency Regularization Weight
    parser.add_argument('--reg_weight', type=float, default=0.1)

    # Stable Model parameters
    parser.add_argument('--stable_model_update_freq', type=float, default=0.70)
    parser.add_argument('--stable_model_alpha', type=float, default=0.999)

    return parser


# =============================================================================
# Mean-ER
# 127
# =============================================================================
class MEANER(ContinualModel):
    NAME = 'meaner'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(MEANER, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        # Initialize plastic and stable model
        self.stable_model = deepcopy(self.net).to(self.device)
        # set regularization weight
        self.reg_weight = args.reg_weight
        # set parameters for stable model
        self.stable_model_update_freq = args.stable_model_update_freq
        self.stable_model_alpha = args.stable_model_alpha

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0

        self.checkpoint = self.net.get_params().data.clone().to(self.device)
        self.big_omega = None
        self.temp_omega = None
        self.small_omega = 0
        self.model_alpha = 0.999
        self.xi = 1.0

        self.stable_flag = True

    def end_task(self, dataset):
        self.current_task += 1
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(self.net.get_params()).to(self.device)
            self.temp_omega = torch.zeros_like(self.net.get_params()).to(self.device)

        self.big_omega += self.small_omega / ((self.net.get_params().data - self.checkpoint) ** 2 + self.xi)

        self.temp_omega.copy_(self.big_omega)
        # 取big_omega的top一半
        top_k = int(self.temp_omega.shape[0] / 3)
        indices_to_zero = self.temp_omega < torch.topk(self.temp_omega, top_k)[0][..., -1, None]
        self.temp_omega[indices_to_zero] = 0  # 将小于一半的设为0，即不修改的

        # store parameters checkpoint and reset small_omega
        self.checkpoint = self.net.get_params().data.clone().to(self.device)
        self.small_omega = 0

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        loss = 0
        if self.current_task > 0:
            if self.flag:
                self.update_working_model_variables(self.stable_model)

        if not self.buffer.is_empty():

            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)

            stable_model_logits = self.stable_model(buf_inputs)

            l_cons = torch.mean(self.consistency_loss(self.net(buf_inputs), stable_model_logits.detach()))

            l_reg = self.args.reg_weight * l_cons
            loss += l_reg

            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/l_cons', l_cons.item(), self.iteration)
                self.writer.add_scalar(f'Task {self.current_task}/l_reg', l_reg.item(), self.iteration)
            ######
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        ce_loss = self.loss(outputs, labels)
        loss += ce_loss

        # Log values
        if hasattr(self, 'writer'):
            self.writer.add_scalar(f'Task {self.current_task}/ce_loss', ce_loss.item(), self.iteration)
            self.writer.add_scalar(f'Task {self.current_task}/loss', loss.item(), self.iteration)

        loss.backward()
        self.opt.step()

        self.small_omega += self.args.lr * self.net.get_grads().data ** 2

        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels[:real_batch_size],
        )

        # Update the ema model
        self.global_step += 1

        if torch.rand(1) < self.stable_model_update_freq:
            self.update_stable_model_variables()
            self.stable_flag = True
        else:
            self.stable_flag = False

        return loss.item()

    def update_stable_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.stable_model_alpha)
        for ema_param, param in zip(self.stable_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def update_working_model_variables(self, ema_model):
        alpha = min(1 - 1 / (self.global_step + 1), self.model_alpha)
        net_params = self.net.get_params()  # 自己铺开
        temp_params = deepcopy(net_params.data).to(self.device)
        ema_params = ema_model.get_params()
        temp_params.mul_(alpha).add_(ema_params.data, alpha=1 - alpha)  # 大就传

        tensor = torch.where(self.temp_omega == 0, net_params.data, temp_params)  # 小于一半的不变，大于的话传
        net_params.data.copy_(tensor)
        listshape = 0

        for param in self.net.parameters():
            p = net_params.data[listshape:(listshape + param.view(-1).shape[0])]
            listshape += param.view(-1).shape[0]
            p = p.reshape(param.data.shape)
            param.data.copy_(p)
