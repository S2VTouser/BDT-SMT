# BDT-SMT

# Enhancing Class-Incremental Learning for Image Classification via Bidirectional Transport and Selective Momentum
This is the *Pytorch Implementation* for the paper Enhancing Class-Incremental Learning for Image Classification via Bidirectional Transport and Selective Momentum.

## Framework
![image]([https://github.com/S2VTouser/Rememory-based-SimSiam/files/13692458/frame.pdf](https://github.com/S2VTouser/Rememory-based-SimSiam/blob/main/img/frame.pdf))

## Abstract
Class-Incremental Learning (Class-IL) aims to continuously learn new knowledge without forgetting old knowledge from a given data stream in the realm of image classification. Recent Class-IL methods strive to balance old and new knowledge and have achieved excellent results in mitigating the forgetting by mainly employing the rehearsal-based strategy. However, the representation learning on new tasks is often impaired since the trade-off is hard to taken between old and new knowledge. To overcome this challenge, based on the Complementary Learning System (CLS) theory, we propose a novel CLS-based method by focusing on the representation of old and new knowledge under the Class-IL setting, which can acquire more new knowledge from new tasks while consolidating the old knowledge so as to make a better balance between them (i.e., enhancing the overall model performance). Specifically, our proposed method has two novel components: (1) To effectively mitigate the forgetting, we first propose a bidirectional transport (BDT) strategy between old and new models, which can better integrate the old knowledge into the new knowledge and meanwhile enforce the old knowledge to be better consolidated by bidirectionally transferring parameters across old and new models. (2) To ensure that the representation of new knowledge is not impaired by the old knowledge, we further devise a selective momentum (SMT) mechanism to give parameters greater flexibility to learn new knowledge while transferring important old knowledge, which is achieved by selectively (momentum) updating network parameters through parameter importance evaluation. Extensive experiments on five benchmarks show that our proposed method significantly outperforms the state-of-the-arts under the Class-IL setting.

### Contributions
* We propose a novel CLS-based method termed BDT-SMT to acquire more new knowledge from new tasks while consolidating the old knowledge so as to make a better balance between them under the Class-IL setting.
* To effectively mitigate the forgetting, we devise a bidirectional transport (BDT) strategy between old and new models, which is quite different from the latest works with only one unidirectional process (i.e., backward transport). Moreover, to ensure that the representation of new knowledge is not impaired by the old knowledge during forward transport, we design a selective momentum (SMT) mechanism to selectively (momentum) update network parameters through parameter importance evaluation.
* Extensive experiments on five benchmarks show that our proposed method significantly outperforms the state-of-the-art methods under the Class-IL setting.  

## Setup
* $ pip install -r requirements.txt
* Run experiments: $ python main.py

## Datasets
* Sequential MNIST 
* Sequential CIFAR-10 
* Sequential CIFAR-100 
* Sequential Tiny ImageNet
* Sequential miniImageNet

## Citation
If you found the provided code useful, please cite our work.

