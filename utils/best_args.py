# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

best_args = {
    'seq-mnist': {
        'sgd': {-1: {'lr': 0.03, 'batch_size': 10, 'n_epochs': 1}},
        'er': {200: {'lr': 0.01,
                     'minibatch_size': 10,
                     'batch_size': 10,
                     'n_epochs': 1},
               },
        'gss': {200: {'lr': 0.1,
                      'minibatch_size': 10,
                      'gss_minibatch_size': 10,
                      'batch_size': 128,
                      'batch_num': 1,
                      'n_epochs': 1},
                },
        'icarl': {200: {'lr': 0.1,
                        'minibatch_size': 10,
                        'wd_reg': 0,
                        'batch_size': 10,
                        'n_epochs': 1},
                  },
        'fdr': {200: {'lr': 0.03,
                      'minibatch_size': 128,
                      'alpha': 0.5,
                      'batch_size': 128, },
                },
        'der': {200: {'lr': 0.03,
                      'minibatch_size': 10,
                      'alpha': 0.2,
                      'batch_size': 10,
                      'n_epochs': 1},
                },
        'derpp': {200: {'lr': 0.03,
                        'minibatch_size': 128,
                        'alpha': 0.2,
                        'beta': 1.0,
                        'batch_size': 10,
                        'n_epochs': 1},
                  },
        'er_ace': {200: {'lr': 0.03,
                         'minibatch_size': 10,
                         'batch_size': 10,
                         'n_epochs': 1
                         },
                   },
        'bdt': {
            200: {
                'reg_weight': 2.0,
                'stable_model_update_freq': 0.9,
                'stable_model_alpha': 0.99,
                'plastic_model_update_freq': 1.0,
                'plastic_model_alpha': 0.99,
                'lr': 0.1,
                'minibatch_size': 128,
                'batch_size': 10,
                'n_epochs': 1, },
        },
    },
    'seq-cifar10': {'sgd': {-1: {'lr': 0.1,
                                 'batch_size': 32,
                                 'n_epochs': 50}},
                    'er': {200: {'lr': 0.1,
                                 'minibatch_size': 32,
                                 'batch_size': 32,
                                 'n_epochs': 50},
                           },
                    'gem': {200: {'lr': 0.03,
                                  'gamma': 0.5,
                                  'batch_size': 32,
                                  'n_epochs': 50},
                            },
                    'gss': {200: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'gss_minibatch_size': 32,
                                  'batch_size': 32,
                                  'n_epochs': 50,
                                  'batch_num': 1},
                            },
                    'icarl': {200: {'lr': 0.1,
                                    'minibatch_size': 0,
                                    'softmax_temp': 2.0,
                                    'wd_reg': 0.00001,
                                    'batch_size': 32,
                                    'n_epochs': 50},
                              },
                    'fdr': {200: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'alpha': 0.3,
                                  'batch_size': 32,
                                  'n_epochs': 50},
                            },
                    'der': {200: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'alpha': 0.3,
                                  'batch_size': 32,
                                  'n_epochs': 50},
                            },
                    'derpp': {200: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'alpha': 0.1,
                                    'beta': 0.5,
                                    'batch_size': 32,
                                    'n_epochs': 50},
                              },
                    ######
                    'er_ace': {200: {'lr': 0.03,  #
                                     'minibatch_size': 32,
                                     'batch_size': 32,
                                     'n_epochs': 50
                                     },
                               },
                    'bdt': {
                        200: {
                            'reg_weight': 0.15,
                            'stable_model_update_freq': 0.1,
                            'stable_model_alpha': 0.999,
                            'plastic_model_update_freq': 0.6,
                            'plastic_model_alpha': 0.999,
                            'lr': 0.1,
                            'minibatch_size': 32,
                            'batch_size': 32,
                            'n_epochs': 50, },
                    },
             },

    'seq-cifar100': {
        'sgd': {-1: {'lr': 0.03,
                     'batch_size': 32,
                     'n_epochs': 50}},
        'er': {200: {'lr': 0.1,
                     'minibatch_size': 32,
                     'batch_size': 32,
                     'n_epochs': 50},
               },
        'icarl': {200: {'lr': 0.3,
                        'minibatch_size': 0,
                        'softmax_temp': 2.0,
                        'wd_reg': 0.00001,
                        'batch_size': 32,
                        'n_epochs': 50},
                  },
        'er_ace': {200: {'lr': 0.1,  # 参数需调
                         'minibatch_size': 32,
                         'batch_size': 32,
                         'n_epochs': 50},
                   },
        'derpp': {200: {'lr': 0.03,
                        'minibatch_size': 32,
                        'alpha': 0.2,
                        'beta': 0.5,
                        'batch_size': 32,
                        'n_epochs': 50},
                  },
        'bdt': {
            200: {
                'reg_weight': 0.15,
                'stable_model_update_freq': 0.1,
                'stable_model_alpha': 0.999,
                'plastic_model_update_freq': 0.3,
                'plastic_model_alpha': 0.999,
                'lr': 0.05,
                'minibatch_size': 32,
                'batch_size': 32,
                'n_epochs': 50, },
        },
    },

    'seq-tinyimg': {'sgd': {-1: {'lr': 0.03,
                                 'batch_size': 32,
                                 'n_epochs': 100}},
                    'er': {200: {'lr': 0.1,
                                 'minibatch_size': 32,
                                 'batch_size': 32,
                                 'n_epochs': 100},
                           },
                    'icarl': {200: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'softmax_temp': 2.0,
                                    'wd_reg': 0.00001,
                                    'batch_size': 32,
                                    'n_epochs': 100},
                              },
                    'fdr': {200: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'alpha': 0.3,
                                  'batch_size': 32,
                                  'n_epochs': 100},
                            },
                    'derpp': {200: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'alpha': 0.1,
                                    'beta': 1.0,
                                    'batch_size': 32,
                                    'n_epochs': 100},
                              },
                    ##########
                    'er_ace': {200: {'lr': 0.03,  # 参数需调
                                    'minibatch_size': 32,
                                    'batch_size': 32,
                                    'n_epochs': 100},
                             },
                    'bdt': {
                        200: {
                            'reg_weight': 0.1,
                            'stable_model_update_freq': 0.04,
                            'stable_model_alpha': 0.999,
                            'plastic_model_update_freq': 0.08,
                            'plastic_model_alpha': 0.999,
                            'lr': 0.05,
                            'minibatch_size': 32,
                            'batch_size': 32,
                            'n_epochs': 50, },
                    },
                   },
    
    'seq-miniimg': {'sgd': {-1: {'lr': 0.03,
                                 'batch_size': 32,
                                 'n_epochs': 100}},

                    'gem': {200: {'lr': 0.1,
                                  'gamma': 0.5,
                                  'batch_size': 32,
                                  'n_epochs': 50},
                            },

                    'er': {200: {'lr': 0.05,
                                 'minibatch_size': 32,
                                 'batch_size': 32,
                                 'n_epochs': 100},
                           },

                    'icarl': {200: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'softmax_temp': 2.0,
                                    'wd_reg': 0.00001,
                                    'batch_size': 32,
                                    'n_epochs': 100},
                              },

                    'er_ace': {200: {'lr': 0.03,
                                     'minibatch_size': 32,
                                     'batch_size': 32,
                                     'n_epochs': 100},
                               },

                    'derpp': {200: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'alpha': 0.1,
                                    'beta': 1.0,
                                    'batch_size': 32,
                                    'n_epochs': 100},
                              },
                    'bdt': {
                        200: {
                            'reg_weight': 0.1,
                            'stable_model_update_freq': 0.1,
                            'stable_model_alpha': 0.999,
                            'plastic_model_update_freq': 0.3,
                            'plastic_model_alpha': 0.999,
                            'lr': 0.05,
                            'minibatch_size': 32,
                            'batch_size': 32,
                            'n_epochs': 50, },
                    },
                    },
}
