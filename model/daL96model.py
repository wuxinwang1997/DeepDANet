import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from base import BaseModel
from .backbones import SEResNet
from utils.model_utils import Identity

def named_da_network(da_model_name, n_input_channels, n_output_channels, **kwargs):

    if da_model_name == 'SEResNet':

        assert np.all([i == 3 for i in kwargs['kernel_sizes']])
        
        n_filters_ks3 = kwargs['filters']
        n_filters_ks1 = [kwargs['filters_ks1_inter'] for i in range(len(n_filters_ks3)-1)]
        n_filters_ks1 = [kwargs['filters_ks1_init']] + n_filters_ks1 + [kwargs['filters_ks1_final']]


        normLayers = {'BN' : paddle.nn.BatchNorm1D,
                      'ID' : Identity(),
                      paddle.nn.BatchNorm2D : paddle.nn.BatchNorm1D,
                      Identity() : Identity()
                     }                   

        model = SEResNet(n_filters_ks3=n_filters_ks3,
                       n_filters_ks1=n_filters_ks1, 
                       n_channels_in=n_input_channels, 
                       n_channels_out=n_output_channels, 
                       padding_mode=kwargs['padding_mode'],
                       layerNorm=normLayers[kwargs['layerNorm']],
                       dropout=kwargs['dropout_rate'],
                       direct_shortcut=kwargs['direct_shortcut'])

    else: 
        raise NotImplementedError()

    return model