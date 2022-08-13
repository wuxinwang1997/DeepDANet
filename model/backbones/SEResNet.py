import imp
import paddle
import paddle.nn as nn
from utils.model_utils import setup_conv, SELayer, Identity

class ResNetBlock(paddle.nn.Layer):
    """A residual block to construct residual networks.
    Comprises 2 Conv1D operations with optional dropout and a normalization layer.

    Parameters
    ----------
    in_channels: int
        Number of channels of input tensor.
    kernel_size: list of (int, int)
        Size of the convolutional kernel for the residual layers.
    hidden_channels: int
        Number of output channels for first residual convolution.
    out_channels: int
        Number of output channels. If not equal to in_channels, will add
        additional 1x1 convolution.
    layerNorm: function
        Normalization layer.
    activation: str
        String specifying nonlinearity.
    padding_mode: str
        How to pad the data ('circular' for wrap-around padding on last axis)

    """
    def __init__(self, in_channels, kernel_size,
                 hidden_channels=None, out_channels=None, layerNorm=Identity,
                 padding_mode='circular', dropout=0.1, activation="relu"):

        super(ResNetBlock, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        hidden_channels = out_channels if hidden_channels is None else hidden_channels

        self.conv1 = setup_conv(in_channels=in_channels,
                                  out_channels=hidden_channels,
                                  kernel_size=kernel_size,
                                  padding_mode=padding_mode)

        n_out_conv2 = out_channels
        self.conv2 = setup_conv(in_channels=hidden_channels,
                                  out_channels=n_out_conv2,
                                  kernel_size=kernel_size,
                                  padding_mode=padding_mode)

        self.norm1 = self.norm2 = layerNorm

        self.conv1x1 = paddle.nn.Conv1D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              weight_attr=nn.initializer.XavierUniform())

        self.conv1x1_d = paddle.nn.Conv1D(in_channels=in_channels+n_out_conv2,
                              out_channels=out_channels,
                              kernel_size=1,
                              weight_attr=nn.initializer.XavierUniform())

        self.norm1x1 = layerNorm

        self.dropout1 = paddle.nn.Dropout(dropout)
        self.dropout2 = paddle.nn.Dropout(dropout)

        if activation == "relu":
            self.activation =  paddle.nn.functional.relu
        elif activation == "gelu":
            self.activation =  paddle.nn.functional.gelu
        elif activation == 'tanh':
            self.activation =  paddle.nn.functional.tanh
        else:
            raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
        
        self.se = SELayer(channel=out_channels)

    def forward(self, x, x_mask=None, x_key_padding_mask=None):
        """Pass the input through the encoder layer.

        Parameters
        ----------
        x: tensor
            The input sequence to the encoder layer.
        x_mask: tensor
            Mask for the input sequence (optional).
        x_key_padding_mask: tensor
            Mask for the x keys per batch (optional).
        """
        z = self.dropout1(self.activation(self.norm1(self.conv1(x))))
        z = self.dropout2(self.activation(self.norm2(self.conv2(z))))
        if z.shape[1] == x.shape[1]:
            x = self.norm1x1(self.conv1x1(x + z))
        else:
            x = self.norm1x1(self.conv1x1_d(paddle.concat((x,z),axis=1)))
        x = self.se(x)
        return x


class SEResNet(paddle.nn.Layer):
    
    def __init__(self, n_filters_ks3, n_filters_ks1=None, n_channels_in=1, n_channels_out=1, 
                 padding_mode='zeros', direct_shortcut=False, 
                 layerNorm=Identity, dropout=0.1):
        
        kernel_size = 4 

        super(SEResNet, self).__init__()
        
        self.n_filters_ks1 = [ [] for i in range(len(n_filters_ks3)+1) ] if n_filters_ks1 is None else n_filters_ks1
        assert len(self.n_filters_ks1) == len(n_filters_ks3) + 1
        
        self.direct_shortcut = direct_shortcut

        self.layers4x4 = []            
        self.layers_ks1 = [ [] for i in range(len(self.n_filters_ks1))]
        self.selayers = []
        n_in = n_channels_in
        for i in range(len(self.n_filters_ks1)):
            for j in range(len(self.n_filters_ks1[i])):
                n_out = self.n_filters_ks1[i][j]
                block = ResNetBlock(in_channels = n_in, 
                                    kernel_size = kernel_size,
                                    hidden_channels= None, 
                                    out_channels= n_out,
                                    layerNorm=layerNorm,
                                    padding_mode='circular',
                                    dropout=dropout, 
                                    activation="gelu")                
                self.layers_ks1[i].append(block)
                n_in = n_out
                
            if i >= len(n_filters_ks3):
                break

            n_out = n_filters_ks3[i]
            layer = setup_conv(in_channels=n_in,
                               out_channels=n_out,
                               kernel_size=kernel_size,
                               padding_mode=padding_mode)
            self.layers4x4.append(layer)
            n_in = n_out
            
        self.layers4x4 = paddle.nn.LayerList(self.layers4x4)
        self.layers1x1 = sum(self.layers_ks1, [])
        self.layers1x1 = paddle.nn.LayerList(self.layers1x1)
        self.final = paddle.nn.Conv1D(in_channels=n_in,
                                     out_channels=n_channels_out,
                                     kernel_size=1,
                                     weight_attr=nn.initializer.XavierUniform())
        self.nonlinearity = paddle.nn.GELU()

    def forward(self, x):

        if self.direct_shortcut:
            out = x
        for layer in self.layers_ks1[0]:   
            x = self.nonlinearity(layer(x))
               
        for i, layer4x4 in enumerate(self.layers4x4):
            x = self.nonlinearity(layer4x4(x))
            for layer in self.layers_ks1[i+1]:
                x = self.nonlinearity(layer(x))

        if self.direct_shortcut:
            return self.final(x) + paddle.unsqueeze(out[:,0,:], axis=1)
        else:
            return self.final(x)