import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from .L96_utils import as_tensor

class PeriodicConv1D(paddle.nn.Conv1D):
    """ Implementing 1D convolutional layer with circular padding.

    """
    def forward(self, input):
        if self._padding_mode == 'circular':
            expanded_padding_circ = (self._padding // 2, (self._padding - 1) // 2)
            return F.conv1d(F.pad(input, expanded_padding_circ, mode='circular', data_format='NCL'), 
                            self.weight, self.bias, self._stride,
                            (0,), self._dilation, self._groups)
        elif self._padding_mode == 'valid':
            expanded_padding_circ = (self.padding[0] // 2, (self.padding[0] - 1) // 2)
            return F.conv1d(F.pad(input, expanded_padding_circ, mode='constant', value=0., data_format='NCL'), 
                            self.weight, self.bias, self._stride,
                            (0,), self._dilation, self._groups)
        return F.conv1d(input, self.weight, self.bias, self._stride,
                        self._padding, self._dilation, self._groups)

def setup_conv(in_channels, out_channels, kernel_size, padding_mode, stride=1):
    """
    Select between regular and circular 1D convolutional layers.
    padding_mode='circular' returns a convolution that wraps padding around the final axis.
    """
    if padding_mode in ['circular', 'valid']:
        return PeriodicConv1D(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=kernel_size,
                      stride=stride,
                      padding_mode=padding_mode,
                      weight_attr=nn.initializer.XavierUniform())
    else:
        return paddle.nn.Conv1D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=(kernel_size-1)//2,
                              stride=stride,
                              weight_attr=nn.initializer.XavierUniform())

class Model_forwarder_rk4default(paddle.nn.Layer):

    def __init__(self, model, dt):
        super(Model_forwarder_rk4default, self).__init__()            
        self.dt = dt
        self.model = model

    def forward(self, x):
        """ Runke-Katta step with 2/6 rule """
        f0 = self.model.forward(x)
        f1 = self.model.forward(x + self.dt/2.*f0)
        f2 = self.model.forward(x + self.dt/2.*f1)
        f3 = self.model.forward(x + self.dt * f2)

        x = x + self.dt/6. * (f0 + 2.* (f1 + f2) + f3)

        return x

class SELayer(paddle.nn.Layer):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = paddle.nn.AdaptiveAvgPool1D(1)
        self.fc = paddle.nn.Sequential(
            paddle.nn.Linear(channel, channel // reduction, weight_attr=nn.initializer.XavierUniform(), bias_attr=False),
            paddle.nn.ReLU(),
            paddle.nn.Linear(channel // reduction, channel, weight_attr=nn.initializer.XavierUniform(), bias_attr=False),
            paddle.nn.Sigmoid()
        )

    def forward(self, x):
        y = paddle.squeeze(self.avg_pool(x), axis=-1)
        y = paddle.unsqueeze(self.fc(y), axis=-1)
        return x * y.expand_as(x)

class Identity(paddle.nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def weighted_mse(weight, input, label):
    weight = as_tensor(weight)
    tmp = label - input
    tmp_res = paddle.matmul(tmp, weight)
    res = paddle.mean(paddle.matmul(tmp_res, paddle.transpose(tmp, perm=[0, 2, 1])))
    return res