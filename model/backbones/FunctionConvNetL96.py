import paddle
import numpy as np
from dapper.mods.Lorenz96 import Force

class FunctionConvNetL96(paddle.nn.Layer):

    def __init__(self):
        
        super(FunctionConvNetL96, self).__init__()

    def shift(self, x, n):
        return paddle.roll(x, -n, axis=-1)

    def dxdt_autonomous(self, x):
        return (self.shift(x, 1)-self.shift(x, -2))*self.shift(x, -1) - x

    def dxdt(self, x):
        return self.dxdt_autonomous(x) + Force

    def forward(self, x):

        assert len(x.shape) == 3 # (N, J+1, K), J 'channels', K locations

        dxdt = self.dxdt(x)

        return dxdt