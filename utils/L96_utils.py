import numpy as np
import paddle
import dapper as dpr
from dapper.da_methods.variational import var_method
from scipy.linalg import pinv
import dapper.mods.Lorenz96 as Lorenz96


def as_tensor(x):
    return paddle.to_tensor(x, dtype=paddle.float32)

def predictor_corrector(fun, y0, times, alpha=0.5):

    y = np.zeros((len(times), *y0.shape), dtype=y0.dtype)
    y[0] = y0.copy()
    for i in range(1,len(times)):        
        dt = times[i] - times[i-1]

        f0 = fun(times[i-1], y[i-1]).copy()
        f1 = fun(times[i],   y[i-1] + dt*f0)

        y[i] = y[i-1] + dt * (alpha*f0 + (1.-alpha)*f1)
        
    return y

def rk4_default(fun, y0, times):

    y = np.zeros((len(times), *y0.shape), dtype=y0.dtype)
    y[0] = y0.copy()
    for i in range(1,len(times)):
        dt = times[i] - times[i-1]

        f0 = fun(times[i-1], y[i-1]).copy()
        f1 = fun(times[i-1] + dt/2., y[i-1] + dt*f0/2.).copy()
        f2 = fun(times[i-1] + dt/2., y[i-1] + dt*f1/2.).copy()
        f3 = fun(times[i],   y[i-1] + dt*f2).copy()

        y[i] = y[i-1] + dt/6. * (f0 + 2.*f1 + 2.*f2 + f3)
        
    return y