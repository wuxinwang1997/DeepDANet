import paddle
import numpy as np
import paddle.nn.functional as F
from utils.L96_utils import as_tensor


def nll_loss(output, target):
    return F.nll_loss(output, target)

def mean_absolute_percentage_error(y_true, y_pred):
    diff = paddle.abs((y_true - y_pred) / paddle.clip(paddle.abs(y_true),
                                            paddle.epsilon(),
                                            None))
    return 100. * paddle.mean(diff, axis=-1)

def loss_function(loss_fun, extra_args={}):
    if loss_fun == 'mse':
        return F.mse_loss
    elif loss_fun == "mape":
        return mean_absolute_percentage_error

    elif loss_fun == 'local_mse':
        n_local = extra_args['n_local']
        pad_local = extra_args['pad_local']
        assert len(pad_local) == 2 # left and right padding along L96 ring of locations

        def local_mse(inputs, targets):
            assert len(inputs.shape)==3
            error = inputs[..., n_local*pad_local[0]:-n_local*pad_local[1]] - targets
            local_mse = paddle.sum((error)**2) / inputs.shape[0]            
            return local_mse
    
    elif loss_fun == 'lat_mse':
        # Copied from weatherbench fork of S. Rasp: 
        weights_lat = np.cos(np.deg2rad(extra_args['lat']))
        weights_lat /= weights_lat.mean()
        weights_lat = paddle.tensor(weights_lat, requires_grad=False)

        def weighted_mse(in1, in2):
            error = in1 - in2
            weighted_mse = (error)**2 * weights_lat[None, None , :, None]
            weighted_mse = paddle.sum(weighted_mse) / in1.shape[0]            
            return weighted_mse

        return weighted_mse

    else:
        raise NotImplementedError()