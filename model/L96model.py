from base import BaseModel
import numpy as np
from .backbones import  FunctionConvNetL96
from utils.model_utils import Model_forwarder_rk4default

def named_network(**kwargs):

    dt = kwargs['f_dt_net']

    model = FunctionConvNetL96()

    if kwargs['f_model_forwarder'] == 'rk4_default':
        model_forwarder = Model_forwarder_rk4default(model, dt=dt)

    return model, model_forwarder