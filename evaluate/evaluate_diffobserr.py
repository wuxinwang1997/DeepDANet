from random import shuffle
import numpy as np
import paddle
import sys
sys.path.append('./')
from utils import as_tensor, simulate, mkdir_from_path, str2bool
from model import named_da_network
from model.backbones import SEResNet
from configargparse import ArgParser
import numpy as np
import os
from mpl_tools import is_notebook_or_qt as nb
import dapper as dpr
import dapper.da_methods as da
import dapper.tools.progressbar as pb
import dapper.mods as modelling
import dapper.mods.Lorenz96 as Lorenz96
from dapper.tools.localization import nd_Id_localization
from dapper.da_methods import da_method
from dapper.tools.progressbar import progbar
import logging
import logging.config
from pathlib import Path
from logger.logger import get_logger, setup_logging
paddle.device.set_device("cpu")
from utils.model_utils import DeepDANet
import time

def run_exp(exp_id, res_dir, K, T, lead_time, seq_length,            
            da_model_name, with_forecast, obs_num, da_method, 
            da_N, da_Inf, da_Bx, da_rot, da_loc_rad, da_xN, da_Lag, obserr, **net_kwargs):

    res_dir = Path(res_dir)
    log_dir = res_dir / 'da_log' / exp_id / 'evaluate_forecast_steps' /str(with_forecast) 
    if not os.path.exists(str(log_dir)):
        log_dir.mkdir(parents=True)
    logger = get_logger(log_dir, "debug")
    logger.info('evaluate results are saved at ' + str(log_dir))

    ## define model
    save_da_dir = './' + str(res_dir) + '/da_models/' + exp_id + '/with_forecast/' + str(with_forecast)
    print('net_kwargs', net_kwargs)

    da_model_fn = f'{exp_id}_dt{lead_time}.pt'

    da_model = named_da_network(da_model_name=da_model_name,
                                n_input_channels=obs_num+1, 
                                n_output_channels=1,
                                seq_length=seq_length,
                                **net_kwargs)
    
    test_input = np.random.normal(size=(10, obs_num+1, K))

    print(f'model output shape to test input of shape {test_input.shape}',
        da_model(as_tensor(test_input)).shape)
    print('total #parameters: ', np.sum([np.prod(item.shape) for item in da_model.state_dict().values()]))
    
    ## train model
    print('loading model from disk')
    da_model.set_state_dict(paddle.load(save_da_dir + '/' + da_model_fn))
#     summary(da_model, (obs_num+1, K))
    mlxa = []
    mlxb = []
    truths = []
    classic_xa = []
    classic_xb = []
    err_xa_c = []
    err_xb_c = []
    err_xa_n = []
    err_xb_n = []
    time_c = []
    time_n = []
    for k in range(10):
        # Sakov uses K=300000, BurnIn=1000*0.05
        OneYear = 0.05 * (24/6) * 365
        tseq = modelling.Chronology(0.01, dto=0.05, T=T*OneYear, BurnIn=2*Lorenz96.Tplot)

        Nx = K
        x0 = Lorenz96.x0(Nx)

        Dyn = {
            'M': Nx,
            'model': Lorenz96.step,
            'linear': Lorenz96.dstep_dx,
            'noise': 0,
        }

        jj = np.arange(Nx)  # obs_inds
        Obs = modelling.partial_Id_Obs(Nx, jj)
        Obs['noise'] = obserr
        Obs['localizer'] = nd_Id_localization((Nx,), (2,))

        xp  = DeepDANet(model=da_model)
        xp.seed = 2022
        X0 = modelling.GaussRV(mu=x0, C=0.001)
        HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
        HMM.liveplotters = Lorenz96.LPs(jj)
        xx, yy, yy_ = simulate(HMM)
        obs = np.zeros(shape=(tseq.Ko, obs_num, Nx), dtype=np.float32)
        
        for j in range(obs_num):
            obs[:,j,:] = yy_[HMM.tseq.dko+j:-1:HMM.tseq.dko,:]

        HMM.tseq.Ko = HMM.tseq.Ko-1
        print(yy.shape)
        print(obs.shape)
        start = time.time()
        xp.assimilate(HMM, xx[:-HMM.tseq.dko], obs) #, liveplots=not nb)
        end = time.time()
        DeepDANet_time = end - start
        err_xa_n.append(np.sqrt(np.mean((xp.stats.mu.a-xx[HMM.tseq.dko:-HMM.tseq.dko:HMM.tseq.dko,:])**2)))
        err_xb_n.append(np.sqrt(np.mean((xp.stats.mu.f-xx[HMM.tseq.dko:-HMM.tseq.dko:HMM.tseq.dko,:])**2)))
        print('exp: ', k, 'DeepDANet analysis rmse: ', np.sqrt(np.mean((xp.stats.mu.a-xx[HMM.tseq.dko:-HMM.tseq.dko:HMM.tseq.dko,:])**2)))
        print('exp: ', k, 'DeepDANet forecast rmse: ', np.sqrt(np.mean((xp.stats.mu.f-xx[HMM.tseq.dko:-HMM.tseq.dko:HMM.tseq.dko,:])**2)))
        print('exp: ', k, 'DeepDANet loop time: ', DeepDANet_time)
        truths.append(xx[HMM.tseq.dko:-HMM.tseq.dko:HMM.tseq.dko,:])
        mlxb.append(xp.stats.mu.f)
        mlxa.append(xp.stats.mu.a)
        time_n.append(DeepDANet_time)
        
        
        if da_method == 'Var3D':
            xp = da.Var3D(xB=da_Bx)
        if da_method == 'Var4D':
            xp = da.Var4D(Lag=1,xB=da_Bx)
        if da_method == 'DEnKF':
            xp = da.EnKF('DEnKF',N=da_N, infl=da_Inf, rot=da_rot)
        if da_method == 'EnKF':
            xp = da.EnKF('Sqrt',N=da_N, infl=da_Inf, rot=da_rot)
        if da_method == 'LETKF':
            xp = da.LETKF(N=da_N , infl=da_Inf, rot=da_rot, loc_rad=da_loc_rad)
        if da_method == 'iEnKS':
            xp = da.iEnKS('Sqrt', N=da_N, Lag=da_Lag, infl=da_Inf, rot=da_rot, xN=da_xN)
        xp.seed = 2022
        HMM.X0=X0
        start = time.time()
        xp.assimilate(HMM, xx[:-HMM.tseq.dko], yy[:-1]) #, liveplots=not nb)
        end = time.time()
        if da_method == 'iEnKS':
            xa = xp.stats.mu.s
        else:
            xa = xp.stats.mu.a
        xb = xp.stats.mu.f
        err_xa_c.append(np.sqrt(np.mean((xa-xx[HMM.tseq.dko:-HMM.tseq.dko:HMM.tseq.dko,:])**2)))
        err_xb_c.append(np.sqrt(np.mean((xb-xx[HMM.tseq.dko:-HMM.tseq.dko:HMM.tseq.dko,:])**2)))
        time_c.append(end - start)
        print('exp: ', k, f'{da_method} analysis rmse: ', np.sqrt(np.mean((xa-xx[HMM.tseq.dko:-HMM.tseq.dko:HMM.tseq.dko,:])**2)))
        print('exp: ', k, f'{da_method} forecast rmse: ', np.sqrt(np.mean((xb-xx[HMM.tseq.dko:-HMM.tseq.dko:HMM.tseq.dko,:])**2)))
        print('exp: ', k, f'{da_method} loop time: ', time_c[k])
        classic_xb.append(xp.stats.mu.f)
        classic_xa.append(xp.stats.mu.a)

    logger.info(f'DeepDANet analysis rmse (mean): {np.mean(err_xa_n)}, DeepDANet analysis rmse (std): {np.std(err_xa_n)}')
    logger.info(f'DeepDANet forecast rmse (mean): {np.mean(err_xb_n)}, DeepDANet analysis rmse (std): {np.std(err_xb_n)}')
    logger.info(f'DeepDANet da time (mean): {np.mean(time_n)}, da time time (std): {np.std(time_n)}')
    logger.info(f'{da_method} analysis rmse (mean): {np.mean(err_xa_c)}, {da_method} analysis rmse (std): {np.std(err_xa_c)}')
    logger.info(f'{da_method} forecast rmse (mean): {np.mean(err_xb_c)}, {da_method} analysis rmse (std): {np.std(err_xb_c)}')
    logger.info(f'{da_method} da time (mean): {np.mean(time_c)}, {da_method} da time (std): {np.std(time_c)}')

def setup(conf_exp=None):
    p = ArgParser()
    p.add_argument('-c', '--conf-exp', is_config_file=True, help='config file path', default=conf_exp)
    p.add_argument('--exp_id', type=str, required=True, help='experiment id')
    p.add_argument('--datadir', type=str, required=True, help='path to data')
    p.add_argument('--res_dir', type=str, required=True, help='path to results')
    p.add_argument('--gpus', type=int, default='0', help='how many gpus to use')
    p.add_argument('--only_eval', type=str2bool, default=False, help='if to evaulate saved model (=False for training)')

    p.add_argument('--K', type=int, required=True, help='number of slow variables (grid cells)')
    p.add_argument('--K_local', type=int, default=-1, help='number of slow variables (grid cells) in local training region')
    p.add_argument('--n_local', type=int, default=0, help='number of local training regions needed for single update step')
    p.add_argument('--T', type=int, required=True, help='length of simulation data (in time units [s])')
    p.add_argument('--dt', type=float, required=True, help='simulation step length (in time units [s])')
    p.add_argument('--N_trials', type=int, default=1, help='number of random starting points for solver')

    p.add_argument('--lead_time', type=int, required=True, help='forecast lead time (in time steps)')
    p.add_argument('--seq_length', type=int, default=1, help='length of input state sequence to network')

    p.add_argument('--loss_fun', type=str, default='mse', help='loss function for model training')
    p.add_argument('--batch_size', type=int, default=32, help='batch-size')
    p.add_argument('--batch_size_eval', type=int, default=-1, help='batch-size for evaluation dataset')
    p.add_argument('--max_epochs', type=int, default=2000, help='epochs')
    p.add_argument('--max_patience', type=int, default=None, help='patience for early stopping')
    p.add_argument('--eval_every', type=int, default=None, help='frequency for checking convergence (in minibatches)')
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    p.add_argument('--lr_min', type=float, default=1e-6, help='minimal learning rate after which stop reducing')
    p.add_argument('--lr_decay', type=float, default=1., help='learning rate decay factor')
    p.add_argument('--max_lr_patience', type=int, default=None, help='patience per learning rate plateau')
        
    p.add_argument('--da_model_name', type=str, required=True, help='designator for neural network')
    p.add_argument('--padding_mode', type=str, default='circular', help='designator for padding mode')
    p.add_argument('--filters', type=int, nargs='+', required=True, help='filter count per layer or block')
    p.add_argument('--kernel_sizes', type=int, nargs='+', required=True, help='kernel sizes per layer or block')
    p.add_argument('--filters_ks1_init', type=int, nargs='+', required=False, help='initial 1x1 convs for network')
    p.add_argument('--filters_ks1_inter', type=int, nargs='+', required=False, help='intermediate 1x1 convs for network')
    p.add_argument('--filters_ks1_final', type=int, nargs='+', required=False, help='final 1x1 convs for network')
    p.add_argument('--additiveResShortcuts', default=None, help='boolean or None, if ResNet shortcuts are additive')
    p.add_argument('--direct_shortcut', type=str2bool, default=False, help='if model has direct input-output residual connection')
    p.add_argument('--weight_decay', type=float, default=0., help='weight decay (L2 norm)')
    p.add_argument('--dropout_rate', type=float, default=0., help='Dropout')
    p.add_argument('--layerNorm', type=str, default='BN', help='normalization layer for some network architectures')
    p.add_argument('--init_net', type=str, default='rand', help='string specfifying weight initialization for some models')
    p.add_argument('--K_net', type=int, default=1, help='number of slow variables (grid cells) for some models')
    p.add_argument('--J_net', type=int, default=0, help='number of fast variables (vertical levels) for some models')
    p.add_argument('--F_net', type=float, default=10., help='number of fast variables (vertical levels) for some models')
    p.add_argument('--model_forwarder', type=str, default='predictor_corrector', help='numerical solver scheme for model')       
    p.add_argument('--dt_net', type=float, default='0.', help='numerical integration time step for some models')
    p.add_argument('--alpha_net', type=float, default='0.5', help='predictor-corrector parameter for some models')

    p.add_argument('--exp_f_id', type=str, required=True, help='experiment id')
    p.add_argument('--f_K_local', type=int, default=-1, help='number of slow variables (grid cells) in local training region')
    p.add_argument('--f_n_local', type=int, default=0, help='number of local training regions needed for single update step')
    p.add_argument('--f_N_trials', type=int, default=1, help='number of random starting points for solver')

    p.add_argument('--f_lead_time', type=int, required=True, help='forecast lead time (in time steps)')
    p.add_argument('--f_seq_length', type=int, default=1, help='length of input state sequence to network')
    
    p.add_argument('--f_model_name', type=str, required=True, help='designator for neural network')
    p.add_argument('--f_padding_mode', type=str, default='circular', help='designator for padding mode')
    p.add_argument('--f_filters', type=int, nargs='+', required=True, help='filter count per layer or block')
    p.add_argument('--f_kernel_sizes', type=int, nargs='+', required=True, help='kernel sizes per layer or block')
    p.add_argument('--f_filters_ks1_init', type=int, nargs='+', required=False, help='initial 1x1 convs for network')
    p.add_argument('--f_filters_ks1_inter', type=int, nargs='+', required=False, help='intermediate 1x1 convs for network')
    p.add_argument('--f_filters_ks1_final', type=int, nargs='+', required=False, help='final 1x1 convs for network')
    p.add_argument('--f_additiveResShortcuts', default=None, help='boolean or None, if ResNet shortcuts are additive')
    p.add_argument('--f_direct_shortcut', type=str2bool, default=False, help='if model has direct input-output residual connection')
    p.add_argument('--f_weight_decay', type=float, default=0., help='weight decay (L2 norm)')
    p.add_argument('--f_dropout_rate', type=float, default=0., help='Dropout')
    p.add_argument('--f_layerNorm', type=str, default='BN', help='normalization layer for some network architectures')
    p.add_argument('--f_init_net', type=str, default='rand', help='string specfifying weight initialization for some models')
    p.add_argument('--f_K_net', type=int, default=1, help='number of slow variables (grid cells) for some models')
    p.add_argument('--f_J_net', type=int, default=0, help='number of fast variables (vertical levels) for some models')
    p.add_argument('--f_F_net', type=float, default=10., help='number of fast variables (vertical levels) for some models')
    p.add_argument('--f_model_forwarder', type=str, default='predictor_corrector', help='numerical solver scheme for model')       
    p.add_argument('--f_dt_net', type=float, default='0.', help='numerical integration time step for some models')
    p.add_argument('--f_alpha_net', type=float, default='0.5', help='predictor-corrector parameter for some models')
    p.add_argument('--with_forecast', type=int, default=0, help='train model with forecast model')
    
    p.add_argument('--obs_num', type=int, default=1, help='the number of observations in a DAW')
    p.add_argument('--da_method', type=str, default='EnKF', help='the method to assimilate')
    p.add_argument('--da_N', type=int, default=20, help='the number of ensemble to do En assimilate')
    p.add_argument('--da_Inf', type=float, default=1., help='the inlation of B matrix')
    p.add_argument('--da_Bx', type=float, default=0.1, help='the B matrix of assimilation method')
    p.add_argument('--da_rot', type=str2bool, default=True, help='the rotation of assimilation method')
    p.add_argument('--da_loc_rad', type=int, default=4, help='the localization rad of assimilation method')
    p.add_argument('--da_xN', type=int, default=1, help='the xN of iEnKS')
    p.add_argument('--da_Lag', type=int, default=1, help='the Lag of iEnKS')
    p.add_argument('--obserr', type=float, default=1.0, help='obserr')
    args = p.parse_args() if conf_exp is None else p.parse_args(args=[])
    return vars(args)

if __name__ == '__main__':
    args = setup()
    args.pop('conf_exp')
    print(args)
    run_exp(**args)
