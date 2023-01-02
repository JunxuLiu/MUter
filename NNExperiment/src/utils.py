from cProfile import label
import os
from builtins import ValueError
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from Recorder import Recorder


def paramters_to_vector(params):

    vec = []

    for param in params:
        vec.append(param.view(-1))
    
    return torch.cat(vec)

def vector_to_parameters(vec, paramters):

    pointer = 0

    for param in paramters:
        num_param = param.numel()
        param.data = vec[pointer : pointer + num_param].view_as(param).data
        pointer += num_param
    
def total_param(model):
    
    number = 0
    
    for param in model.parameters():
        number = number + np.prod(list(param.shape))

    return number

class DataPreProcess:

    def __init__(self, args):
        
        self.args = args
    
    def processing(self, image):

        if self.args.dataset == 'Cifar100':
            return image * 2 - 1
        elif self.args.dataset == 'ImageNet':
            return image * 2 - 1
        elif self.args.dataset == 'Lacuna-100':
            return image 

def get_cg_iters(args):

    if args.dataset == 'ImageNet':
        return 20
    elif args.dataset == 'Lacuna-100':
        return 10
            
def get_layers(str, input_features=640, output_features=10, isBias=False):

    if str == 'linear':
        return nn.Linear(in_features=input_features, out_features=output_features, bias=isBias)
    elif str == 'MLP':
        return nn.Sequential(
            nn.Linear(in_features=input_features, out_features=100, bias=isBias),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=output_features, bias=isBias)
        )
    else:
        raise Exception('No such method called {}, please recheck !'.format(str))

def cg_solve(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-5, x_init=None):
    """
    Goal: Solve Ax=b equivalent to minimizing f(x) = 1/2 x^T A x - x^T b
    Assumption: A is PSD, no damping term is used here (must be damped externally in f_Ax)
    Algorithm template from wikipedia
    Verbose mode works only with numpy
    """
       
    if type(b) == torch.Tensor:
        x = torch.zeros(b.shape[0]) if x_init is None else x_init
        x = x.to(b.device)
        if b.dtype == torch.float16:
            x = x.half()
        r = b - f_Ax@x
        p = r.clone()
    elif type(b) == np.ndarray:
        x = np.zeros_like(b) if x_init is None else x_init
        r = b - f_Ax(x)
        p = r.copy()
    else:
        print("Type error in cg")

    fmtstr = "%10i %10.3g %10.3g %10.3g"
    titlestr = "%10s %10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm", "obj fn"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
            norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
            print(fmtstr % (i, r.dot(r), norm_x, obj_fn))

        rdotr = r.dot(r)
        Ap = f_Ax@p
        alpha = rdotr/(p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = r.dot(r)
        beta = newrdotr/rdotr
        p = r + beta * p

        if newrdotr < residual_tol:
            # print("Early CG termination because the residual was small")
            print('this is {}, i stoped !!!'.format(i))
            break

    if callback is not None:
        callback(x)
    if verbose: 
        obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
        norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
        print(fmtstr % (i, r.dot(r), norm_x, obj_fn))
    return x



def get_random_sequence(total_lenth, resort_lenth, seed=None, isSort=True):

    if seed != None:
        random.seed(seed)
    
    resort_sequence = random.sample(range(0, total_lenth), resort_lenth)
    if isSort:
        resort_sequence.sort()
    another_sequence = [i for i in range(total_lenth) if i not in resort_sequence]
    random_sequence = np.concatenate([resort_sequence, another_sequence])

    if len(random_sequence) != total_lenth:
        raise Exception('Random sequence error !')
    
    return list(random_sequence)

def get_BatchRemove_sequence(args, isPretrain=False):

    def Add_pre_zero(isPretrain, arr):
        if isPretrain:
            arr.insert(0, 0)
        return arr

    if args.isBatchRemove == 0:
        if args.dataset in ['ImageNet', 'Cifar100', 'Cifar10']:
            return Add_pre_zero(isPretrain, [1, 200, 500, 1000, 2000, 4000])
        elif args.dataset == 'Lacuna-100':
            return Add_pre_zero(isPretrain, [1, 20, 50, 100, 200, 400])
        else:
            raise ValueError
    elif args.isBatchRemove == 1:
        if args.dataset in ['ImageNet', 'Cifar100', 'Cifar10']:
            return Add_pre_zero(isPretrain, [2500, 5000, 7500, 10000])
        elif args.dataset == 'Lacuna-100':
            return Add_pre_zero(isPretrain, [220, 440, 660, 880])
        else:
            raise ValueError

def get_pretrain_model_path(str):

    if str == 'ImageNet':
        return 'data/model/pretrain_model/imagenet_wrn_baseline_epoch_99.pt'
    elif str == 'Cifar100':
        return 'data/model/pretrain_model/cifar100_wrn34_model_epoch_80.pt'
    elif str == 'Lacuna-100':
        return 'data/model/pretrain_model/Lacuna-100_wrn28_model_epoch80.pt'
    else:
        raise ValueError

def get_goal_dataset(str):
     
    if str in ['ImageNet', 'Cifar100']:
        return 'Cifar10'
    elif str == 'Lacuna-100':
        return 'Lacuna-10'
    else:
        raise ValueError

def generate_save_name(args, remain_head):
    str = ''
    
    if args.adv_type == 'FGSM':
        str += 'FGSM_'
    else:
        str += 'PGD_'
    
    if args.isBatchRemove == 0 or args.isBatchRemove == 3:
        str += 'Schur_'
    else:
        str += 'Batch_'
    
    str += 'model_ten_{}_times{}'.format(remain_head, args.times)
    print('The name is : {}'.format(str))
    return str



