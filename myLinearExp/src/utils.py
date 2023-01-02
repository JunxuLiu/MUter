import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pathlib
from dataloder import *
from torchattacks import PGD
from collections import OrderedDict
from functorch import vmap

class LossFunction(torch.nn.Module):
    def __init__(self, model_type):
        super(LossFunction, self).__init__()
        self.model_type = model_type
        
    def forward(self, output, label):
        if self.model_type == 'logistic':
            loss_fn = nn.BCELoss()
            return loss_fn(output.to(torch.float32), label.to(torch.float32))
            
        else: # ridge
            return torch.tensor(0.5) * (((output - label)**2 ).mean())

def get_featrue(args):
    featrue_dict = {
        'binaryMnist':784, 
        'covtype':54, 
        'epsilon':2000, 
        'gisette':5000, 
        'ijcnn1':22, 
        'higgs':28,  
        'madelon':500, 
        'phishing':68, 
        'splice':60,
    }
    return featrue_dict[args.dataset]

def vec_param(parameters):
    vec = []
    for param in parameters:
        vec.append(param.view(-1))
    return torch.cat(vec).unsqueeze(dim=1)

def training_param(args, isAttack=False, isPGD=False):
    """
    return training epochs, eps, alpha...
    for logistic phishing(epoch 500, lr 0.01, neumanan 3, eps 10/255, 0.005, 12)
    """
    epoch_dict = {
        'binaryMnist':100, 
        'covtype':15, 
        'epsilon':30, 
        'gisette':150, 
        'ijcnn1':50, 
        'higgs':10, 
        'madelon':1000, 
        'phishing':200, 
        'splice': 1000,
    }
    lr_dict = {
        'binaryMnist':0.01, 
        'covtype':0.1, 
        'epsilon':0.01, 
        'gisette':0.01, 
        'ijcnn1':0.01, 
        'higgs':0.01, 
        'madelon':0.01, 
        'phishing':0.01, 
        'splice':0.01
    }

    ridge_lr_dict = {
        'binaryMnist':0.01, 
        'covtype':0.1, 
        'epsilon':0.001, 
        'gisette':0.01, 
        'ijcnn1':0.01, 
        'higgs':0.01, 
        'madelon':0.01, 
        'phishing':0.01, 
        'splice':0.01
    }
    
    ### for logistic
    # for FGSM (epsilon, steps, alpha) 
    fgsm_dict = {
        'binaryMnist':(64/255, 64/255, 1), 
        'covtype':(4/255, 4/255, 1), 
        'epsilon':(2/255, 2/255, 1), 
        'gisette':(16/255, 16/255, 1), 
        'ijcnn1':(16/255, 16/255, 1),
        'higgs':(1/255, 1/255, 1), 
        'madelon':(2/255, 2/255, 1), 
        'phishing':(10/255, 10/255, 1), 
        'splice':(20/255, 20/255, 1)
    }
    
    # for PGD (epsilon, steps, alpha)
    pgd_dict = {
        'binaryMnist':(64/255, 8/255, 15), 
        'covtype':(4/255, 0.004, 7), 
        'epsilon':(2/255, 0.002, 7), 
        'gisette':(16/255, 0.01, 12), 
        'ijcnn1':(16/255, 0.01, 12),
        'higgs':(1/255, 0.001, 7), 
        'madelon':(2/255, 0.001, 12), 
        'phishing':(10/255, 0.005, 12), 
        'splice':(20/255, 0.01, 12)
    }

    ### for ridge
    # for FGSM (epsilon, steps, alpha) 
    ridge_fgsm_dict = {
        'binaryMnist':(64/255, 64/255, 1), 
        'covtype':(4/255, 4/255, 1), 
        'epsilon':(2/255, 2/255, 1), 
        'gisette':(16/255, 16/255, 1), 
        'ijcnn1':(16/255, 16/255, 1),
        'higgs':(1/255, 1/255, 1), 
        'madelon':(2/255, 2/255, 1), 
        'phishing':(6/255, 6/255, 1), 
        'splice':(20/255, 20/255, 1)
    }
    
    # for PGD (epsilon, steps, alpha)
    ridge_pgd_dict = {
        'binaryMnist':(64/255, 8/255, 15), 
        'covtype':(4/255, 0.004, 7), 
        'epsilon':(2/255, 0.002, 7), 
        'gisette':(16/255, 0.01, 12), 
        'ijcnn1':(16/255, 0.01, 12),
        'higgs':(1/255, 0.001, 7), 
        'madelon':(2/255, 0.001, 12), 
        'phishing':(6/255, 0.75/255, 12), 
        'splice':(20/255, 0.01, 12)
    }


    if isAttack == True:
        if isPGD == False:
            return lr_dict[args.dataset], epoch_dict[args.dataset], fgsm_dict[args.dataset]
        else:
            return lr_dict[args.dataset], epoch_dict[args.dataset], pgd_dict[args.dataset]

    if args.model == 'logistic':
        if args.adv == 'FGSM':
            return lr_dict[args.dataset], epoch_dict[args.dataset], fgsm_dict[args.dataset]
        elif args.adv == 'PGD':
            return lr_dict[args.dataset], epoch_dict[args.dataset], pgd_dict[args.dataset]
        else: # CLEAN
            return lr_dict[args.dataset], epoch_dict[args.dataset], (0.0, 0.0, 0)
    else:
        if args.adv == 'FGSM':
            return ridge_lr_dict[args.dataset], epoch_dict[args.dataset], ridge_fgsm_dict[args.dataset]
        elif args.adv == 'PGD':
            return ridge_lr_dict[args.dataset], epoch_dict[args.dataset], ridge_pgd_dict[args.dataset]
        else: # CLEAN
            return ridge_lr_dict[args.dataset], epoch_dict[args.dataset], (0.0, 0.0, 0)

def get_iters(args):
    dicter = {
        'binaryMnist': 10,
        'covtype': 20,
        'epsilon': 20,
    }
    return dicter[args.dataset]

def total_param(model):
    number = 0
    for param in model.parameters():
        number = number + np.prod(list(param.shape))
    print('total param : {}'.format(number))

    return number

# old version
def loss_grad(weight, x, y, args):
    """
    weight : [m, 1]
    x : [m, 1]
    y : [1, 1]
    """
    device = 'cuda'
    x = x.to(device)
    y = y.to(device)
    weight = weight.to(device)

    if args.model == 'logistic':
        return ((torch.sigmoid(y*(x.t().mm(weight)))-1)*y*x + (args.lam*weight)).detach()
    elif args.model == 'ridge':
        return ((x.t().mm(weight)-y)*x + (args.lam*weight)).detach()
    else:
        raise Exception('no such loss function, please recheck !')

# old version
def hessian(x, y, weight, args):
    """
    x: [m, 1]
    y: [m, 1]
    weight: [m, 1]
    return partial_ww
    """
    device = 'cuda'
    x = x.to(device)
    y = y.to(device)
    weight = weight.to(device)
    x_size = x.shape[0]
    weight_size = weight.shape[0]
    if args.model == 'logistic':
        z = torch.sigmoid(y*(x.t().mm(weight)))
        D = z * (1 - z)
        partial_ww = (D * (x.mm(x.t()))) + (args.lam * torch.eye(weight_size)).to(device)
        return partial_ww.detach()

    elif args.model == 'ridge':
        partial_ww = (x.mm(x.t())) + (args.lam * torch.eye(weight_size)).to(device)
        return partial_ww.detach()
    
    else:
        raise Exception('no such loss function, please recheck !')

# old version
def partial_hessian(x, y, weight, public_partial_xx_inv, args, isUn_inv=False, public_partial_xx=None):
    """
    x: [m, 1]
    y: [1, 1]   
    weight: [m, 1]
    return Dww, partial_ww, partial_wx, partial_xx_inv, patial_xw

    if isUn_inv == True, when we do unleaning satge, we can avoid the inverse operation.
    then return Dww, _ww, _wx, xx_inv, xw, xx
    """
    device = 'cuda'
    x = x.to(device)
    y = y.to(device)
    weight = weight.to(device)
    public_partial_xx_inv = public_partial_xx_inv.to(device)
    x_size = x.shape[0]
    weight_size = weight.shape[0]

    if args.model == 'logistic':

        z = torch.sigmoid(y*(x.t().mm(weight)))
        D = z * (1 - z)
        partial_ww = (D * (x.mm(x.t()))) + (args.lam * torch.eye(weight_size)).to(device)
        partial_wx = (D * (x.mm(weight.t()))) + ((z-1) * y * torch.eye(x_size).to(device))
        partial_xx_inv = (1/D) * public_partial_xx_inv
        #partial_xx_inv = D * public_partial_xx_inv # to verify is right
        partial_xw = (D * (weight.mm(x.t()))) + ((z-1) * y * torch.eye(weight_size).to(device))
        Dww = partial_ww - (partial_wx.mm(partial_xx_inv.mm(partial_xw)))
        if isUn_inv == False:
            return Dww.detach(), partial_ww.detach(), partial_wx.detach(), partial_xx_inv.detach(), partial_xw.detach()
        else:
            public_partial_xx = public_partial_xx.to(device)
            partial_xx = D * public_partial_xx
            return Dww.detach(), partial_ww.detach(), partial_wx.detach(), partial_xx_inv.detach(), partial_xw.detach(), partial_xx.detach()

    elif args.model == 'ridge':
        partial_ww = (x.mm(x.t())) + (args.lam * torch.eye(weight_size)).to(device)
        partial_wx = (x.mm(weight.t())) + (weight.t().mm(x)-y)*torch.eye(x_size).to(device)
        partial_xx_inv = public_partial_xx_inv
        partial_xw = (weight.mm(x.t())) + (weight.t().mm(x)-y)*torch.eye(x_size).to(device)
        Dww = partial_ww - (partial_wx.mm(partial_xx_inv.mm(partial_xw)))
        if isUn_inv == False:
            return Dww.detach(), partial_ww.detach(), partial_wx.detach(), partial_xx_inv.detach(), partial_xw.detach()
        else:
            partial_xx = public_partial_xx.to(device)
            return Dww.detach(), partial_ww.detach(), partial_wx.detach(), partial_xx_inv.detach(), partial_xw.detach(), partial_xx.detach()

    else:
        raise Exception('no such loss function, please recheck !')


def derive_inv(matrix, method='Neumann', iter=3, coefficient=1e-5):
    """
    test differential method: Neumann, rank1, damp+inv, pinverse, Neu_damp(Neuman + damp)
    """
    device = 'cuda'
    if method == 'Neumann':
        return Neumann(matrix, iter)
    elif method == 'rank1':
        return rank1_pinverse(matrix)
    elif method == 'damp':
        damp_I = coefficient * torch.eye(matrix.shape[0]).to(device).detach()
        matrix = matrix + damp_I
        return matrix.inverse()
    elif method == 'pinverse':
        return matrix.pinverse()
    elif method == 'Neumann_damp':
        damp_I = coefficient * torch.eye(matrix.shape[0]).to(device).detach()
        matrix = matrix + damp_I
        return Neumann(matrix, iter=10)
    else:
        raise Exception('No such method, please recheck !')

def Neumann(matrix, iter=3):
    device = 'cuda'
    size = matrix.shape[0]
    matrix_inv = torch.zeros((size, size)).to(device).detach()
    I = torch.eye(size).to(device).detach()
    ans = torch.eye(size).to(device).detach()
    # 0+I = I
    # (-I)*(I-ww^T) = (-I+ww^T)
    # I+(-I+ww^T) = ww^T
    # (-I+ww^T)*(I-ww^T) = -(I-ww^T)^2
    # ww^T - (I-ww^T)^2 
    # -(I-ww^T)^2 * ((I-ww^T) + (I-ww^T)^2) = -(I-ww^T)^3 - (I-ww^T)^4
    # ww^T - (I-ww^T)^2 - (I-ww^T)^3 - (I-ww^T)^4
    # ...
    for i in range(iter):
        matrix_inv = matrix_inv + ans
        ans = ans.mm(I - matrix)
    return matrix_inv

def rank1_pinverse(matrix):
    device = 'cuda'
    inner = torch.trace(matrix).to(device).detach()
    return (1 / (inner * inner)) * matrix

def calculate_memory_matrix(model, train_loader, args, method='MUter', isDelta=True):
    """
    according to args.method chose the memory matrix such as [MUter, Newton, Fisher, Influence]
    return matrix shape[w_size, w_size]
    isDelta : point out using perturbed/unperturbed samples to get the memory matrix
    """
    device = 'cuda'
    feature = get_featrue(args)
    matrix = torch.zeros((feature, feature)).to(device)
    model = model.to(device)
    _, _, atk_info = training_param(args)
    atk = PGD(model, eps=atk_info[0], alpha=atk_info[1], steps=atk_info[2], lossfun=LossFunction(args.model), lam=args.lam)

    # calculate the public part and using Neumann Series to approximate the inverse of partial_xx
    weight = vec_param(model.parameters()).detach()
    public_partial_xx = (weight.mm(weight.t())).detach()    
    #public_partial_xx_inv = ((torch.tensor(2.0)*torch.eye(feature)).to(device) - public_partial_xx).detach() 
    #public_partial_xx_inv = ((torch.tensor(2.0)*torch.eye(feature)).to(device) - public_partial_xx + (torch.eye(feature).to(device) - public_partial_xx).mm(torch.eye(feature).to(device) - public_partial_xx)).detach()
    public_partial_xx_inv = derive_inv(public_partial_xx, method='Neumann', iter=args.iterneumann)
    lenth = len(train_loader)
    for index, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        if image.shape[0] != 1:
            raise Exception('the sample must be single pass to calculate the memory matrix !')
            return None
        if isDelta == True:
            image = atk(image, label).to(device)
        Dww, partial_ww, partial_wx, partial_xx_inv, partial_xw = partial_hessian(image.view(feature, 1), label, weight, public_partial_xx_inv, args)

        if method == 'MUter':
            matrix = matrix + Dww.detach()
        elif method == 'Newton':
            matrix = matrix + partial_ww.detach()
        elif method == 'Fisher':
            grad_w = loss_grad(weight, image.view(feature, 1), label, args).to(device)
            fisher_matrix = grad_w.mm(grad_w.t()).detach()
            matrix = matrix + fisher_matrix
        elif method == 'Influence':
            matrix = matrix + partial_ww.detach()
        else:
            raise Exception('no such method called {}, please recheck the method'.format(method))
            return None
        if index % 1000 == 0:
            print('process [{}/{}]'.format(index, lenth))

    str = 'perturbed samples'
    if isDelta == False:
        str = 'unperturbed samples'
    print('calculate the {} method matrix, using {}'.format(method, str))
    
    return matrix




# CPU version (old)
def forming_blocks(M, H_11, H_12, H_22, H_21):
    """
    version for numpy
    M: memory matrix [w_size, w_size]
    H_11: partial_ww [w_size, w_size]
    H_12: partial_wx [w_size, x_size]
    H_22: partial_xx [x_size, x_size]
    H_21: partial_xw [x_size, w_size]

    retrun | M-H_11    H_12 |
           |  H21      H_22 |
    """ 

    A = np.bmat([[M-H_11, H_12], [H_21, -H_22]])
    print('black matrix A shape {}, type {}'.format(A.shape, type(A)))
    return A

# GPU version (old)
def buliding_matrix(M, H_11, H_12, H_22, H_21):
    """
    version for torch
    using gpu without data change between gpu and cpu
    M: memory matrix [w_size, w_size]
    H_11: partial_ww [w_size, w_size]
    H_12: partial_wx [w_size, x_size]
    H_22: partial_xx [x_size, x_size]
    H_21: partial_xw [x_size, w_size]

    retrun | M-H_11    H_12 |
           |  H21      H_22 |
    """
    device = 'cuda'
    A = torch.cat([torch.cat([M-H_11, H_12], dim=1), torch.cat([H_21, H_22], dim=1)], dim=0).to(device)
    print('black matrix A shape {}, type {}'.format(A.shape, type(A)))
    return A.detach()

# to verify the result(partila hessian) is right?
# logistic : pass
# ridge : pass
def test_auto_partial_hessian(model, x, y, args):
    device = 'cuda'
    
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)

    w_size = total_param(model)
    x_size = np.prod(list(x.shape))

    criterion = LossFunction(args.model)
    
    x.requires_grad = True

    output = model(x)
    loss = criterion(output, y)

    ## auto grad to calculate partial hessian
    partial_ww = torch.zeros((w_size, w_size))
    partial_wx = torch.zeros((w_size, x_size))
    partial_xx = torch.zeros((x_size, x_size))
    partial_xw = torch.zeros((w_size, w_size))

    grad_w = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    grad_w = torch.cat([g.view(-1) for g in grad_w])

    grad_x = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)
    grad_x = torch.cat([g.view(-1) for g in grad_x])

    ## calculate partial_ww
    for index, grad_wi in enumerate(grad_w):
        grad_wi_wj = torch.autograd.grad(grad_wi, model.parameters(), create_graph=True)
        grad_wi_wj = torch.cat([g.view(-1) for g in grad_wi_wj])
        partial_ww[index] = grad_wi_wj.detach()

    ## calculate partial_wx
    for index, grad_wi in enumerate(grad_w):
        grad_wi_xj = torch.autograd.grad(grad_wi, x, create_graph=True)
        grad_wi_xj = torch.cat([g.view(-1) for g in grad_wi_xj])
        partial_wx[index] = grad_wi_xj.detach()

    ## calculate partial_xx
    for index, grad_xi in enumerate(grad_x):
        grad_xi_xj = torch.autograd.grad(grad_xi, x, create_graph=True)
        grad_xi_xj = torch.cat([g.view(-1) for g in grad_xi_xj])
        partial_xx[index] = grad_xi_xj.detach()
    
    ## calculate partial_xw
    for index, grad_xi in enumerate(grad_x):
        grad_xi_wj = torch.autograd.grad(grad_xi, model.parameters(), create_graph=True)
        grad_xi_wj = torch.cat([g.view(-1) for g in grad_xi_wj])
        partial_xw[index] = grad_xi_wj.detach()

    return partial_ww.detach(), partial_wx.detach(), partial_xx.detach(), partial_xw.detach()

"""
logistic: pass
ridge : pass
"""
def auto_grad(model, x, y, args):
    device = 'cuda'
    
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)

    w_size = total_param(model)
    x_size = np.prod(list(x.shape))

    criterion = LossFunction(args.model)
    
    x.requires_grad = True

    output = model(x)
    loss = criterion(output, y)


    grad_w = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    grad_w = torch.cat([g.view(-1) for g in grad_w])
    grad_w.unsqueeze_(dim=1)

    return grad_w.detach()

def cg_solve(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10, x_init=None):
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
            break

    if callback is not None:
        callback(x)
    if verbose: 
        obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
        norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
        print(fmtstr % (i, r.dot(r), norm_x, obj_fn))
    return x

def get_delta_w_dict(delta_w, model):
    """
    reshape delta_w to the shape of model
    """
    device = 'cuda'

    delta_w_dict = OrderedDict()
    params_visited = 0
    for k, p in model.named_parameters():
        num_params = np.prod(list(p.shape))
        update_params = delta_w[params_visited:params_visited+num_params]
        delta_w_dict[k] = torch.tensor(update_params).to(device).view_as(p)
        params_visited+=num_params
    return delta_w_dict

def model_distance(modelA, modelB):
    device = 'cuda'
    distance = torch.tensor(0.0).to(device)
    for (paramA, paramB) in zip(modelA.parameters(), modelB.parameters()):
        distance = distance + (paramA.data - paramB.data).pow(2.0).sum().detach()
    return distance.sqrt()

def init_res_saver():

    clean_acc = dict(retrain=[],
                    MUter=[],
                    Newton_delta=[],
                    Fisher_delta=[],
                    Influence_delta=[],
                    Newton=[],
                    Fisher=[],
                    Influence=[])

    perturb_acc = dict(retrain=[],
                    MUter=[],
                    Newton_delta=[],
                    Fisher_delta=[],
                    Influence_delta=[],
                    Newton=[],
                    Fisher=[],
                    Influence=[])

    distance = dict(retrain=[],
                    MUter=[],
                    Newton_delta=[],
                    Fisher_delta=[],
                    Influence_delta=[],
                    Newton=[],
                    Fisher=[],
                    Influence=[])
    
    saver = dict(
        unlearning_time_sequence = [],
        retrain_time_sequence = [],
        clean_acc = clean_acc,
        perturb_acc = perturb_acc,
        distance = distance
    )

    return saver

def covert(arr):
    arr = [item.cpu().detach().numpy() for item in arr]
    return arr
    
def res_save(arr, method, metric, args):
    remove_type = ['one_step_single_point', 'one_step_batch_points', 'multiple_steps_single_point', 'multiple_steps_batch_points']
    
    root = '../data/Result/{}/{}'.format(args.dataset, remove_type[args.remove_type])

    if not os.path.exists(root):
        os.makedirs(root)
    
    path = os.path.join(root, '{}_adv_{}_model_{}_method_{}_experiment{}'.format(metric, args.adv, args.model, method, args.times))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    np.save(path, arr)
    print('save done !')

def save(saver, args):
    # retrain 
    res_save(saver.get('perturb_acc').get('retrain'), 'retrain', 'perturbed_acc', args)
    res_save(saver.get('clean_acc').get('retrain'), 'retrain', 'acc', args)
    res_save(saver.get('retrain_time_sequence'), 'retrain', 'time', args)
    res_save(covert(saver.get('distance').get('retrain')), 'retrain', 'distance', args)

    # unlearning 
    res_save(saver.get('perturb_acc').get('MUter'), 'MUter', 'perturbed_acc', args)
    res_save(saver.get('clean_acc').get('MUter'), 'MUter', 'acc', args)
    res_save(saver.get('unlearning_time_sequence'), 'MUter', 'time', args)
    res_save(covert(saver.get('distance').get('MUter')), 'MUter', 'distance', args)

    # Newton_delta
    res_save(saver.get('perturb_acc').get('Newton_delta'), 'Newton_delta','perturbed_acc', args)
    res_save(saver.get('clean_acc').get('Newton_delta'), 'Newton_delta','acc', args)
    res_save(covert(saver.get('distance').get('Newton_delta')), 'Newton_delta','distance', args)

    # Newton
    res_save(saver.get('perturb_acc').get('Newton'), 'Newton', 'perturbed_acc', args)
    res_save(saver.get('clean_acc').get('Newton'), 'Newton', 'acc', args)
    res_save(covert(saver.get('distance').get('Newton')), 'Newton', 'distance', args)

    # Influence_delta
    res_save(saver.get('perturb_acc').get('Influence_delta'), 'influence_delta', 'perturbed_acc', args)
    res_save(saver.get('clean_acc').get('Influence_delta'), 'influence_delta', 'acc', args)
    res_save(covert(saver.get('distance').get('Influence_delta')), 'influence_delta', 'distance', args)

    # Influence
    res_save(saver.get('perturb_acc').get('Influence'), 'Influence', 'perturbed_acc', args)
    res_save(saver.get('clean_acc').get('Influence'), 'Influence', 'acc', args)
    res_save(covert(saver.get('distance').get('Influence')), 'Influence', 'distance', args)

    # Fisher_delta
    res_save(saver.get('perturb_acc').get('Fisher_delta'), 'Fisher_delta', 'perturbed_acc', args)
    res_save(saver.get('clean_acc').get('Fisher_delta'), 'Fisher_delta', 'acc', args)
    res_save(covert(saver.get('distance').get('Fisher_delta')), 'Fisher_delta', 'distance', args)

    # Fisher
    res_save(saver.get('perturb_acc').get('Fisher'), 'Fisher', 'perturbed_acc', args)
    res_save(saver.get('clean_acc').get('Fisher'), 'Fisher', 'acc', args)
    res_save(covert(saver.get('distance').get('Fisher')), 'Fisher', 'distance', args)


if __name__ == "__main__":


    # # vec = torch.tensor([0., 0., 0.002, 0.001, 0.004, 0., 0.]).unsqueeze_(dim=1)
    # # matrix = vec.mm(vec.t()) + 0.0001 * torch.eye(7)
    # matrix = torch.tensor([[0.2, 0.], [0., 0.3]])
    # print(matrix.pinverse())
    # print(matrix.mm(matrix.pinverse()))

    # print(Neumann(matrix, iter=10))
    # print(matrix.mm(Neumann(matrix, iter=10)))

    # print(rank1_pinverse(vec))
    # print(matrix.mm(rank1_pinverse(vec)))
    # test the blockmatrix function-- forming_blocks
    ##############################################################################################
    # device = 'cuda'
    # M = torch.tensor([[5., 5., 5., 5.], [5., 5., 5., 5.], [5., 5., 5., 5.], [5., 5., 5., 5.]]).to(device)
    # H_11 = torch.tensor([[1., 2., 3., 4.], [3., 4., 2., 1.], [5., 5., 5., 5.], [5., 5., 5., 5.]]).to(device)
    # H_12 = torch.tensor([[5., 6.], [7., 8.], [2., 2.], [3., 3.]]).to(device)
    # H_21 = torch.tensor([[2., 4., 1., 3.], [1., 3., 2., 4.]]).to(device)
    # H_22 = torch.tensor([[7., 9.], [6., 8.]]).to(device)
    # print('M shape {}'.format(M.shape))
    # print('H_11 shape {}'.format(H_11.shape))
    # print('H_12 shape {}'.format(H_12.shape))
    # print('H_21 shape {}'.format(H_21.shape))
    # print('H_22 shape {}'.format(H_22.shape))

    # #block_matrix = forming_blocks(M, upper_left, upper_right, lower_right, lower_left)
    # block_matrix = buliding_matrix(M, H_11, H_12, H_22, H_21)
    # print(block_matrix)

    # args = argument()
    # device = 'cuda'

    # feature = get_featrue(args)
    # train_data, test_data = Load_Data(args)
    # train_loader = make_loader(train_data, batch_size=128)
    # pass_loader = make_loader(train_data, batch_size=1)
    # test_loader = make_loader(test_data, batch_size=128)
    # model, training_time = train(train_loader, test_loader, args, verbose=True)
    # model = model.to(device)

    # weight = vec_param(model.parameters()).detach()
    # public_partial_xx = (weight.mm(weight.t())).detach()    
    # public_partial_xx_inv = ((torch.tensor(2.0)*torch.eye(feature)).to(device) - public_partial_xx).detach() 

    # h_11 = None
    # h_12 = None
    # h_22 = None
    # h_21 = None

    # for index, (image, label) in enumerate(pass_loader):
    #     Dww, partial_ww, partial_wx, partial_xx, partial_xw = partial_hessian(image.view(feature, 1), label, weight, public_partial_xx_inv, args)
    #     h_11 = partial_ww.detach()
    #     h_12 = partial_wx.detach()
    #     h_22 = partial_xx.detach()
    #     h_21 = partial_xw.detach()
    #     if index == 0:
    #         break
    # print('h_11 shape {}'.format(h_11.shape))
    # print('h_12 shape {}'.format(h_12.shape))
    # print('h_22 shape {}'.format(h_22.shape))
    # print('h_21 shape {}'.format(h_21.shape))
    
    # memory_matrix = load_memory_matrix(args)
    # print('memory_matrix shape {}, type {}'.format(memory_matrix.shape, type(memory_matrix)))

    # block_matrix = buliding_matrix(memory_matrix, h_11, h_12, h_22, h_21)
    # print('block matrix shape {}, type {}'.format(block_matrix.shape, type(block_matrix)))

    # # test save and load memory matrix
    # ################################################################################################
    # args = argument()
    # device = 'cuda'

    # feature = get_featrue(args)
    # train_data, test_data = Load_Data(args)
    # train_loader = make_loader(train_data, batch_size=128)
    # pass_loader = make_loader(train_data, batch_size=1)
    # test_loader = make_loader(test_data, batch_size=128)
    # model, training_time = train(train_loader, test_loader, args, verbose=True)
    # model = model.to(device)

    # print('training done !')
    # print('preparing to store memory matrix...')

    # matrix = calculate_memory_matrix(model, pass_loader, args)
    # store_memory_matrix(matrix, args)
    # matrix = load_memory_matrix(args)
    # print('matrix shape {}, type {}'.format(matrix.shape, type(matrix)))




    # test the correct of derivate
    ##################################################################################################
    # # to test the isRight for our derivate 
    # args = argument()
    # device = 'cuda'

    # featrue = get_featrue(args)
    # train_data, test_data = Load_Data(args)
    # train_loader = make_loader(train_data, batch_size=128)
    # test_loader = make_loader(test_data, batch_size=128)
    # model, training_time = train(train_loader, test_loader, args, verbose=True)
    # model = model.to(device)

    # _, _, atk_info = training_param(args)
    # atk = PGD(model, eps=atk_info[0], alpha=atk_info[1], steps=atk_info[2], lossfun=LossFunction(args.model), lam=args.lam)

    # weight = vec_param(model.parameters()).detach()
    # public_partial_xx = (weight.mm(weight.t())).detach()

    # pass_loader = make_loader(train_data, batch_size=1)
    # for index, (image, label) in enumerate(pass_loader):
    #     print('pass the {} th sampe'.format(index))

    #     label = label.to(device)
    #     image = atk(image, label).to(device)

    #     # Dww, partial_ww, partial_wx, partial_xx, partial_xw = partial_hessian(image.view(featrue, 1), label, weight, public_partial_xx, args)
    #     # print('Dww shape {}'.format(Dww.shape))
    #     # print('partial_ww shape {}'.format(partial_ww.shape))
    #     # print('partial_wx shape {}'.format(partial_wx.shape))
    #     # print('partial_xx shape {}'.format(partial_xx.shape))
    #     # print('partial_xw shape {}'.format(partial_xw.shape))
    #     # print(partial_xx.sum())
    #     grad_w = loss_grad(weight, image.view(featrue, 1), label, args)
    #     #print(grad_w.sum())
    #     # auto_ww, auto_wx, auto_xx, auto_xw = test_auto_partial_hessian(model, image, label, args)
    #     # print('auto_ww shape {}'.format(auto_ww.shape))
    #     # print('auto_wx shape {}'.format(auto_wx.shape))
    #     # print('auto_xx shape {}'.format(auto_xx.shape))
    #     # print('auto_xw shape {}'.format(auto_xw.shape))
    #     # print(auto_xx.sum())
    #     auto_grad_w = auto_grad(model, image, label, args)
    #     #print(auto_grad_w.sum())
    #     print()
    #     verify(grad_w, auto_grad_w)
    #     # verify(partial_xx, auto_xx)
    #     if index == 1000:
    #           break
    """
    end
    """