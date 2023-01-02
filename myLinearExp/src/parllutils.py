import os
import math
import pathlib
import torch
from torchattacks import PGD
from functorch import vmap
from argument import argument
from tqdm.autonotebook import trange, tqdm

from utils import LossFunction, get_featrue, training_param, derive_inv, vec_param, get_delta_w_dict

def parll_loss_grad(weight, X, Y, args):
    """
    weight : [m, 1]
    X : [batch, m]
    Y : [batch, 1]

    return sum(parlll_loss_grad) shape [m, 1]
    """
    device = 'cuda'
    X = X.to(device)
    Y = Y.to(device)
    weight = weight.to(device)

    if args.model == 'logistic':
        z = torch.sigmoid(Y * X.mm(weight))
        return (X.t().mm((z-1) * Y) + args.lam * X.size(0) * weight).detach()
    elif args.model == 'ridge':
        return (X.t().mm(X.mm(weight) - Y) + args.lam * X.size(0) * weight).detach()
    else:
        raise Exception('no such loss function, please recheck !')

def batch_hessian(weight, X, Y, args, batch_size=50000):
    device = 'cuda'
    weight = weight.to(device)
    X = X.to(device)
    Y = Y.to(device)

    if args.model == 'logistic':

        z = torch.sigmoid(Y * X.mm(weight))
        D = z * (1 - z)
        H = None    
        num_batch = int(math.ceil(X.size(0) / batch_size)) 
        for i in range(num_batch):
            lower = i * batch_size
            upper = min((i+1) * batch_size, X.size(0))
            X_i = X[lower:upper]
            if H is None:
                H = X_i.t().mm(D[lower:upper] * X_i)
            else:
                H += X_i.t().mm(D[lower:upper] * X_i)
        
        return (H + args.lam * X.size(0) * torch.eye(X.size(1)).to(device)).detach()
    
    elif args.model == 'ridge':
        H = None
        num_batch = int(math.ceil(X.size(0) / batch_size))

        for i in range(num_batch):
            lower = i * batch_size
            upper = min((i+1) * batch_size, X.size(0))
            X_i = X[lower:upper]
            if H is None:
                H = X_i.t().mm(X_i)
            else:
                H += X_i.t().mm(X_i)
        return (H + args.lam * X.size(0) * torch.eye(X.size(1)).to(device)).detach()
    
    else:
        raise Exception('no such loss function, please recheck !')

def batch_fisher(weight, X, Y, args):
    device = 'cuda'
    X = X.to(device)
    Y = Y.to(device)

    if args.model == 'logistic':
        z = torch.sigmoid(Y * (X.mm(weight)))
        grad = (((z-1) * Y) * X).t() + args.lam * weight
        return grad.mm(grad.t()).detach()
    elif args.model == 'ridge':
        grad = ((X.mm(weight) - Y) * X).t() + args.lam * weight
        return grad.mm(grad.t()).detach()
    else:
        raise Exception('no such loss function, please recheck !')

def logistic_partial_hessian(x, y, weight, public_partial_xx_inv):
    """
    for loss function == 'logistic'
    calculate single sample's partial_hessian, then using vamp function to 
    implement parll
    """
    device = 'cuda'
    size = weight.shape[0]

    z = torch.sigmoid(y * (x.t().mm(weight)))
    D = z * (1 - z)
    partial_wx = (D * (x.mm(weight.t()))) + ((z-1) * y * torch.eye(size).to(device))
    partial_xx_inv = (1/D) * public_partial_xx_inv
    partial_xw = (D * (weight.mm(x.t()))) + ((z-1) * y * torch.eye(size).to(device))
    
    return  partial_wx.mm(partial_xx_inv.mm(partial_xw))

def ridge_partial_hessian(x, y, weight, public_partial_xx_inv):
    device = 'cuda'
    size = weight.shape[0]

    partial_wx = (x.mm(weight.t())) + (weight.t().mm(x)-y)*torch.eye(size).to(device)
    partial_xx_inv = public_partial_xx_inv
    partial_xw = (weight.mm(x.t())) + (weight.t().mm(x)-y)*torch.eye(size).to(device)

    return partial_wx.mm(partial_xx_inv.mm(partial_xw))

def batch_indirect_hessian(args):
    """
    uisng vmap
    loader must setting the batch_size, such as 16, 32, 64 .... for parll calculate
    for batch remove method to calculate the indirect partial_hessian
    """
    parll_func = None

    if args.model == 'logistic':
        parll_func = vmap(logistic_partial_hessian, in_dims=(0, 0, None, None))
    elif args.model == 'ridge':
        parll_func = vmap(ridge_partial_hessian, in_dims=(0, 0, None, None))
    else:
        raise Exception('no such loss function, please recheck !')
    return parll_func

def parll_calculate_memory_matrix(model, loader, args, method='MUter', isDelta=True):
    device = 'cuda'
    feature = get_featrue(args) # featrue_dict[‘binaryMnist’]=784
    matrix = torch.zeros((feature, feature)).to(device)
    model = model.to(device)
    _, _, atk_info = training_param(args)
    atk = PGD(model, eps=atk_info[0], alpha=atk_info[1], steps=atk_info[2], lossfun=LossFunction(args.model), lam=args.lam)
    # 将所有参数 vectorization，即 flatten 成一个一维矩阵
    weight = vec_param(model.parameters()).detach()
    
    public_partial_xx = (weight.mm(weight.t())).detach() # w*wT
    public_partial_xx_inv = derive_inv(public_partial_xx, method='Neumann', iter=args.iterneumann)

    # lenth = len(loader)
    if method == 'MUter':
        parll_partial = batch_indirect_hessian(args) # func
        # lenth = len(loader)
        with tqdm(total=len(loader)) as t:
            for image, label in loader:
                t.set_description(f"M matrix computing")
                image = image.to(device)
                label = label.to(device)
                image = atk(image, label).to(device)
                matrix += batch_hessian(weight, image.view(image.shape[0], feature), label, args)
                matrix -= parll_partial(image.view(image.shape[0], feature, 1), label, weight, public_partial_xx_inv).sum(dim=0).detach()
                t.update()
        return matrix
    
    else:
        with tqdm(total=len(loader)) as t:
            for image, label in loader:
                t.set_description(f"M matrix computing")
                image = image.to(device)
                label = label.to(device)
                if isDelta == True:
                    image = atk(image, label).to(device)
                
                if method == 'Newton' or method == 'Influence':
                    matrix = matrix + batch_hessian(weight, image.view(image.shape[0], feature), label, args)
                elif method == 'Fisher':
                    matrix = matrix + batch_fisher(weight, image.view(image.shape[0], feature), label, args)
                else:
                    raise Exception('no such method called {}, please recheck !'.format(method))
                t.update()
                
        return matrix

def load_memory_matrix(filename, model, loader, method, isDelta, args):
    dir = os.path.join('..','data','MemoryMatrix')
    path = os.path.join(dir, f"{filename}.pt")

    if not os.path.exists(path):
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        matrix = parll_calculate_memory_matrix(model, loader, args, method, isDelta)
        torch.save(matrix, path)

        print('saving memory matrix of {} method to: {}'.format(method, path))
        print('======== matrix info ========')
        print(matrix)
        print('matrix shape {}, type {}'.format(matrix.shape, type(matrix)))
        return matrix
    
    matrix = torch.load(path)

    print('loading memory matrix of {} method from: {}'.format(method, path))
    print('======== matrix info ========')
    print(matrix)
    print('matrix shape {}, type {}'.format(matrix.shape, type(matrix)))
    return matrix

def update_w(delta_w, model):
    device = 'cuda'
    direction = get_delta_w_dict(delta_w, model)

    for k,p in model.named_parameters():
        p.data += (direction[k]).to(device)

    print('Update done !')

def update_grad(grad, clean_grad, w, x, x_delta, y, feature, args):
    grad += parll_loss_grad(w, x_delta.view(x_delta.shape[0], feature), y, args).detach()
    clean_grad += parll_loss_grad(w, x.view(x.shape[0], feature), y, args).detach()
    return grad, clean_grad

def update_matrix(matrices, w, x, x_delta, y, feature, public_partial_xx_inv, args, flag='full'):
    """
    
    augments:
        flag: specify the updating items, for example, 'full' means update all matrics, 'muter' means only updata matrix of MUter 
                and 'baseline' means update matrics of other baselines.
    """
    
    perturbed_partial_ww = batch_hessian(w, x_delta.view(x_delta.shape[0], feature), y, args)
    clean_partial_ww = batch_hessian(w, x.view(x.shape[0], feature), y, args)

    perturbed_fisher_matrix = batch_fisher(w, x_delta.view(x_delta.shape[0], feature), y, args)
    clean_fisher_matrix = batch_fisher(w, x.view(x.shape[0], feature), y, args)

    if flag == 'muter' or flag == 'full':
        parll_partial = batch_indirect_hessian(args)
        indirect_hessian = parll_partial(x_delta.view(x_delta.shape[0], feature, 1), y, w, public_partial_xx_inv).sum(dim=0).detach()

        matrices['MUter'] -= (perturbed_partial_ww - indirect_hessian)

    if flag == 'baseline' or flag == 'full':

        matrices['Influence_delta'] -= perturbed_partial_ww
        matrices['Influence'] -= clean_partial_ww

        matrices['Newton_delta'] -= perturbed_partial_ww
        matrices['Newton'] -= clean_partial_ww

        matrices['Fisher_delta'] -= perturbed_fisher_matrix
        matrices['Fisher'] -= clean_fisher_matrix

if __name__ == "__main__":
    
    device = 'cuda'
    args = argument()

    ## test batch_loss_grad
    # X = torch.ones((128, 784)).to(device)
    # Y = torch.ones((128, 1)).to(device)
    # weight = torch.ones((784, 1)).to(device)

    # start = time.time()
    # print(parll_loss_grad(weight, X, Y, args).shape)
    # end = time.time()
    # print('time : {:.2f} seconds'.format(end -start))

    # ## test batch_hessian
    # X = torch.randn((128, 784)).to(device)
    # Y = torch.randn((128, 1)).to(device)
    # weight = torch.randn((784, 1)).to(device)

    # start = time.time()
    # print(batch_hessian(weight, X, Y, args).shape)
    # end = time.time()
    # print('time : {:.2f} seconds'.format(end -start))

    # ## test batch_fisher
    # X = torch.randn((128, 784)).to(device)
    # Y = torch.randn((128, 1)).to(device)
    # weight = torch.randn((784, 1)).to(device)

    # start = time.time()
    # print(batch_fisher(weight, X, Y, args).shape)
    # end = time.time()
    # print('time : {:.2f} seconds'.format(end -start))

    ## test batch_partial
    # X = torch.randn(32, 784, 1).to(device)
    # Y = torch.randn(32, 1).to(device)
    # weight = torch.randn(784, 1).to(device)
    # public_public_xx_inv = torch.randn((784, 784)).to(device)
    # indirect_fun = batch_indirect_hessian(args)
    # print(indirect_fun(X, Y, weight, public_public_xx_inv).sum(dim=0).detach().shape)