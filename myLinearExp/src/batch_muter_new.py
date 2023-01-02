import torch
import random
import time
import copy
import numpy as np

from torchattacks import PGD

from dataloder import *
from argument import *
from model import *
from pretrain import *
from utils import *
from parllutils import *
from modules import *

args = argument()
device = 'cuda'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# random seed
setup_seed(args.times)

delete_num = args.deletenum # default 0
delete_batch = args.deletebatch # default 1
pass_batch = args.parllsize

## load data
train_data, test_data, re_sequence = Load_Data(args, delete_num, shuffle=True)
train_loader = make_loader(train_data, batch_size=args.batchsize)
test_loader = make_loader(test_data, batch_size=args.batchsize)

# Adversarisal training
model_path = os.path.join('..', 'data', 'ATM', f"dataset_{args.dataset}_adv_{args.adv}_model_{args.model}_points_{len(train_loader.dataset)}_{args.times}.pth")
model, training_time = train(train_loader, test_loader, args, desc='Pre-Adv Training', verbose=True, model_path=model_path)

# pre-unlearning
pass_loader = make_loader(train_data, batch_size=pass_batch)

# Calculate the hessian matrix of partial_dd
matrices = dict(MUter=None,
                Newton_delta=None,
                Fisher_delta=None,
                Influence_delta=None,
                Newton=None,
                Fisher=None,
                Influence=None)

for name in matrices.keys():
    method = name.split('_')
    isDelta = True if len(method) > 1 else False
    ssr = 'perturbed' if len(method) > 1 else 'unperturbed'
    filename = f'dataset_{args.dataset}_adv_{args.adv}_model_{args.model}_method_{method[0]}_sample_{ssr}_{args.times}.pt'
    print(name)
    matrices[name] = load_memory_matrix(filename, model, pass_loader, method[0],isDelta, args)

# Inner level attack method
_, _, atk_info = training_param(args)
atk = PGD(model, atk_info[0], atk_info[1], atk_info[2], lossfun=LossFunction(args.model), lam=args.lam)

# Calculate the public part partial_xx and partial_xx_inv for linear model
feature = get_featrue(args)
weight = vec_param(model.parameters()).detach()
public_partial_xx = (weight.mm(weight.t())).detach()
public_partial_xx_inv = derive_inv(public_partial_xx, method='Neumann', iter=args.iterneumann)

# record all results
saver = init_res_saver()

step = 1 # record unlearning times
remove_list = get_remove_list(args.dataset)

# Init gradinet informations
grad = torch.zeros((feature, 1)).to(device)
clean_grad = torch.zeros((feature, 1)).to(device)
parll_partial = batch_indirect_hessian(args)
# 
# 每次删 delete_num 个数据点
# deletebatch = 1: 每次只删一个点
for batch_delete_num in range(delete_batch, delete_num+1, delete_batch):
    if args.remove_type == 2:
        if batch_delete_num not in remove_list:
            continue

        else:
            if batch_delete_num > 5:
                index = remove_list.index(batch_delete_num)
                pre_index = index - 1
                sub_seq = re_sequence[remove_list[pre_index]:remove_list[index]-1]

                # remove matrix and add grad
                temp_loader = make_loader(train_data, batch_size=pass_batch, head=remove_list[pre_index], rear=remove_list[index]-1)
                
                for index, (image, label) in enumerate(temp_loader):
                    image = image.to(device)
                    label = label.to(device)
                    image_perturbed = atk(image, label).to(device)

                    update_grad(grad, clean_grad, weight, image, image_perturbed, label, feature, args)
                    update_matrix(matrices, weight, image, image_perturbed, label, feature, public_partial_xx_inv, args)

    print('\n=================')
    print('The {}-th delete'.format(step))
    step = step + 1

    unlearning_model = copy.deepcopy(model).to(device)  # for MUter method
    delete_loader = make_loader(train_data, batch_size=pass_batch, head=(batch_delete_num-delete_batch), rear=batch_delete_num)
    
    ## retrain_from_scratch
    retrain_loader = make_loader(train_data, batch_size=128, head=batch_delete_num)
    retrain_model = retrain_from_scratch(retrain_loader, test_loader, args, saver)

    # calculate the aggregated grad & clean_grad
    for index, (image, label) in enumerate(delete_loader):
        image = image.to(device)
        label = label.to(device)
        image_perturbed = atk(image, label).to(device)

        update_grad(grad, clean_grad, weight, image, image_perturbed, label, feature, args)
    
    # unlearning stage
    ## MUter
    Dww, H_11, H_12, H_21, neg_H_22 = None, None, None, None, None
    start_time = time.time()
    for index, (image, label) in enumerate(delete_loader):
        image = image.to(device)
        label = label.to(device)
        image_perturbed = atk(image, label).to(device)

        if args.isbatch ==  False:
            Dww, H_11, H_12, _, H_21, neg_H_22 = partial_hessian(image_perturbed.view(feature, 1), label, weight, public_partial_xx_inv, args, isUn_inv=True, public_partial_xx=public_partial_xx)
        else: # for mini-batch
            update_matrix(matrices, weight, image, image_perturbed, label, feature, public_partial_xx_inv, args, flag='muter')
    
    unlearn_muter(matrices['MUter'], model, grad, Dww, H_11, H_12, H_21, neg_H_22, feature, device, start_time, retrain_model, test_loader, args, saver)

    ## Influence_delta / Influence
    # start_time = time.time()
    unlearn_baseline('Influence_delta', matrices['Influence_delta'], model, grad, device, retrain_model, test_loader, args, saver)
    # start_time = time.time()
    unlearn_baseline('Influence', matrices['Influence'], model, clean_grad, device, retrain_model, test_loader, args, saver)

    for index, (image, label) in enumerate(delete_loader):
        image = image.to(device)
        label = label.to(device)
        image_perturbed = atk(image, label).to(device)

        update_matrix(matrices, weight, image, image_perturbed, label, feature, public_partial_xx_inv, args, flag='baselines')

    ## Newton_delta / Newton / Fisher_delta / Fisher
    # start_time = time.time()
    unlearn_baseline('Newton_delta', matrices['Newton_delta'], model, grad, device, retrain_model, test_loader, args, saver)
    # start_time = time.time()
    unlearn_baseline('Newton', matrices['Newton'], model, clean_grad, device, retrain_model, test_loader, args, saver)
    # start_time = time.time()
    unlearn_baseline('Fisher_delta', matrices['Fisher_delta'], model, grad, device, retrain_model, test_loader, args, saver)
    # start_time = time.time()
    unlearn_baseline('Fisher', matrices['Fisher'], model, clean_grad, device, retrain_model, test_loader, args, saver)

## save 
save(saver, args)