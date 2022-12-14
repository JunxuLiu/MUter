import torch
from dataloder import *
from argument import *
from model import *
from pretrain import *
from utils import *
from parllutils import *
from functorch import vmap
args = argument()
device = 'cuda'

delete_num = args.deletenum # default 0
delete_batch = args.deletebatch # default 1
pass_batch = args.parllsize

## load data
train_data, test_data, re_sequence = Load_Data(args, delete_num, shuffle=True)
train_loader = make_loader(train_data, batch_size=args.batchsize)
test_loader = make_loader(test_data, batch_size=args.batchsize)

# Adversarisal training  
model, training_time = train(train_loader, test_loader, args, verbose=True)

# pre-unlearning
pass_loader = make_loader(train_data, batch_size=pass_batch)

# Calculate the hessian matrix of partial_dd
matrix = parll_calculate_memory_matrix(model, pass_loader, args, method='MUter')

# method x-delta: with adversarial perturbation
Newton_matrix_perturbed = parll_calculate_memory_matrix(model, pass_loader, args, method='Newton')
Fisher_matrix_perturbed = parll_calculate_memory_matrix(model, pass_loader, args, method='Fisher')
Influence_matrix_perturbed = parll_calculate_memory_matrix(model, pass_loader, args, method='Influence')

# method x: without adversarial perturbation
Newton_matrix_unperturbed = parll_calculate_memory_matrix(model, pass_loader, args, method='Newton', isDelta=False)
Fisher_matrix_unperturbed = parll_calculate_memory_matrix(model, pass_loader, args, method='Fisher', isDelta=False)
Influence_matrix_unperturbed = parll_calculate_memory_matrix(model, pass_loader, args, method='Influence', isDelta=False)

store_memory_matrix(matrix, args, method='MUter')

store_memory_matrix(Newton_matrix_perturbed, args, method='Newton')
store_memory_matrix(Newton_matrix_unperturbed, args, method='Newton', isDelta=False)

store_memory_matrix(Fisher_matrix_perturbed, args, method='Fisher')
store_memory_matrix(Fisher_matrix_unperturbed, args, method='Fisher', isDelta=False)

store_memory_matrix(Influence_matrix_perturbed, args, method='Influence')
store_memory_matrix(Influence_matrix_unperturbed, args, method='Influence', isDelta=False)


del matrix

del Newton_matrix_perturbed
del Newton_matrix_unperturbed

del Fisher_matrix_perturbed
del Fisher_matrix_unperturbed

del Influence_matrix_perturbed
del Influence_matrix_unperturbed

matrix = load_memory_matrix(args, method='MUter').to(device)

Newton_matrix_perturbed = load_memory_matrix(args, method='Newton').to(device)
Newton_matrix_unperturbed = load_memory_matrix(args, method='Newton', isDelta=False).to(device)

Fisher_matrix_perturbed = load_memory_matrix(args, method='Fisher').to(device)
Fisher_matrix_unperturbed = load_memory_matrix(args, method='Fisher', isDelta=False).to(device)

Influence_matrix_perturbed = load_memory_matrix(args, method='Influence').to(device)
Influence_matrix_unperturbed = load_memory_matrix(args, method='Influence', isDelta=False).to(device)

from torchattacks import PGD
import copy
from utils import cg_solve, model_distance, hessian, update_w, derive_inv
import time
from torch.utils.data import DataLoader, TensorDataset

# Inner level attack method
_, _, atk_info = training_param(args)
atk = PGD(model, atk_info[0], atk_info[1], atk_info[2], lossfun=LossFunction(args.model), lam=args.lam)

# Calculate the public part partial_xx and partial_xx_inv for linear model
feature = get_featrue(args)
weight = vec_param(model.parameters()).detach()
public_partial_xx = (weight.mm(weight.t())).detach()
public_partial_xx_inv = derive_inv(public_partial_xx, method='Neumann', iter=args.iterneumann)


# record time
unlearning_time_sequence = []
retrain_time_sequence = []
sisa_time_sequence = []

# for train from scratch
retrain_clean_acc = []
retrain_perturb_acc = []
retrain_distance = []
original_distance = []
generating_samples_acc = []

# for MUter method
unlearning_clean_acc = []
unlearning_perturb_acc = []
unlearning_distance = []
unlearning_original_distance = []
unlearning_retrain_generating_samples_acc = []

# for Newton with perturbed samples
Newton_delta_clean_acc = []
Newton_delta_perturb_acc = []
Newton_delta_distance = []
Newton_delta_original_distance = []
Newton_delta_retrain_generating_samples_acc = []

# for Newton without perturbed samples  
Newton_clean_acc = []
Newton_perturb_acc = []
Newton_distance = []
Newton_original_distance = []
Newton_retrain_generating_samples_acc = []

# for Fisher with perturbed samples
Fisher_delta_clean_acc = []
Fisher_delta_perturb_acc = []
Fisher_delta_distance = []
Fisher_delta_original_distance = []
Fisher_delta_retrain_generating_samples_acc = []

# for Fisher without perturbed samples
Fisher_clean_acc = []
Fisher_perturb_acc = []
Fisher_distance = []
Fisher_original_distance = []
Fisher_retrain_generating_samples_acc = []

# for Influence with perturbed samples
Influence_delta_clean_acc = []
Influence_delta_perturb_acc = []
Influence_delta_distance = []
Influence_delta_original_distance = []
Influence_delta_retrain_generating_samples_acc = []

# for Influence without perturbed samples
Influence_clean_acc = []
Influence_perturb_acc = []
Influence_distance = []
Influence_original_distance = []
Influence_retrain_generating_samples_acc = []

# Init gradinet informations
grad = torch.zeros((feature, 1)).to(device)
clean_grad = torch.zeros((feature, 1)).to(device)

step = 1 # record unlearning times
## compare with removal list [1, 2, 3, 4, 5, ~1%, ~2%, ~3%, ~4%, ~5%] 
remove_list = None
if args.dataset == 'binaryMnist':
    remove_list = [1, 2, 3, 4, 5, 120, 240, 360, 480, 600]  # for mnist
elif args.dataset == 'phishing':
    remove_list = [1, 2, 3, 4, 5, 100, 200, 300, 400, 500]  # for phsihing
elif args.dataset == 'madelon':
    remove_list = [1, 2, 3, 4, 5, 20, 40, 60, 80, 100]  # for madelon
elif args.dataset == 'covtype':
    remove_list = [1, 2, 3, 4, 5, 5000, 10000, 15000, 20000, 25000]
elif args.dataset == 'epsilon':
    remove_list = [1, 2, 3, 4, 5, 4000, 8000, 12000, 16000, 20000]
else:
    remove_list = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]  # for splice

# 每次删 delete_num 个数据点
# deletebatch = 1: 每次只删一个点
parll_partial = batch_indirect_hessian(args)
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

                    # for perturbed grad
                    grad = grad + parll_loss_grad(weight, image_perturbed.view(image_perturbed.shape[0], feature), label, args)

                    # for clean grad
                    clean_grad = clean_grad + parll_loss_grad(weight, image.view(image.shape[0], feature), label, args).detach()

                    # for MUter matrix
                    matrix = matrix - (batch_hessian(weight, image_perturbed.view(image_perturbed.shape[0], feature), label, args) - parll_partial(image_perturbed.view(image_perturbed.shape[0], feature, 1), label, weight, public_partial_xx_inv).sum(dim=0).detach())

                    # for other 6 variant
                    perturbed_partial_ww = batch_hessian(weight, image_perturbed.view(image_perturbed.shape[0], feature), label, args)
                    clean_partial_ww = batch_hessian(weight, image.view(image.shape[0], feature), label, args)

                    perturbed_fisher_matrix = batch_fisher(weight, image_perturbed.view(image_perturbed.shape[0], feature), label, args)
                    clean_fisher_matrix = batch_fisher(weight, image.view(image.shape[0], feature), label, args)

                    Newton_matrix_perturbed = Newton_matrix_perturbed - perturbed_partial_ww
                    Newton_matrix_unperturbed = Newton_matrix_unperturbed - clean_partial_ww

                    Influence_matrix_perturbed = Influence_matrix_perturbed - perturbed_partial_ww
                    Influence_matrix_unperturbed = Influence_matrix_unperturbed - clean_partial_ww

                    Fisher_matrix_perturbed = Fisher_matrix_perturbed - perturbed_fisher_matrix
                    Fisher_matrix_unperturbed = Fisher_matrix_unperturbed - clean_fisher_matrix

    print()           
    print('The {}-th delete'.format(step))
    step = step + 1

    # prepare work
    unlearning_model = copy.deepcopy(model).to(device)  # for MUter method

    Newton_delta_model = copy.deepcopy(model).to(device)
    Fisher_delta_model = copy.deepcopy(model).to(device)
    Influence_delta_model = copy.deepcopy(model).to(device)

    Newton_model = copy.deepcopy(model).to(device)
    Fisher_model = copy.deepcopy(model).to(device)
    Influence_model = copy.deepcopy(model).to(device)

    delete_loader = make_loader(train_data, batch_size=pass_batch, head=(batch_delete_num-delete_batch), rear=batch_delete_num)
    
    retrain_loader = make_loader(train_data, batch_size=128, head=batch_delete_num)
    retrain_model, retrain_time = train(retrain_loader, test_loader, args, verbose=False)
    clean_acc, perturb_acc = Test_model(retrain_model, test_loader, args)
    
    retrain_clean_acc.append(clean_acc)
    retrain_perturb_acc.append(perturb_acc)
    retrain_time_sequence.append(retrain_time)
    retrain_distance.append(model_distance(retrain_model, retrain_model))

    unlearning_time = 0.0 # record one batch spending time for MUter
    # building matrix
    Dww = None
    H_11 = None
    H_12 = None
    H_21 = None
    neg_H_22 = None

    start_time = time.time()
    for index, (image, label) in enumerate(delete_loader):
        image = image.to(device)
        label = label.to(device)
        image_perturbed = atk(image, label).to(device)

        if args.isbatch ==  False:
            Dww, H_11, H_12, _, H_21, neg_H_22 = partial_hessian(image_perturbed.view(feature, 1), label, weight, public_partial_xx_inv, args, isUn_inv=True, public_partial_xx=public_partial_xx)    
            
        else: # for mini-batch
            matrix = matrix - (batch_hessian(weight, image_perturbed.view(image_perturbed.shape[0], feature), label, args) - parll_partial(image_perturbed.view(image_perturbed.shape[0], feature, 1), label, weight, public_partial_xx_inv).sum(dim=0).detach())

        grad = grad + parll_loss_grad(weight, image_perturbed.view(image_perturbed.shape[0], feature), label, args)

    # unlearning stage
    ## 0.for MUter method
    if args.isbatch == False:
        block_matrix = buliding_matrix(matrix, H_11, H_12, -neg_H_22, H_21)
        print('block_matrix shape {}'.format(block_matrix.shape))
        grad_cat_zero = torch.cat([grad, torch.zeros((feature, 1)).to(device)], dim=0)
        print('grad_cat_zeor shape {}'.format(grad_cat_zero.shape))

        delta_w_cat_alpha = cg_solve(block_matrix, grad_cat_zero.squeeze(dim=1), get_iters(args))
        delta_w = delta_w_cat_alpha[:feature]

        update_w(delta_w, unlearning_model)
        matrix = matrix - Dww
    else:
        delta_w = cg_solve(matrix, grad.squeeze(dim=1), get_iters(args))
        update_w(delta_w, unlearning_model)

    end_time = time.time()
    unlearning_time = unlearning_time + (end_time - start_time)
    unlearning_time_sequence.append(unlearning_time)

    print()
    print('MUter unlearning:')
    clean_acc, perturb_acc = Test_model(unlearning_model, test_loader, args) 
    unlearning_clean_acc.append(clean_acc)
    unlearning_perturb_acc.append(perturb_acc)

    model_dist = model_distance(retrain_model, unlearning_model)
    unlearning_distance.append(model_dist)
    print('model norm distance: {:.4f}'.format(model_dist))
    print()
    ''' unused
    # unlearn_origin_dist = model_distance(model, unlearning_model)
    # unlearning_original_distance.append(unlearn_origin_dist)
    # unlearning_retrain_generating_samples_acc.append(test(unlearning_model, test_loader=adv_loader))
    '''
    # calculate clean_grad_sum
    for index, (image, label) in enumerate(delete_loader):
        image = image.to(device)
        label = label.to(device)
        clean_grad = clean_grad + parll_loss_grad(weight, image.view(image.shape[0], feature), label, args).detach()
    
    ## for Influence and Influence_delta
    # 1.Influence_delta
    delta_w_Influence_delta = cg_solve(Influence_matrix_perturbed, grad.squeeze(dim=1), get_iters(args))
    update_w(delta_w_Influence_delta, Influence_delta_model)

    print('Influnece-delta unlearning:')
    clean_acc, perturb_acc = Test_model(Influence_delta_model, test_loader, args)
    Influence_delta_clean_acc.append(clean_acc)
    Influence_delta_perturb_acc.append(perturb_acc)

    print('model norm distance: {:.4f}'.format(model_distance(retrain_model, Influence_delta_model)))
    Influence_delta_distance.append(model_distance(retrain_model, Influence_delta_model))
    # Influence_delta_original_distance.append(model_distance(model, Influence_delta_model))
    # Influence_delta_retrain_generating_samples_acc.append(test(Influence_delta_model, test_loader=adv_loader))

    print()

    # 2.Influence
    delta_w_Influence = cg_solve(Influence_matrix_unperturbed, clean_grad.squeeze(dim=1), get_iters(args))
    update_w(delta_w_Influence, Influence_model)

    print('Influence unlearning:')
    clean_acc, perturb_acc = Test_model(Influence_model, test_loader, args)
    Influence_clean_acc.append(clean_acc)
    Influence_perturb_acc.append(perturb_acc)

    print('model norm distance: {:.4f}'.format(model_distance(retrain_model, Influence_model)))
    Influence_distance.append(model_distance(retrain_model, Influence_model))
    # Influence_original_distance.append(model_distance(model, Influence_model))
    # Influence_retrain_generating_samples_acc.append(test(Influence_model, test_loader=adv_loader))
    print()

    for index, (image, label) in enumerate(delete_loader):
        image = image.to(device)
        label = label.to(device)
        image_perturbed = atk(image, label).to(device)
        
        perturbed_partial_ww = batch_hessian(weight, image_perturbed.view(image_perturbed.shape[0], feature), label, args)
        clean_partial_ww = batch_hessian(weight, image.view(image.shape[0], feature), label, args)

        perturbed_fisher_matrix = batch_fisher(weight, image_perturbed.view(image_perturbed.shape[0], feature), label, args)
        clean_fisher_matrix = batch_fisher(weight, image.view(image.shape[0], feature), label, args)

        Newton_matrix_perturbed = Newton_matrix_perturbed - perturbed_partial_ww
        Newton_matrix_unperturbed = Newton_matrix_unperturbed - clean_partial_ww

        Influence_matrix_perturbed = Influence_matrix_perturbed - perturbed_partial_ww
        Influence_matrix_unperturbed = Influence_matrix_unperturbed - clean_partial_ww

        Fisher_matrix_perturbed = Fisher_matrix_perturbed - perturbed_fisher_matrix
        Fisher_matrix_unperturbed = Fisher_matrix_unperturbed - clean_fisher_matrix

    ## for Newton/Newton_delta, Fisher/Fisher_delta
    # 3.Newton_delta
    delta_w_Newton_delta = cg_solve(Newton_matrix_perturbed, grad.squeeze(dim=1), get_iters(args))
    update_w(delta_w_Newton_delta, Newton_delta_model)

    print('Newton-delta unlearning:')
    clean_acc, perturb_acc = Test_model(Newton_delta_model, test_loader, args)
    Newton_delta_clean_acc.append(clean_acc)
    Newton_delta_perturb_acc.append(perturb_acc)

    print('model norm distance: {:.4f}'.format(model_distance(retrain_model, Newton_delta_model)))
    Newton_delta_distance.append(model_distance(retrain_model, Newton_delta_model))
    # Newton_delta_original_distance.append(model_distance(model, Newton_delta_model))
    # Newton_delta_retrain_generating_samples_acc.append(test(Newton_delta_model, test_loader=adv_loader))

    print()

    # 4.Newton
    delta_w_Newton = cg_solve(Newton_matrix_unperturbed, clean_grad.squeeze(dim=1), get_iters(args))
    update_w(delta_w_Newton, Newton_model)

    print('Newton unlearning:')
    clean_acc, perturb_acc = Test_model(Newton_model, test_loader, args)
    Newton_clean_acc.append(clean_acc)
    Newton_perturb_acc.append(perturb_acc)

    print('model norm distance: {:.4f}'.format(model_distance(retrain_model, Newton_model)))
    Newton_distance.append(model_distance(retrain_model, Newton_model))
    # Newton_original_distance.append(model_distance(model, Newton_model))
    # Newton_retrain_generating_samples_acc.append(test(Newton_model, test_loader=adv_loader))
    print()

    # 5.Fisher_delta
    delta_w_Fisher_delta = cg_solve(Fisher_matrix_perturbed, grad.squeeze(dim=1), get_iters(args))
    update_w(delta_w_Fisher_delta, Fisher_delta_model)

    print('Fisher-delta unlearning:')
    clean_acc, perturb_acc = Test_model(Fisher_delta_model, test_loader, args)
    Fisher_delta_clean_acc.append(clean_acc)
    Fisher_delta_perturb_acc.append(perturb_acc)

    print('model norm distance: {:.4f}'.format(model_distance(retrain_model, Fisher_delta_model)))
    Fisher_delta_distance.append(model_distance(retrain_model, Fisher_delta_model))
    # Fisher_delta_original_distance.append(model_distance(model, Fisher_delta_model))
    # Fisher_delta_retrain_generating_samples_acc.append(test(Fisher_delta_model, test_loader=adv_loader))
    print()

    # 6.Fisher
    delta_w_Fisher = cg_solve(Fisher_matrix_unperturbed, clean_grad.squeeze(dim=1), get_iters(args))
    update_w(delta_w_Fisher, Fisher_model)

    print('Fisher unlearning')
    clean_acc, perturb_acc = Test_model(Fisher_model, test_loader, args)
    Fisher_clean_acc.append(clean_acc)
    Fisher_perturb_acc.append(perturb_acc)

    print('model norm distance: {:.4f}'.format(model_distance(retrain_model, Fisher_model)))
    Fisher_distance.append(model_distance(retrain_model, Fisher_model))
    # Fisher_original_distance.append(model_distance(model, Fisher_model))
    # Fisher_retrain_generating_samples_acc.append(test(Fisher_model, test_loader=adv_loader))
    print()

## save 

# retrain 
res_save(retrain_perturb_acc, 'retrain', 'perturbed_acc', args)
res_save(retrain_clean_acc, 'retrain', 'acc', args)
res_save(retrain_time_sequence, 'retrain', 'time', args)
res_save(covert(retrain_distance), 'retrain', 'distance', args)

# unlearning 
res_save(unlearning_perturb_acc, 'MUter', 'perturbed_acc', args)
res_save(unlearning_clean_acc, 'MUter', 'acc', args)
res_save(unlearning_time_sequence, 'MUter', 'time', args)
res_save(covert(unlearning_distance), 'MUter', 'distance', args)

# Newton_delta
res_save(Newton_delta_perturb_acc, 'Newton_delta','perturbed_acc', args)
res_save(Newton_delta_clean_acc, 'Newton_delta','acc', args)
res_save(covert(Newton_delta_distance), 'Newton_delta','distance', args)


# Newton
res_save(Newton_perturb_acc, 'Newton', 'perturbed_acc', args)
res_save(Newton_clean_acc, 'Newton', 'acc', args)
res_save(covert(Newton_distance), 'Newton', 'distance', args)

# Influence_delta
res_save(Influence_delta_perturb_acc, 'influence_delta', 'perturbed_acc', args)
res_save(Influence_delta_clean_acc, 'influence_delta', 'acc', args)
res_save(covert(Influence_delta_distance), 'influence_delta', 'distance', args)

# Influence
res_save(Influence_perturb_acc, 'Influence', 'perturbed_acc', args)
res_save(Influence_clean_acc, 'Influence', 'acc', args)
res_save(covert(Influence_distance), 'Influence', 'distance', args)

# Fisher_delta
res_save(Fisher_delta_perturb_acc, 'Fisher_delta', 'perturbed_acc', args)
res_save(Fisher_delta_clean_acc, 'Fisher_delta', 'acc', args)
res_save(covert(Fisher_delta_distance), 'Fisher_delta', 'distance', args)

# Fisher
res_save(Fisher_perturb_acc, 'Fisher', 'perturbed_acc', args)
res_save(Fisher_clean_acc, 'Fisher', 'acc', args)
res_save(covert(Fisher_distance), 'Fisher', 'distance', args)
