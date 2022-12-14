import torch
from dataloder import *
from argument import *
from model import *
from pretrain import *
from utils import *
from parllutils import *
from functorch import vmap
import copy

def unlearn_baseline(func_name, matrix, _model, grad, device, retrain_model, test_loader, args, saver=None):

    model = copy.deepcopy(_model).to(device)
    delta_w = cg_solve(matrix, grad.squeeze(dim=1), get_iters(args))
    update_w(delta_w, model)

    end_time = time.time()

    clean_acc, perturb_acc = Test_model(model, test_loader, args)
    model_dist = model_distance(retrain_model, model)
    # unlearning_time = end_time - start_time

    print()
    print('{} unlearning'.format(func_name))
    print('model test acc: clean_acc {:.4f}, preturb_acc: {:.4f}'.format(clean_acc,perturb_acc))
    print('model norm distance: {:.4f}'.format(model_dist))
    # print('unlearning time: {:.4f}'.format(unlearning_time))
    print()

    if saver:
        saver.get('clean_acc').get(func_name).append(clean_acc)
        saver.get('perturb_acc').get(func_name).append(perturb_acc)
        saver.get('distance').get(func_name).append(model_dist)

def unlearn_muter(matrix, _model, grad, Dww, H_11, H_12, H_21, neg_H_22, feature,
                device, start_time, retrain_model, test_loader, args, saver=None):
                
    # building matrix
    model = copy.deepcopy(_model).to(device)
    if args.isbatch == False:
        block_matrix = buliding_matrix(matrix, H_11, H_12, -neg_H_22, H_21)
        grad_cat_zero = torch.cat([grad, torch.zeros((feature, 1)).to(device)], dim=0)
        # print('block_matrix shape {}'.format(block_matrix.shape))
        # print('grad_cat_zeor shape {}'.format(grad_cat_zero.shape))

        delta_w_cat_alpha = cg_solve(block_matrix, grad_cat_zero.squeeze(dim=1), get_iters(args))
        delta_w = delta_w_cat_alpha[:feature]

        update_w(delta_w, model)
        matrix = matrix - Dww

    else:
        delta_w = cg_solve(matrix, grad.squeeze(dim=1), get_iters(args))
        update_w(delta_w, model)

    end_time = time.time()
    
    clean_acc, perturb_acc = Test_model(model, test_loader, args) 
    model_dist = model_distance(retrain_model, model)
    unlearning_time = end_time - start_time

    print()
    print('MUter unlearning:')
    print('model test acc: clean_acc {:.4f}, preturb_acc: {:.4f}'.format(clean_acc,perturb_acc))
    print('model norm distance: {:.4f}'.format(model_dist))
    print('unlearning time: {:.4f}'.format(unlearning_time))
    print()

    if saver:
        saver.get('clean_acc').get('MUter').append(clean_acc)
        saver.get('perturb_acc').get('MUter').append(perturb_acc)
        saver.get('distance').get('MUter').append(model_dist)
        saver.get('unlearning_time_sequence').append(unlearning_time)

    
    ''' unused
    # unlearn_origin_dist = model_distance(_model, model)
    # unlearning_original_distance.append(unlearn_origin_dist)
    # unlearning_retrain_generating_samples_acc.append(test(model, test_loader=adv_loader))
    '''

def retrain_from_scratch(retrain_loader, test_loader, args, saver=None):
    num_points = len(retrain_loader.dataset)
    model_path = os.path.join('..', 'data', 'ATM', f"dataset_{args.dataset}_adv_{args.adv}_model_{args.model}_points_{num_points}_{args.times}.pth")
    retrain_model, retrain_time = train(retrain_loader, test_loader, args, desc='Re-Adv Training', verbose=False, model_path=model_path)    
    clean_acc, perturb_acc = Test_model(retrain_model, test_loader, args)
    
    print()
    print('Retrain from scratch:')
    print('model test acc: clean_acc {:.4f}, preturb_acc: {:.4f}'.format(clean_acc,perturb_acc))
    print('retrain time: {:.4f}'.format(retrain_time))
    print()

    if saver:
        saver.get('clean_acc').get('retrain').append(clean_acc)
        saver.get('perturb_acc').get('retrain').append(perturb_acc)
        saver.get('retrain_time_sequence').append(retrain_time)

    return retrain_model