import torch
import argparse
import os
import time
import numpy as np
from Train import Neter
from Remover import MUterRemover, NewtonRemover, InfluenceRemover, FisherRemover, SchurMUterRemover
from Recorder import Recorder
from data_utils import Dataer
from utils import get_layers, get_pretrain_model_path
from utils import get_random_sequence, generate_save_name, get_BatchRemove_sequence, get_goal_dataset
from Noise_inject import Get_metrics

"""
main code for machine unlearning,
"""

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='ImageNet')
parser.add_argument('--epochs', type=int, default=99, help='custom the training epochs')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batchsize', type=int, default=128, help='the traning batch size')
parser.add_argument('--times', type=int, default=0, help='do repeat experiments')
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--ngpu', default=1, type=int)

# for remove type chose
parser.add_argument('--adv_type', type=str, default='PGD', help='the adv training type')
parser.add_argument('--isBatchRemove', type=int, default=0, help='0: no batch, Schur complement. 1: batch, Neumann')

# for pretrain type
parser.add_argument('--isPretrain', default=True, type=bool)
parser.add_argument('--layers', default=28, type=int, help='total number of layers')
parser.add_argument('--widen_factor', default=10, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')
parser.add_argument('--tuning_epochs', default=10, type=int)
parser.add_argument('--tuning_lr', default=0.001, type=float)
parser.add_argument('--tuning_layer', default='linear', type=str)
parser.add_argument('--isBias', default=False, type=bool)

# for repeat experiments
parser.add_argument('--seed', default=666, type=int, help='determate the remove data id')

args = parser.parse_args()


remove_squence_dict = {
    0: get_BatchRemove_sequence(args, isPretrain=False),
    1: get_BatchRemove_sequence(args, isPretrain=False),
    3: [400, ]
}

remove_squence = remove_squence_dict[args.isBatchRemove]

"""
1) traninig a robust model for unlearning
2) pre calculate the matrix, store and load
3) the unlearning request coming, do unlearning and measure the metrics.
4) post of unlearning 
"""

### pre work
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu_id)
recorder = Recorder(args=args)

pretrain_param = None
if args.isPretrain:
    pretrain_param = {
        'layers': args.layers,
        'widen_factor': args.widen_factor,
        'droprate': args.droprate,
        'root_path': get_pretrain_model_path(args.dataset),
        'epochs': args.tuning_epochs,
        'lr': args.tuning_lr,
        'new_last_layer': get_layers(args.tuning_layer, isBias=args.isBias),
    }

#####
# Stage 1) traninig a robust model for unlearning
#####

dataer = Dataer(dataset_name=args.dataset, dataset=get_goal_dataset(args.dataset))
resort_sequence = get_random_sequence(dataer.train_data_lenth, resort_lenth=int(0.2 * dataer.train_data_lenth), seed=args.seed)
dataer.set_sequence(sequence=resort_sequence)
neter = Neter(dataer=dataer, args=args, isTuning=args.isPretrain, pretrain_param=pretrain_param)

# pretrain model with delete [0, 1, 20, 50, ...] for saving time, where the consistence is maked sure by setting same seed. 
neter.load_model(generate_save_name(args, 0))
neter.initialization(isCover=True)  



# ########
# ### stage 2) pre calculate the matrix, store and load
# ########
if args.isBatchRemove == 1 or args.isBatchRemove == 2:
    muter = MUterRemover(basic_neter=neter, dataer=dataer, isDelta=True, remove_method='MUter', args=args)
else:
    muter = SchurMUterRemover(basic_neter=neter, dataer=dataer, isDelta=True, remove_method='MUter', args=args)

# fmuter = FMuterRemover(basic_neter=neter, dataer=dataer, isDelta=True, remove_method='FMUter', args=args)

newton_delta = NewtonRemover(basic_neter=neter, dataer=dataer, isDelta=True, remove_method='Newton_delta', args=args)
newton = NewtonRemover(basic_neter=neter, dataer=dataer, isDelta=False, remove_method='Newton', args=args)
influence_delta = InfluenceRemover(basic_neter=neter, dataer=dataer, isDelta=True, remove_method='Influence_delta', args=args)
influence = InfluenceRemover(basic_neter=neter, dataer=dataer, isDelta=False, remove_method='Influence', args=args)
fisher_delta = FisherRemover(basic_neter=neter, dataer=dataer, isDelta=True, remove_method='Fisher_delta', args=args)
fisher = FisherRemover(basic_neter=neter, dataer=dataer, isDelta=False, remove_method='Fisher', args=args)


# ####
# stage 3) the unlearning request coming, do unlearning and measure the metrics.
# stage 4) post of unlearning 
# ####

for index, remain_head in enumerate(remove_squence):

    remove_head = 0
    if index > 0:
        remove_head = remove_squence[index - 1]
    print('Unlearning deomain [{} -- {})'.format(remove_head, remain_head))

    # 1) for retrain
    retrain_neter = Neter(dataer=dataer, args=args, isTuning=args.isPretrain, pretrain_param=pretrain_param)
    retrain_neter.load_model(generate_save_name(args, remain_head)) 
    recorder.metrics_clean_acc_record('retrain', retrain_neter.test(isAttack=False))
    recorder.metrics_perturbed_acc_record('retrain', retrain_neter.test(isAttack=True))

    # 2) for MUter
    muter.Unlearning(head=remove_head, rear=remain_head)
    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=muter)

    ## 3) for Newton_delta, Newton
    newton_delta.Unlearning(head=remove_head, rear=remain_head)
    newton.Unlearning(head=remove_head, rear=remain_head)

    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=newton_delta)
    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=newton)

    ## 4) for Influence_delta, Influence
    influence_delta.Unlearning(head=remove_head, rear=remain_head)
    influence.Unlearning(head=remove_head, rear=remain_head)

    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=influence_delta)
    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=influence)

    ## 5) for Fisher_delta, Fisher
    fisher_delta.Unlearning(head=remove_head, rear=remain_head)
    fisher.Unlearning(head=remove_head, rear=remain_head)

    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=fisher_delta)
    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=fisher)


# save information
recorder.save()

if args.isBatchRemove in [0, 1]:

    print('Total Summary:')
    print('=='*20)

    print('Clean Accuracy:')
    print(recorder.clean_acc_dict)
    print('**'*20)

    print('Norm Distance')
    print(recorder.distance_dict)
    print('**'*20)

    print('Perturbed Accuracy')
    print(recorder.perturbed_acc_dict)
    print('**'*20)

elif args.isBatchRemove == 3:
    noise_list = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    

    noise_muter = Get_metrics(dataer, args, muter, retrain_neter, noise_list)
    print(noise_muter)

    noise_newton = Get_metrics(dataer, args, newton, retrain_neter, noise_list)
    print(noise_newton)

    noise_newton_delta = Get_metrics(dataer, args, newton_delta, retrain_neter, noise_list)
    print(noise_newton_delta)



    noise_influence = Get_metrics(dataer, args, influence, retrain_neter, noise_list)
    print(noise_influence)

    noise_influence_delta = Get_metrics(dataer, args, influence_delta, retrain_neter, noise_list)
    print(noise_influence_delta)


    noise_fisher = Get_metrics(dataer, args, fisher, retrain_neter, noise_list)
    print(noise_fisher)

    noise_fisher_delta = Get_metrics(dataer, args, fisher_delta, retrain_neter, noise_list)
    print(noise_fisher_delta)

