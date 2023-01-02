## using this python file pre retrain the model for using, 
## recording the retrain time, retrain model and so on.

import torch
import argparse
import os

from Train import Neter
from Recorder import Recorder
from data_utils import Dataer
from utils import get_layers, get_pretrain_model_path
from SISA import SISA
from utils import generate_save_name, get_random_sequence, get_BatchRemove_sequence, get_goal_dataset

"""
using to test and pre-fine-tuning model
"""
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='Cifar100')
# parser.add_argument('--remove_batch', type=int, default=2500, help='using the mini batch remove method')
# parser.add_argument('--remove_numbers', type=int, default=10000, help='total number for delete')
parser.add_argument('--epochs', type=int, default=150, help='custom the training epochs')
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
# parser.add_argument('--pretrain_path', default='data/model/pretrain_model/imagenet_wrn_baseline_epoch_', type=str)
# parser.add_argument('--pretrain_path', default='data/model/pretrain_model/cifar100_wrn34_model_epoch_', type=str)
# parser.add_argument('--pretrain_path', default='Lacuna-100_wrn28_model_epoch', type=str)
# parser.add_argument('--pretrain_model_number', default=80, type=int)
parser.add_argument('--tuning_epochs', default=10, type=int)
parser.add_argument('--tuning_lr', default=0.1, type=float)
parser.add_argument('--tuning_layer', default='linear', type=str)
parser.add_argument('--isBias', default=False, type=bool)

# for repeat experiments
parser.add_argument('--seed', default=998, type=int, help='determate the remove data id')

parser.add_argument('--shards', default=5, type=int)
parser.add_argument('--isDK', default=0, type=int, help='0: no, 1: yes.')

args = parser.parse_args()


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

# #####
# # Stage 1) traninig a roubust model for unlearning (adding SISA)
# #####

remove_squence_dict = {
    0: get_BatchRemove_sequence(args, isPretrain=True),
    1: get_BatchRemove_sequence(args, isPretrain=True),
    7: [0, ], # test demo mode
}

remove_squence = remove_squence_dict[args.isBatchRemove]

dataer = Dataer(dataset_name=args.dataset, dataset=get_goal_dataset(args.dataset))
resort_sequence = get_random_sequence(dataer.train_data_lenth, resort_lenth=int(0.2 * dataer.train_data_lenth), seed=args.seed)
dataer.set_sequence(sequence=resort_sequence)


# neter = Neter(dataer=dataer, args=args, isTuning=args.isPretrain, pretrain_param=pretrain_param)

# # after pre save model, we could load model
# neter.load_model(name='final_retrain_model_ten_7500')

# print('Train acc: {:.2f}%'.format(neter.test(isTrainset=True) * 100))
# print('Test acc: {:.2f}%'.format(neter.test(isTrainset=False) * 100))
# print('Adv Train test acc: {:.2f}%'.format(neter.test(isTrainset=True, isAttack=True)*100))
# print('Adv Test acc: {:.2f}%'.format(neter.test(isTrainset=False, isAttack=True)*100))
# neter.initialization(isCover=True)  # init generate the adv samples, inner output files.

# sisaer = SISA(dataer=dataer, args=args, shards_num=5, slices_num=5)
# sisaer.Reload()
# sisaer.sisa_train(isAdv=True)
# sisaer.sisa_remove(sequence=[17211, ], isTrain=True, isAdv=True)


# test inner output acc 
# print('clean train acc {:.2f}%'.format(neter.test_inner_out_acc(isTrain=True, isAdv=False) * 100))
# print('adv train acc {:.2f}%'.format(neter.test_inner_out_acc(isTrain=True, isAdv=True) * 100))

# print('clean test acc {:.2f}%'.format(neter.test_inner_out_acc(isTrain=False, isAdv=False) * 100))
# print('adv test acc {:.2f}%'.format(neter.test_inner_out_acc(isTrain=False, isAdv=True) * 100))


# # pre save model
# time = neter.training(epochs=args.epochs, lr=args.lr, batch_size=args.batchsize, isAdv=True)
# print('time {:.2f}'.format(time))
# neter.save_model()


# ########
# ### stage 2) pre calculate the matrix, store and load
# ########


# ####
# stage 3) the unlearning request coming, do unlearning and measure the metrics.
# stage 4) post of unlearning 
# ####

for remain_head in remove_squence: ##TODO  !!! this remove_batch + 1 need to be remove_batch !!!

    print('remain head : {}'.format(remain_head))

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

    ## 1) for retrain
    retrain_neter = Neter(dataer=dataer, args=args, isTuning=args.isPretrain, pretrain_param=pretrain_param)
    # retrain_neter = Neter(dataer=dataer, args=args)
    # print(retrain_neter.test(isTrainset=True, isAttack=False))
    # print(retrain_neter.test(isTrainset=False, isAttack=False))
    # print(retrain_neter.test(isTrainset=True, isAttack=True))
    # print(retrain_neter.test(isTrainset=False, isAttack=True))
    # retrain_neter.load_model(generate_save_name(args, remain_head))
    spending_time = retrain_neter.training(args.epochs, lr=args.lr, batch_size=args.batchsize, head=remain_head, isAdv=True, isFinaltest=False)
    print('spending time : {:.2f} seconds'.format(spending_time))
    recorder.metrics_time_record(method='Retrain', time=spending_time)
    recorder.metrics_clean_acc_record('retrain', retrain_neter.test(isTrainset=False, isAttack=False))
    recorder.metrics_perturbed_acc_record('retrain', retrain_neter.test(isTrainset=False, isAttack=True))
    retrain_neter.save_model(name=generate_save_name(args, remain_head))
    # del retrain_neter

    ## 2) for SISA
    ## under construction

# save information
recorder.save()
