import torch
import argparse
import os
import time

from Train import Neter
from Remover import MUterRemover, NewtonRemover, InfluenceRemover, FisherRemover, SchurMUterRemover, FMuterRemover
from Recorder import Recorder
from data_utils import Dataer
from utils import get_layers, get_pretrain_model_path
from SISA import SISA
from utils import get_random_sequence, generate_save_name, get_BatchRemove_sequence, get_goal_dataset
"""
mainly code for machine unlearning, un see the detail of
the concrete code about how to calculate the matrix and its inverse
or sub operation or save or load matrix and so on.
"""

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='Cifar100')
# parser.add_argument('--remove_batch', type=int, default=2500, help='using the mini batch remove method')
# parser.add_argument('--remove_numbers', type=int, default=10000, help='total number for delete')
parser.add_argument('--epochs', type=int, default=300, help='custom the training epochs')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batchsize', type=int, default=128, help='the traning batch size')
parser.add_argument('--times', type=int, default=0, help='do repeat experiments')
parser.add_argument('--gpu_id', default=3, type=int)
parser.add_argument('--ngpu', default=1, type=int)

# for remove type chose
parser.add_argument('--adv_type', type=str, default='PGD', help='the adv training type')
parser.add_argument('--isBatchRemove', type=int, default=2, help='0: no batch, Schur complement. 1: batch, Neumann')

# for pretrain type
parser.add_argument('--isPretrain', default=True, type=bool)
parser.add_argument('--layers', default=34, type=int, help='total number of layers')
parser.add_argument('--widen_factor', default=10, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')
# parser.add_argument('--pretrain_path', default='data/model/pretrain_model/imagenet_wrn_baseline_epoch_', type=str)
# parser.add_argument('--pretrain_path', default='data/model/pretrain_model/cifar100_resnet_baseline_epoch_', type=str)
# parser.add_argument('--pretrain_path', default='data/model/pretrain_model/cifar100_wrn34_model_epoch_', type=str)
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


remove_squence_dict = {
    0: get_BatchRemove_sequence(args, isPretrain=False),
    1: get_BatchRemove_sequence(args, isPretrain=False),
    2: [5, ]
}

remove_squence = remove_squence_dict[args.isBatchRemove]

"""
1) traninig a roubust model for unlearning (adding SISA)
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
# Stage 1) traninig a roubust model for unlearning (adding SISA)
#####

dataer = Dataer(dataset_name=args.dataset, dataset=get_goal_dataset(args.dataset))
if args.isDK == 0:
    resort_sequence = get_random_sequence(dataer.train_data_lenth, resort_lenth=int(0.2 * dataer.train_data_lenth), seed=args.seed, isSort=False)
else:
    resort_sequence = get_random_sequence(dataer.train_data_lenth, resort_lenth=int(0.2 * dataer.train_data_lenth), seed=args.seed, isSort=True)

dataer.set_sequence(sequence=resort_sequence)

neter = Neter(dataer=dataer, args=args, isTuning=args.isPretrain, pretrain_param=pretrain_param)

# after pre save model, we could load model
neter.load_model(generate_save_name(args, 0))



neter.initialization(isCover=True)  # init generate the adv samples, inner output files.

sisaer = SISA(dataer=dataer, args=args, shards_num=args.shards, slices_num=5)
# sisaer.Reload()
sisaer.sisa_train(isAdv=True)


# ########
# ### stage 2) pre calculate the matrix, store and load
# ########
if args.isBatchRemove == 1 or args.isBatchRemove == 2:
    muter = MUterRemover(basic_neter=neter, dataer=dataer, isDelta=True, remove_method='MUter', args=args)
else:
    muter = SchurMUterRemover(basic_neter=neter, dataer=dataer, isDelta=True, remove_method='MUter', args=args)

fmuter = FMuterRemover(basic_neter=neter, dataer=dataer, isDelta=True, remove_method='FMUter', args=args)

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
    spending_time = retrain_neter.training(args.epochs, lr=args.lr, batch_size=args.batchsize, head=remain_head)
    retrain_neter.load_model(generate_save_name(args, remain_head))
    recorder.metrics_time_record(method='Retrain', time=spending_time)

    # 2) for SISA
    # under construction
    if args.isBatchRemove == 0 and remove_head + 1 < remain_head:
        sisaer.manager.remove_indexs(sequence=resort_sequence[remove_head : remain_head - 1])
        remove_head = remain_head - 1

    start_time = time.time()
    sisaer.sisa_remove(sequence=resort_sequence[remove_head : remain_head ], isTrain=True, isAdv=True)
    end_time = time.time()

    recorder.metrics_time_record(method='SISA', time=(end_time - start_time))

    # 3) for MUter
    unlearning_time = muter.Unlearning(head=remove_head, rear=remain_head)

    recorder.metrics_time_record(method=muter.remove_method, time=unlearning_time)
    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=muter)

    # 3.1) for Fisher-MUter
    # fmuter_unlearning_time = fmuter.Unlearning(head=remove_head, rear=remain_head)

    # recorder.metrics_time_record(method=fmuter.remove_method+'{}'.format(remain_head), time=fmuter_unlearning_time)
    # recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=fmuter)

    ## 4) for Newton_delta, Newton
    newton_delta.Unlearning(head=remove_head, rear=remain_head)
    newton.Unlearning(head=remove_head, rear=remain_head)

    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=newton_delta)
    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=newton)


    ## 5) for Influence_delta, Influence
    influence_delta.Unlearning(head=remove_head, rear=remain_head)
    influence.Unlearning(head=remove_head, rear=remain_head)

    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=influence_delta)
    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=influence)


    ## 6) for Fisher_delta, Fisher
    fisher_delta.Unlearning(head=remove_head, rear=remain_head)
    fisher.Unlearning(head=remove_head, rear=remain_head)

    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=fisher_delta)
    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=fisher)


# save information
recorder.save()





