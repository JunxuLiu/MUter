from logging import handlers
import os
from sys import stderr
from builtins import ValueError
from matplotlib.transforms import Transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from TempArgs import args
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
            # return (image - self.mu) / self.std
            return image * 2 - 1
        elif self.args.dataset == 'ImageNet':
            return image * 2 - 1
        elif self.args.dataset == 'Lacuna-100':
            return image 
            
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

def Transfrom_string(str, args):

    prefix = '{}_{}_'.format(args.adv_type, args.isBatchRemove)

    method_sequence = [
        'MUter',
        'Newton_delta',
        'Influence_delta',
        'Fisher_delta',
        'Newton',
        'Influence',
        'Fisher',
        'FMUter',
        'retrain',
    ]

    for method in method_sequence:
        new_str = prefix + method
        if new_str == str:
            return method
    
    print('No match method !')
    return str

def Transform_to_dataframe(dicter, index_sequence, args, isIgnoreRetrain=True):
    """transform the dicter type into dataframe for plot pic.

    Args:
        index_sequence: should be a list
        dicter (_type_): {remove_method: [x_1, x_2, x_3,..., x_n-1, x_n]}
        return : df[coloum_1: method, coloum_2: index, coloum_3: value]
        method: remove way
        index: remove_number
        value: eval value
    """
    prefix = '{}_{}_'.format(args.adv_type, args.isBatchRemove)

    reTrans_dict = {
        'method': [],
        'index': [],
        'value': [],
    }


    for i, dex in enumerate(index_sequence):
        for key, value in dicter.items():
            if key == prefix + 'retrain' and isIgnoreRetrain == True:
                continue
            reTrans_dict['method'].append(Transfrom_string(key, args))
            reTrans_dict['index'].append(dex)
            reTrans_dict['value'].append(value[i])
    
    
    return pd.DataFrame(reTrans_dict)

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
    
    if args.isBatchRemove == 0:
        str += 'Schur_'
    else:
        str += 'Batch_'
    
    str += 'model_ten_{}_times{}'.format(remain_head, args.times)
    print('The name is : {}'.format(str))
    return str

def line_plot(df, metrics):

    # set plot style
    sns.set_style('darkgrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})

    if metrics == 'distance':
        markers = ['o' for i in range(7)]
    else:
        markers = ['o' for i in range(8)]

    ax = sns.lineplot(
        x='index', 
        y='value', 
        data=df, 
        hue='method', 
        style='method',
        markers=markers,
        dashes=False,
        ci=None,
    )

    plt.ylabel(metrics)
    plt.xlabel('Remove Numbers')

    plt.close()

    return ax

def trans_retrain(recorder):
    prefix = recorder.prefix
    recorder.clean_acc_dict[prefix + 'retrain'] = recorder.clean_acc_dict[prefix + 'retrain'][1:]
    recorder.perturbed_acc_dict[prefix + 'retrain'] = recorder.perturbed_acc_dict[prefix + 'retrain'][1: ]

def bar_plot(df, metrics):

    # set plot style
    sns.set_style('darkgrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
    # plt.figure(figsize=(21, 7))
    # ax = sns.barplot(
    #     data=df,
    #     x='index',
    #     y='value',
    #     hue='method',
    # )
    ax = sns.lineplot(
        data=df,
        x='index',
        y='value',
        hue='method'
    )
    plt.legend(title='Remove Numbers')
    plt.xlabel('')
    plt.xticks(fontsize=16)
    plt.ylabel(metrics)
    # plt.ylim(0.0, 0.2)
    
    ax.set_ylabel(ax.get_ylabel(), size=16)
    
    plt.setp(ax.get_legend().get_texts(), fontsize='16')
    plt.setp(ax.get_legend().get_title(), fontsize='16')


    plt.close()

    return ax

def Time_summary(args, times=[0, 1, 2]):

    recorder_list = []
    for i in times:
        temp = Recorder(args)
        temp.load(times=i)
        recorder_list.append(temp)
    
    statistics_dict = {}
    prefix = '{}_{}_'.format(args.adv_type, args.isBatchRemove)

    for recorder in recorder_list:
        for key, value in recorder.time_dict.items():
            if key not in statistics_dict:
                statistics_dict[key] = []
            if key == prefix + 'Retrain':  
                statistics_dict[key].append(value[1:])
            else:
                statistics_dict[key].append(value)

    result_list = []

    for key, value in statistics_dict.items():
        result_list.append([sum(e)/len(e) for e in zip(*value)])
    
    lenth = len(result_list[0])

    for index in range(lenth):
        str = ''
        for item in result_list:
            str += '&{:.2f} '.format(item[index])
        str += '&100.00 \\\\'
        print(str)

def Drawing_extension_time(args, times=[11, 12, 13]):

    prefix = '{}_{}_'.format(args.adv_type, args.isBatchRemove)
    sets = []
    sisa_name = ['SISA_shards5', 'SISA_shards10', 'SISA_shards20', 'SISA-DK_shards5', 'SISA-DK_shards10', 'SISA-DK_shards20', ]

    for time in times:
        temp = Recorder(args)
        time_method_list_1 = ['SISA_shards5_{}'.format(i*500) for i in range(1, 7)]
        # time_method_list_2 = ['SISA_shards10_{}'.format(i*500) for i in range(1, 7)]
        # time_method_list_3 = ['SISA_shards20_{}'.format(i*500) for i in range(1, 7)]

        time_method_list_4 = ['SISA-DK_shards5_{}'.format(i*500) for i in range(1, 7)]
        time_method_list_5 = ['SISA-DK_shards10_{}'.format(i*500) for i in range(1, 7)]
        time_method_list_6 = ['SISA-DK_shards20_{}'.format(i*500) for i in range(1, 7)]

        tupler = (time_method_list_1, 
                # time_method_list_2,
                # time_method_list_3,
                time_method_list_4,
                time_method_list_5,
                time_method_list_6,
                )

        arr = np.concatenate(tupler)
        temp.load(time_method_list=arr, times=time)

        dicter = {
            'method': [],
            'index': [],
            'value': [],
        }

        for method, lister in zip(sisa_name, tupler):
            for index, key in enumerate(lister):
                dicter['method'].append(method)
                dicter['index'].append((index + 1) * 500)
                dicter['value'].append(temp.time_dict[prefix + key][0])
        sets.append(dicter)
    print(sets)

def Drawing_fisher_muter(args, times=[8, 9, 10]):

    remove_sequence_dict = {
        0: [1, 200, 500, 1000, 2000, 4000],
        1: [2500, 5000, 7500, 10000],
    }
    remove_sequence = remove_sequence_dict[args.isBatchRemove]

    recorder_list = []
    for i in times:
        temp = Recorder(args)
        temp.load(times=i)
        recorder_list.append(temp)
    
    time_df_list = [Transform_to_dataframe(recorder.time_dict, remove_sequence, args) for recorder in recorder_list]


    time_df = pd.concat(time_df_list, ignore_index=True)

    # # set plot style
    # sns.set_style('darkgrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})

    markers = ['o' for i in range(2)]

    ax = sns.lineplot(
        x='index', 
        y='value', 
        data=time_df, 
        hue='method', 
        style='method',
        markers=markers,
        dashes=False,
        palette=['r', 'blue'],
        linewidth=2.5,
    )

    plt.ylabel('Time')
    plt.xlabel('Remove Numbers')
    plt.ylim(35, 55)

    # plt.close()

    return ax

def Drawing_summary(args, times=[0, 1, 2]):

    remove_sequence_dict = {
        0: [1, 200, 500, 1000, 2000, 4000],
        1: [2500, 5000, 7500, 10000],
    }
    remove_sequence = remove_sequence_dict[args.isBatchRemove]

    recorder_list = []
    for i in times:
        temp = Recorder(args)
        temp.load(times=i)
        acc_abs(temp)
        recorder_list.append(temp)

    disatnce_df_list = [Transform_to_dataframe(recorder.distance_dict, remove_sequence, args) for recorder in recorder_list]
    clean_acc_df_list = [Transform_to_dataframe(recorder.clean_acc_dict, remove_sequence, args) for recorder in recorder_list]
    perturbed_acc_df_list = [Transform_to_dataframe(recorder.perturbed_acc_dict, remove_sequence, args) for recorder in recorder_list]

    distance_df = pd.concat(disatnce_df_list, ignore_index=True)
    clean_acc_df = pd.concat(clean_acc_df_list, ignore_index=True)
    perturbed_acc_df = pd.concat(perturbed_acc_df_list, ignore_index=True)

    # return bar_plot(clean_acc_df, metrics='Clean Accuracy'), bar_plot(perturbed_acc_df, metrics='Perturbed Accuracy'), line_plot(distance_df, metrics='Distance')
    # return bar_plot(perturbed_acc_df, metrics='Perturbed Accuracy Gap')
    return bar_plot(clean_acc_df, metrics='Clean Accuracy Gap')
    # return line_plot(distance_df, metrics='Distance')

def time_convert_df_save(
  args,
  times=[0, 1, 2]  
):
    recorder_list = []
    for index in range(len(times)):
        temp_recorder = Recorder(args)
        temp_recorder.load(times=times[index])
        recorder_list.append(temp_recorder)


    for recorder in recorder_list:
        print(recorder.time_dict)
    # distance_df_list = [
    #     Transform_to_dataframe(recorder.distance_dict, get_BatchRemove_sequence(args, False), args, isIgnoreRetrain=True) for recorder in recorder_list
    # ]


    # df_distance = pd.concat(distance_df_list, ignore_index=True)
    # df_clean_acc = pd.concat(clean_acc_df_list, ignore_index=True)
    # df_perturbed_acc = pd.concat(perturbed_acc_df_list, ignore_index=True)

    # path = 'record/{}/Dfdata'.format(args.dataset)
    # if os.path.exists(path) == False:
    #     os.mkdir(path)
    # prefix = '{}_{}_'.format(args.adv_type, args.isBatchRemove)

    # df_distance.to_csv(os.path.join(path, prefix + 'distance.csv'))
    # df_clean_acc.to_csv(os.path.join(path, prefix + 'clean_acc.csv'))
    # df_perturbed_acc.to_csv(os.path.join(path, prefix + 'perturbed_acc.csv'))
    # print('Save Done !')


def convert_df_save(
    args, 
    times=[0, 1, 2],
):
    """
    the sort order is FGSM(isBatch0, isBatch1), PGD(isBatch0, isBatch1)
    """
    recorder_list = []
    for index in range(len(times)):
        temp_recorder = Recorder(args)
        temp_recorder.load(times=times[index])
        trans_retrain(temp_recorder)   ## retrain record the remain head zero, need to be remove.
        recorder_list.append(temp_recorder)

    distance_df_list = [
        Transform_to_dataframe(recorder.distance_dict, get_BatchRemove_sequence(args, False), args, isIgnoreRetrain=True) for recorder in recorder_list
    ]
    clean_acc_df_list = [
        Transform_to_dataframe(recorder.clean_acc_dict, get_BatchRemove_sequence(args, False), args, isIgnoreRetrain=False) for recorder in recorder_list
    ]
    perturbed_acc_df_list = [
        Transform_to_dataframe(recorder.perturbed_acc_dict, get_BatchRemove_sequence(args, False), args, isIgnoreRetrain=False) for recorder in recorder_list
    ]

    df_distance = pd.concat(distance_df_list, ignore_index=True)
    df_clean_acc = pd.concat(clean_acc_df_list, ignore_index=True)
    df_perturbed_acc = pd.concat(perturbed_acc_df_list, ignore_index=True)

    path = 'record/{}/Dfdata'.format(args.dataset)
    if os.path.exists(path) == False:
        os.mkdir(path)
    prefix = '{}_{}_'.format(args.adv_type, args.isBatchRemove)

    df_distance.to_csv(os.path.join(path, prefix + 'distance.csv'))
    df_clean_acc.to_csv(os.path.join(path, prefix + 'clean_acc.csv'))
    df_perturbed_acc.to_csv(os.path.join(path, prefix + 'perturbed_acc.csv'))
    print('Save Done !')


def FigPlot(
    save_name,
    adv_type='PGD',
    isBatchRemove=0,
    datasets_list=['Cifar10', 'Lacuna-100'],
):
    """
    len(datasets_list) rows, one row includes (distance, clean_acc, perturbed_acc) 
    """
    def translate(name):
        if name == 'Lacuna-100':
            return 'Lacuna-10'
        return name

    fig = plt.figure(figsize=(18, len(datasets_list) * 4))
    sns.set_style('whitegrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
    sns.set_palette(palette=sns.color_palette('bright'))

    metrics_list = ['distance', 'clean_acc', 'perturbed_acc']

    for index in range(len(datasets_list) * 3):

        ax = plt.subplot(len(datasets_list), 3, index + 1)
        
        dataset = datasets_list[index // 3]
        path = 'record/{}/Dfdata'.format(dataset)
        prefix = '{}_{}_'.format(adv_type, isBatchRemove)
        df = pd.read_csv(os.path.join(path, prefix + metrics_list[index % 3] + '.csv'))


        if metrics_list[index % 3] == 'distance':
            markers = ['o', 'v', '^', 's', 'v', '^', 's']
            palette=['#023EFF', '#1AC938', '#E8000B', '#8B2BE2', '#9F4800', '#F14CC1', '#A3A3A3']  
        else:
            markers = ['o', 'o', 'v', '^', 's', 'v', '^', 's']
            palette=['#FF7C00', '#023EFF', '#1AC938', '#E8000B', '#8B2BE2', '#9F4800', '#F14CC1', '#A3A3A3']
        
        if metrics_list[index % 3] == 'distance':
            temp = sns.lineplot(
                x='index',
                y='value',
                data=df,
                style='method',
                hue='method',
                markers=markers,
                dashes=False,
                linewidth=1.5,
                ax=ax,
                ci=None,
                palette=palette,
            )
        else:

            for dex, value in enumerate(df['method']):
                if value == 'retrain':
                    df['method'][dex] = 'Retrain'

            temp = sns.lineplot(
                x='index',
                y='value',
                data=df,
                style='method',
                hue='method',
                markers=markers,
                dashes=False,
                ci=None,
                linewidth=1.5,
                ax=ax,
                palette=palette
            )

        line_handles, line_labels = ax.get_legend_handles_labels()
        if metrics_list[index % 3] == 'distance':
            order_list = [0, 4, 5, 6, 1, 2, 3]
        else:
            order_list = [0, 1, 5, 6, 7, 2, 3, 4]
        ax.legend(handles=[line_handles[j] for j in order_list], labels=[line_labels[j] for j in order_list])

        if metrics_list[index % 3] == 'distance':
            plt.ylabel('Distance')
        elif metrics_list[index % 3] == 'clean_acc':
            plt.ylabel('Clean Accuracy')
        elif metrics_list[index % 3] == 'perturbed_acc':
            plt.ylabel('Perturbed Accuracy')
            
        plt.xlabel('Removal Numbers')
        # plt.title('{} {} BatchRemove type {}'.format(
        #         translate(dataset),
        #         adv_type,
        #         isBatchRemove,
        #     )
        # )

        temp.set_title(ax.get_title(), size=14)
        temp.set_ylabel(ax.get_ylabel(), size=14)
        temp.set_xlabel(ax.get_xlabel(), size=14) 

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig('{}.pdf'.format(save_name), dpi=800, bbox_inches='tight', pad_inches=0.2)
    print('Save Done !')
    plt.close()

if __name__ == "__main__":

    # for dataset in ['Cifar10', 'Cifar100', 'Lacuna-100']:
    #     for adv_type in ['FGSM', 'PGD']:
    #         for isBatchRemove in [0, 1]:
    #             convert_df_save(args=args(dataset, adv_type, isBatchRemove))

    # FigPlot(metrics='distance')
    # FigPlot(metrics='clean_acc')
    # FigPlot(metrics='perturbed_acc')
    
    FigPlot('Context_NN_Metrics', adv_type='PGD')
    FigPlot('Appendix_NN_Metrics', adv_type='FGSM')
    # time_convert_df_save(args(dataset='Lacuna-100', adv_type='FGSM', isBatchRemove=2))