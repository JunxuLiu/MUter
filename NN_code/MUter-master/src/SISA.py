import torch
from Train import Neter
from data_utils import Dataer
import math
import numpy as np
import os
from functorch import vmap
from utils import get_layers, get_pretrain_model_path


# TODO After the remove ,the size become zero, need to be pay attention~
# TODO def get_majority(self, preds, index): have a range(10), nned to be a variable

class Manager:
    """
    setting for SISA, giving the arrange for shards and slices about add, delete, modify, search...
    """
    def __init__(self, shards_nums, slices_nums, total_lenth):

        self.shards_nums = shards_nums
        self.slices_nums = slices_nums
        self.total_lenth = total_lenth
        self.shards_info = {}

        self.shards_size = math.ceil(self.total_lenth / self.shards_nums)

        for index in range(self.shards_nums):
            self.shards_info[index] = {}
            self.shards_info[index]['id'] = '{}'.format(index)
            self.shards_info[index]['head'] = index * self.shards_size
            self.shards_info[index]['rear'] = min((index + 1) * self.shards_size, self.total_lenth)
            self.shards_info[index]['lenth'] = self.shards_info[index]['rear'] - self.shards_info[index]['head']
            
            slices_size = math.ceil(self.shards_info[index]['lenth'] / self.slices_nums)
            self.shards_info[index]['slices'] = []

            start = self.shards_info[index]['head']
            end = min(start + slices_size, self.shards_info[index]['rear'])
            for dex in range(self.slices_nums):
                self.shards_info[index]['slices'].append([item for item in range(start, end)])
                start = end
                end =  min(start + slices_size, self.shards_info[index]['rear'])
    
    def find(self, index, isGetSlicesDex=False):
        """
        give a index for the total dataset index, return the tuple about which shards, slices it is. 
        Args:
            index (_type_): _description_
        """
        shards_index = index // self.shards_size
        if index < self.shards_info[shards_index]['head'] or index >= self.shards_info[shards_index]['rear']:
            raise Exception('out of the limitation about the index, recheck the find method code !')
       
        if isGetSlicesDex:
            for dex in range(self.slices_nums):
                if index in self.shards_info[shards_index]['slices'][dex]:
                    return (shards_index, dex)

        return (shards_index, None) # the index may have been remove

    def get_incremental_list(self, shards_index, times):

        if times >= self.slices_nums:
            raise Exception('Out of the limitation of Slices numbers !')
        
        arr = self.shards_info[shards_index]['slices'][0]

        for dex in range(1, times + 1):
            arr = np.concatenate((arr, self.shards_info[shards_index]['slices'][dex]))

        return arr
    
    def remove_indexs(self, sequence):
        """
        return the list about the head of what shards/ slices have been be removed.
        Args:
            sequence (_type_): the sequence should be sorted. like [1, 3, 4, 32, 78..]
            return info: [(), (shard, slice_head), ...()]
        """
        if len(sequence) == 0:
            raise Exception('The remove sequence lenth is zero, please recheck !')
        
        info = []
        info.append(self.find(index=sequence[0], isGetSlicesDex=True))
        for item in sequence[1:]:
            temp = self.find(index=item, isGetSlicesDex=False)
            if(temp[0] != info[-1][0]):
                info.append(self.find(index=item, isGetSlicesDex=True))
        
        self.remove(sequence=sequence)

        return info

    def remove(self, sequence):

        if len(sequence) == 0:
            raise Exception('The remove sequence lenth is zero, please recheck !')
        
        for item in sequence:
            temp = self.find(index=item, isGetSlicesDex=True)
            self.shards_info[temp[0]]['slices'][temp[1]].remove(item)  # if after the remove, the size become zero, need to be sign!!!
            self.shards_info[temp[0]]['lenth'] -= 1

        print('Remove done !')



class SISA:

    def __init__(self, dataer, args, shards_num, slices_num):

        self.dataer = dataer
        self.args = args
        self.manager = Manager(shards_nums=shards_num, slices_nums=slices_num, total_lenth=self.dataer.train_data_lenth)
        self.model_list = []

        self.basic_path = 'data/SISA/{}'.format(self.dataer.dataset_name)

        if os.path.exists(self.basic_path) == False:
            os.makedirs(self.basic_path)

    def Reload(self, ):

        self.model_list.clear()

        for index in range(self.manager.shards_nums):
            pretrain_param = None
            if self.args.isPretrain:
                pretrain_param = {
                    'layers': self.args.layers,
                    'widen_factor': self.args.widen_factor,
                    'droprate': self.args.droprate,
                    'root_path': get_pretrain_model_path(self.args.dataset),
                    'new_last_layer': get_layers(self.args.tuning_layer, isBias=self.args.isBias),
                }

            sub_model = Neter(dataer=self.dataer, args=self.args, isTuning=self.args.isPretrain, pretrain_param=pretrain_param)
            sub_model.load_sisa_model(os.path.join(self.basic_path, 'shard{}_slice{}.pt'.format(index, self.manager.slices_nums - 1)))
            
            self.model_list.append(sub_model)
        
        print('sisa submodel load done !')

    def sisa_train(self, batch_size=128, isTrain=True, isAdv=False, verbose=False):
        
        """
        train sub_model by shards.
        """
        sub_epochs = math.ceil((self.args.tuning_epochs * 2) / (self.manager.shards_nums + 1))
        for index in range(self.manager.shards_nums):  #traverse the shards
            
            pretrain_param = None
            if self.args.isPretrain:
                pretrain_param = {
                    'layers': self.args.layers,
                    'widen_factor': self.args.widen_factor,
                    'droprate': self.args.droprate,
                    'root_path': get_pretrain_model_path(self.args.dataset),
                    'epochs': sub_epochs,
                    'lr': self.args.tuning_lr,
                    'new_last_layer': get_layers(self.args.tuning_layer, isBias=self.args.isBias),
                }

            sub_model = Neter(dataer=self.dataer, args=self.args, isTuning=self.args.isPretrain, pretrain_param=pretrain_param)

            for dex in range(self.manager.slices_nums):
                sequence = self.manager.get_incremental_list(shards_index=index, times=dex)
                train_loader = self.dataer.get_customized_loader(sequence=sequence, batch_size=batch_size, isTrain=True)
                train_info = {
                    'train_loader': train_loader,
                    'save_path': os.path.join(self.basic_path, 'shard{}_slice{}.pt'.format(index, dex)),
                }    
                sub_model.training(
                    lr=self.args.tuning_lr, 
                    batch_size=batch_size, 
                    isAdv=isAdv, 
                    verbose=verbose, 
                    isSISA=True, 
                    SISA_info=train_info,
                )
                
                if dex == self.manager.shards_nums - 1:
                    self.model_list.append(sub_model)
        

    def sisa_remove(self, sequence, batch_size=128, isTrain=True, isAdv=False, verbose=False):
    
        """
        retrain a shard or sub shards
        """
        sequence.sort()

        sub_epochs = math.ceil((self.args.tuning_epochs * 2) / (self.manager.shards_nums + 1))
        print('Sub Epochs is {}'.format(sub_epochs))
        retrain_info = self.manager.remove_indexs(sequence=sequence)

        print('Retrain following shards and slice ...')
        print(retrain_info)

        for (shard_index, slice_head) in retrain_info:
            pretrain_param = None
            if self.args.isPretrain:
                pretrain_param = {
                    'layers': self.args.layers,
                    'widen_factor': self.args.widen_factor,
                    'droprate': self.args.droprate,
                    'root_path': get_pretrain_model_path(self.args.dataset),
                    'epochs': sub_epochs,
                    'lr': self.args.tuning_lr,
                    'new_last_layer': get_layers(self.args.tuning_layer, isBias=self.args.isBias),
                }

            sub_model = Neter(dataer=self.dataer, args=self.args, isTuning=self.args.isPretrain, pretrain_param=pretrain_param) # load the savbe slice model
            if slice_head != 0:
                sub_model.load_sisa_model(os.path.join(self.basic_path, 'shard{}_slice{}.pt'.format(shard_index, slice_head - 1)))
            
            for dex in range(slice_head, self.manager.slices_nums):

                retrain_sequence = self.manager.get_incremental_list(shards_index=shard_index, times=dex)
                train_loader = self.loader = self.dataer.get_customized_loader(sequence=retrain_sequence, batch_size=batch_size, isTrain=isTrain)
                train_info = {
                    'train_loader': train_loader,
                    'save_path': os.path.join(self.basic_path, 'shard{}_slice{}.pt'.format(shard_index, dex))
                }
                sub_model.training(
                    lr=self.args.tuning_lr, 
                    batch_size=batch_size, 
                    isAdv=isAdv, 
                    verbose=verbose, 
                    isSISA=True, 
                    SISA_info=train_info,
                    isFinaltest=False,
                )
                
                if dex == self.manager.slices_nums - 1:
                    self.model_list[shard_index] = sub_model
        
        print('SISA Remove Done !')

    def sisa_test(self, batch_size=128, isTrain=False, isAttack=False):
        """
        using the common vote method to do this.
        """ 
        preds = []       
        for shard_index in range(self.manager.shards_nums):
            preds.append(self.model_list[shard_index].get_pred(batch_size=batch_size, isTrain=False, isAttack=isAttack))
            
        total = 0
        correct = 0
        loader = self.dataer.get_loader(batch_size=batch_size, isTrain=isTrain)
        for dex, (image, label) in enumerate(loader):
            label = label.to('cuda')

            total += label.shape[0]

            correct += (self.get_majority(preds, dex) == label).sum()

        return float(correct) / total
    


    def get_majority(self, preds, index):
        
        def get_most(arr):
            nums = [0 for i in range(10)]
            for item in arr:
                nums[item] += 1
            dex = nums.index(max(nums))

            return torch.tensor(dex)
            
        sub_pred = [item[index] for item in preds]
        mat = torch.stack(sub_pred, dim=0)
        mat = mat.transpose()

        get_mosts = vmap(get_most)

        return get_most(mat)


if __name__ == "__main__":

    manager = Manager(5, 5, 10000)
    info = manager.remove_indexs([2003, 2005, 3201, 3999])
    arr = manager.get_incremental_list(1, 4)
    print(len(arr))

    print(info)