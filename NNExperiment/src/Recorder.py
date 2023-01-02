import os
import torch
import time
import numpy as np

import matplotlib.pyplot as plt


class Recorder:

    def __init__(self, args):
        """
        Args:
            args (_type_): args have 'adv_type' and 'isBatchRemove', using this distingusih different remove method,
            like FGSM_1_Fisher_delta_times0.npy for batch, PGD_0_Newton_times2.npy for no batch
        """
        
        self.time_dict = {}
        self.clean_acc_dict = {}
        self.perturbed_acc_dict = {}
        self.distance_dict = {}
        self.args = args
        self.prefix = '{}_{}_'.format(self.args.adv_type, self.args.isBatchRemove)

        self.root_path = 'record/{}'.format(args.dataset)
        if os.path.exists(self.root_path) == False:
            os.makedirs(self.root_path)

        self.times = self.args.times
        

    def commom_log(self, str):
        
        path = os.path.join(self.root_path, 'info.txt')
        if os.path.exists(path) == False:
            os.makedirs(path)
        
        time_str = ''
        time_str += time.ctime()
        time_str += '\n'
        f = open(path, 'a+')
        f.write(time_str)
        f.write(str)
        f.close()

    
    def metrics_time_record(self, method, time):

        method = self.prefix + method
        if method not in self.time_dict:
            self.time_dict[method] = []
        
        self.time_dict[method].append(time)

    def metrics_clean_acc_record(self, method, acc):

        method = self.prefix + method
        if method not in self.clean_acc_dict:
            self.clean_acc_dict[method] = []

        self.clean_acc_dict[method].append(acc)

    def metrics_perturbed_acc_record(self, method, acc):

        method = self.prefix + method
        if method not in self.perturbed_acc_dict:
            self.perturbed_acc_dict[method] = []

        self.perturbed_acc_dict[method].append(acc)
    
    def metrics_distance_record(self, retrain_neter, compared_remover):

        method = self.prefix + compared_remover.remove_method
        if method not in self.distance_dict:
            self.distance_dict[method] = []

        distance = torch.tensor(0.0).cuda()
        # for name, param in retrain_neter.net.module.name_parameters():
        #     distance += (param.data - compared_remover.neter.net.name_parameters()[name].data).pow(2.0).sum().detach()
        retrain_fc = retrain_neter.net.module.fc
        compared_fc = compared_remover.neter.net.module.fc
        for (paramA, paramB) in zip(retrain_fc.parameters(), compared_fc.parameters()):
            distance += (paramA.data - paramB.data).pow(2.0).sum().detach()
        self.distance_dict[method].append(distance.sqrt().cpu().numpy())
    
    def log_metrics(self, retrain_neter, compared_remover):

        self.metrics_clean_acc_record(compared_remover.remove_method, compared_remover.neter.test(isTrainset=False, isAttack=False))
        self.metrics_perturbed_acc_record(compared_remover.remove_method, compared_remover.neter.test(isTrainset=False, isAttack=True))
        self.metrics_distance_record(retrain_neter=retrain_neter, compared_remover=compared_remover)


    def save(self):
        
        path = os.path.join(self.root_path, 'metrics')
        if os.path.exists(path) == False:
            os.makedirs(path)
        
        time_path = os.path.join(path, 'time')
        clean_acc_path = os.path.join(path, 'clean_acc')
        perturbed_acc_path = os.path.join(path, 'perturbed_acc')
        distance_path = os.path.join(path, 'distance')

        if os.path.exists(time_path) == False:
            os.makedirs(time_path)

        if os.path.exists(clean_acc_path) == False:
            os.makedirs(clean_acc_path)

        if os.path.exists(perturbed_acc_path) == False:
            os.makedirs(perturbed_acc_path)

        if os.path.exists(distance_path) == False:
            os.makedirs(distance_path)

        for key, value in self.time_dict.items():
            save_path = os.path.join(time_path, '{}_times{}.npy'.format(key, self.args.times))
            np.save(save_path, value)

        for key, value in self.clean_acc_dict.items():
            save_path = os.path.join(clean_acc_path, '{}_times{}.npy'.format(key, self.args.times))
            np.save(save_path, value)
            
        for key, value in self.perturbed_acc_dict.items():
            save_path = os.path.join(perturbed_acc_path, '{}_times{}.npy'.format(key, self.args.times))
            np.save(save_path, value)

        for key, value in self.distance_dict.items():
            save_path = os.path.join(distance_path, '{}_times{}.npy'.format(key, self.args.times))
            np.save(save_path, value)

    def load(
        self, 
        method_list = ['retrain', 'MUter', 'Newton_delta', 'Influence_delta', 'Fisher_delta', 'Newton', 'Influence', 'Fisher', 'FMUter'], 
        time_method_list=['Retrain', 'MUter', 'SISA', 'FMUter'],
        times=None,
        ):
        
        if times != None:
            self.times = times

        time_path = os.path.join(self.root_path, 'metrics', 'time')
        clean_acc_path = os.path.join(self.root_path, 'metrics', 'clean_acc')
        perturbed_acc_path = os.path.join(self.root_path, 'metrics', 'perturbed_acc')
        distance_path = os.path.join(self.root_path, 'metrics', 'distance')

        for method in method_list:

            method = self.prefix + method

            if os.path.exists(os.path.join(clean_acc_path, '{}_times{}.npy'.format(method, self.times))):
                self.clean_acc_dict[method] = np.load(os.path.join(clean_acc_path, '{}_times{}.npy'.format(method, self.times)))

            if os.path.exists(os.path.join(perturbed_acc_path, '{}_times{}.npy'.format(method, self.times))):
                self.perturbed_acc_dict[method] = np.load(os.path.join(perturbed_acc_path, '{}_times{}.npy'.format(method, self.times)))

            if os.path.exists(os.path.join(distance_path, '{}_times{}.npy'.format(method, self.times))):
                self.distance_dict[method] = np.load(os.path.join(distance_path, '{}_times{}.npy'.format(method, self.times)))

        for method in time_method_list:

            method = self.prefix + method

            if os.path.exists(os.path.join(time_path, '{}_times{}.npy'.format(method, self.times))):
                self.time_dict[method] = np.load(os.path.join(time_path, '{}_times{}.npy'.format(method, self.times)))




if __name__ == "__main__":

    pass