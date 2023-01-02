"""
where we abstract the remove way as class, the mainly fun here about such:
0)attribute: id(about what method I used to calculte the matrix and clean or perturbed?) is need for extend?? 
1)attribute: matrix or sub_matrix or layer_matrix(load and store the matrix, using his inverse with the grad to get the \delta_w)
2)attribute: clean_grad
3)attribute: perturbed_grad
4)method: load/save matrix
5)method: calculate the \delta_w
6)attribute: neter(manager the network), for neter(need to add function used to update paramter)
7)method: test model (get from neter)
"""

from builtins import print, super
from unittest import result
from tqdm import tqdm
from Train import Neter
import torch
import torch.nn as nn
import os
from utils import paramters_to_vector, total_param, cg_solve
from functorch import jacrev, make_functional, vmap
import math
from functorch import grad as fast_grad
import time
import copy

class Remover:
    """
    the basic scheme is by pretrain arch, args.isTuning == True, in this way, we code the code.
    """

    def __init__(self, basic_neter, dataer, isDelta, remove_method, args):
        
        self.basic_neter = basic_neter ## the origin neter, for reference and copy
        self.dataer = dataer
        self.isDelta = isDelta
        self.args = args
        self.remove_method = remove_method
        self.matrix = None
        self.grad = None
        self.delta_w = None
        self.path = None
        self.root_path = 'data/preMatrix/{}'.format(self.args.dataset)

        self.f_theta = [] # input for the last layer

        self.neter = None # init making the neter is None, everytime calling the unlearning method, register a new net copy from basic_neter
        # self.neter = Neter(dataer=self.dataer, args=self.args)
        # self.neter.net_copy(self.basic_neter) # deepcopy the basic_net's model parameters, only need to [update_parameters, test] is ok.


        if os.path.exists(self.root_path) == False:
            os.makedirs(self.root_path)
        
        # construct the save matrix path
        if self.isDelta:  
            self.path = os.path.join(self.root_path, '{}_delta.pt'.format(self.remove_method))        
        else:
            self.path = os.path.join(self.root_path, '{}.pt'.format(self.remove_method))

        if isDelta:
            self.basic_neter.save_adv_sample(isTrain=True)
        self.basic_neter.save_inner_output(isTrain=True, isAdv=True)

    def Load_matrix(self, ):
        
        if os.path.exists(self.path) == False:
            raise Exception('No such path for load matrix <{}>'.format(self.path))
        print('Loading matrix from <{}>'.format(self.path))
        self.matrix = torch.load(self.path)
        print('load done !')

    def Save_matrix(self, ):
        
        if self.matrix == None:
            raise Exception('No init the pre save matrix !')
        if self.isDelta:
            print('saving the {}_delta matrix to the path <{}> ...'.format(self.remove_method, self.path))
        else:
            print('saving the {} matrix to the path <{}> ...'.format(self.remove_method, self.path))
        
        torch.save(self.matrix, f=self.path)
        print('save done !')
        


    def get_pure_hessian(self, head=-1, rear=-1):
        """
        the pure hessian is \sum \partial_ww for the sample list [head ,rear)
        detail:
        1) using the sum_loss
        2) using the torch.autograd.grad and unit_vec to do this.
        """
        def get_unit_vec(vec, index):  # using the unit vector for HV operation.
            
            if index !=0:
                vec[index - 1] = 0
            vec[index] = 1
            return vec

        loader = self.dataer.get_loader( # get the inner output loader for the last layer
            head=head, 
            rear=rear, 
            batch_size=self.dataer.train_data_lenth,  # using the total_size and sum_loss to get the pure_hessian
            isAdv=self.isDelta,
            isInner=True
        )

        classifier = self.basic_neter.net.module.fc.to(self.basic_neter.device) #get the last layer
        params_number = total_param(classifier)
        unit_vec = torch.zeros(params_number).to(self.basic_neter.device)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        hessian_ww = None
        for inner_out, label in loader:
            inner_out = inner_out.to(self.basic_neter.device)
            label = label.to(self.basic_neter.device)
            output = classifier(inner_out)
            loss = criterion(output, label)

            grad_w = paramters_to_vector(torch.autograd.grad(loss, classifier.parameters(), create_graph=True, retain_graph=True)[0])
            grad_ww = [paramters_to_vector(torch.autograd.grad(grad_w, classifier.parameters(), retain_graph=True, grad_outputs=get_unit_vec(unit_vec, index))[0]) for index in tqdm(range(params_number))]

            hessian_ww = grad_ww

        hessian_ww = torch.stack(hessian_ww, 0).detach()

        return hessian_ww

    def get_fisher_matrix(self, head=-1, rear=-1, batch_size=4096):

        loader = self.dataer.get_loader(
            head=head,
            rear=rear,
            batch_size=batch_size,
            isAdv=self.isDelta,
            isInner=True
        )
        classifier = self.basic_neter.net.module.fc.to(self.basic_neter.device) #get the last layer
        func, classifier_params = make_functional(classifier)
        criterion = nn.CrossEntropyLoss()

        def compute_loss(params, inner_out, label):
            image = inner_out.unsqueeze(0)
            target = label.unsqueeze(0)
            output = func(params, image)
            return criterion(output, target)
        
        get_grad = fast_grad(compute_loss, argnums=0)
        batch_get_grad = vmap(get_grad, in_dims=(None, 0, 0))

        grad_list = []
        for (inner_out, label) in tqdm(loader):
            inner_out = inner_out.to(self.basic_neter.device)
            label = label.to(self.basic_neter.device)
            grad_list.append(batch_get_grad(classifier_params, inner_out, label)[0].detach().reshape(inner_out.shape[0], -1))
        
        grads = torch.cat(grad_list, 0)

        return grads.t().mm(grads)

    def get_sum_grad(self, head=-1, rear=-1, batch_size=128):

        loader = self.dataer.get_loader(
            head=head,
            rear=rear,
            batch_size=batch_size,
            isAdv=self.isDelta,
            isInner=True
        )
        classifier = self.basic_neter.net.module.fc.to(self.basic_neter.device) #get the last layer
        criterion = nn.CrossEntropyLoss(reduction='sum')

        grad_list = []
        
        for (inner_out, label) in tqdm(loader):
            inner_out = inner_out.to(self.basic_neter.device)
            label = label.to(self.basic_neter.device)
            output = classifier(inner_out)
            loss = criterion(output, label)
            grad_list.append(paramters_to_vector(torch.autograd.grad(loss, classifier.parameters())[0]))

        grad_tensor = torch.stack(grad_list, 0)
        
        return grad_tensor.sum(0)

    def Update_net_parameters(self, ):
        damp_cof = 0.0001
        damp_factor = damp_cof * torch.eye(self.matrix.shape[0]).cuda()
        delta_w = cg_solve(self.matrix + damp_factor, self.grad)
        # delta_w = torch.mv(torch.linalg.pinv(self.matrix), self.grad)
        self.neter.Reset_last_layer(delta_w)

        print('Update done !')

    def Unlearning(self, ):
        pass
    
class MUterRemover(Remover):
    """
    MUter using the remove function \delta_w = (\partial_ww - \partial_wx.\partial_xx^{-1}.\partial_xw)^{-1}.g
    method : init() to calculate the sum of total_hessian. For \partial_ww, we use the sum loss for samples to get, for \p_xx, \p_xw(wx),
    we use the vmap, jaccre from functorch to do this. difficulty: the \p_xx and \p_xw need to be replace by \partial_f_{\theta}f_{\theta} and so on to do this.
    """

    def __init__(self, basic_neter, dataer, isDelta, remove_method, args, mini_batch=128):
        
        super(MUterRemover, self).__init__(basic_neter, dataer, isDelta, remove_method, args)
        self.matrix = self.get_pure_hessian() - self.get_indirect_hessian(mini_batch=mini_batch)   ## total hessian
        self.Save_matrix()

    def get_indirect_hessian(self, head=-1, rear=-1, mini_batch=128, block_wise_size=20):
        """
        Args:
            head (int, optional): _description_. Defaults to -1.
            rear (int, optional): _description_. Defaults to -1.
            mini_batch (int, optional): _description_. Defaults to 128.
            block_wise_size (int, optional): _description_. Defaults to 100., using the block_wise strategy to generate indirect hessian, 
            which could economy cuda_memory.
        """

        # loader = self.dataer.get_loader( # get the inner output loader for the last layer
        #     head=head, 
        #     rear=rear, 
        #     batch_size=1,  # using the total_size and sum_loss to get the pure_hessian
        #     isAdv=self.isDelta,
        #     isInner=True
        # )
        classifier = self.basic_neter.net.module.fc.to(self.basic_neter.device) #get the last layer

        func, classifier_params = make_functional(classifier)
        criterion = nn.CrossEntropyLoss(reduction='mean')

        def compute_loss(params, sample, label):
            output = func(params, sample)
            return criterion(output, label)
        
        def Neumann(matrix, multipler, series, times=100):

            result = torch.zeros_like(matrix).cuda()

            for steps in range(times):
                result += series
                series = series.mm(multipler)
            return result

        def get_single_inverse(matrix, multipler, series):
            # return torch.linalg.pinv(matrix) ## after replaced by Neumann
            # I = 0.01 * torch.eye(matrix.shape[0]).cuda()  for inverse
            return Neumann(matrix, multipler, series)
            # return torch.linalg.inv(matrix + I)
            # return 2 * torch.eye(matrix.shape[0]).cuda() - matrix
        
        def get_single_indirect_hessian(partial_xx_inv, partial_xw):
            return partial_xw.t().mm(partial_xx_inv.mm(partial_xw))

        indirect_hessian = None

        if head == -1:
            head = 0
        if rear == -1:
            rear = self.dataer.train_data_lenth
        data_size = rear - head

        steps = math.ceil(data_size / block_wise_size)

        for i in tqdm(range(steps)):

            get_partial_xx = jacrev(jacrev(compute_loss, argnums=1), argnums=1)
            get_partial_xw = jacrev(jacrev(compute_loss, argnums=1), argnums=0)

            start = head + i * block_wise_size
            end = min(head + (i + 1) * block_wise_size, rear)
            loader = self.dataer.get_loader( # get the inner output loader for the last layer
                head=start, 
                rear=end, 
                batch_size=1,  # using the total_size and sum_loss to get the pure_hessian
                isAdv=self.isDelta,
                isInner=True
            )
            partial_xx_list = []
            partial_xw_list = []

            for (inner_out, label) in loader:
                inner_out = inner_out.to(self.basic_neter.device)
                inner_out.requires_grad = True
                label = label.to(self.basic_neter.device)

                partial_xx_list.append(get_partial_xx(classifier_params, inner_out, label)[0].detach().reshape(inner_out.shape[1], -1)) # TODO the output dimesion problem, need to be reshape
                partial_xw_list.append(get_partial_xw(classifier_params, inner_out, label)[0].detach().reshape(inner_out.shape[1], -1))
            

            partial_xx_tensor = torch.stack(partial_xx_list, 0) # expected tensor shape is [total_batch, x_size, s_size]
            partial_xw_tensor = torch.stack(partial_xw_list, 0) #expected tensor shape is [total_batch, x_size, w_size]

            del partial_xx_list
            del partial_xw_list

            batch_get_inverse = vmap(get_single_inverse)
            total_lenth = partial_xx_tensor.shape[0]
            batch_numbers = math.ceil(total_lenth / mini_batch)
            partial_xx_inv_list = []
            for dex in range(batch_numbers):
                sub_partial_xx_tensor = partial_xx_tensor[dex * mini_batch : min((dex + 1) * mini_batch, total_lenth)]
                identiy = torch.eye(sub_partial_xx_tensor.shape[1]).reshape((1, sub_partial_xx_tensor.shape[1], sub_partial_xx_tensor.shape[1]))
                batch_identiy = identiy.repeat(sub_partial_xx_tensor.shape[0], 1, 1).cuda()
                series = copy.deepcopy(batch_identiy).cuda()
                multipler = batch_identiy - sub_partial_xx_tensor
                partial_xx_inv_list.append(batch_get_inverse(sub_partial_xx_tensor, multipler, series))
                
                del batch_identiy
                del multipler
                del series

            # partial_xx_inv_list = [batch_get_inverse(partial_xx_tensor[dex * mini_batch : min((dex + 1) * mini_batch, total_lenth)]) for dex in range(batch_numbers)]
            
                
            partial_xx_inv_tensor = torch.cat(partial_xx_inv_list, 0)
            
            del partial_xx_inv_list
            del partial_xx_tensor
            
            batch_get_indirect_hessian = vmap(get_single_indirect_hessian, in_dims=(0, 0))
            indirect_hessian_list = [batch_get_indirect_hessian(
                partial_xx_inv_tensor[dex * mini_batch : min((dex + 1) * mini_batch, total_lenth)],
                partial_xw_tensor[dex * mini_batch : min((dex + 1) * mini_batch, total_lenth)]
            ).sum(0) for dex in range(batch_numbers)]

            indirect_hessian_tensor = torch.stack(indirect_hessian_list, 0)
            del indirect_hessian_list
            del partial_xx_inv_tensor
            del partial_xw_tensor

            if indirect_hessian == None:
                indirect_hessian = indirect_hessian_tensor.sum(0).detach()
            else:
                indirect_hessian += indirect_hessian_tensor.sum(0).detach()
            torch.cuda.empty_cache()

        return indirect_hessian

    def Unlearning(self, head, rear, mini_batch=128):
        

        start = time.time()

        if self.neter != None:
            temp = self.neter
            self.neter = None
            del temp

        self.neter = Neter(dataer=self.dataer, args=self.args)
        self.neter.net_copy(self.basic_neter) # deepcopy the basic_net's model parameters, only need to [update_parameters, test] is ok.

        self.Load_matrix()
        self.matrix -= (self.get_pure_hessian(head=head, rear=rear) - self.get_indirect_hessian(head=head, rear=rear, mini_batch=mini_batch))
        if self.grad == None:
            self.grad = self.get_sum_grad(head=head, rear=rear)
        else:
            self.grad += self.get_sum_grad(head=head, rear=rear)
        self.Save_matrix()

        self.Update_net_parameters()

        end = time.time()

        return (end - start)

class FMuterRemover(MUterRemover):

    def __init__(self, basic_neter, dataer, isDelta, remove_method, args, mini_batch=128):
        
        super(MUterRemover, self).__init__(basic_neter, dataer, isDelta, remove_method, args)
        self.matrix = self.get_fisher_matrix() - self.get_indirect_hessian(mini_batch=mini_batch)   ## total hessian
        self.Save_matrix() 

    def Unlearning(self, head, rear, mini_batch=128):
        start = time.time()

        if self.neter != None:
            temp = self.neter
            self.neter = None
            del temp

        self.neter = Neter(dataer=self.dataer, args=self.args)
        self.neter.net_copy(self.basic_neter) # deepcopy the basic_net's model parameters, only need to [update_parameters, test] is ok.

        self.Load_matrix()
        self.matrix -= (self.get_fisher_matrix(head=head, rear=rear) - self.get_indirect_hessian(head=head, rear=rear, mini_batch=mini_batch))
        if self.grad == None:
            self.grad = self.get_sum_grad(head=head, rear=rear)
        else:
            self.grad += self.get_sum_grad(head=head, rear=rear)
        self.Save_matrix()

        self.Update_net_parameters()

        end = time.time()

        return (end - start)

class SchurMUterRemover(MUterRemover):
    """extend from MUter, need his get_indirect_hessian.

    Args:
        MUterRemover (_type_): _description_
    """
    def __init__(self, basic_neter, dataer, isDelta, remove_method, args, mini_batch=128):
        # like MUter, the pre work save the total D_ww
        super(MUterRemover, self).__init__(basic_neter, dataer, isDelta, remove_method, args) 
    
    def get_indirect_second_order_info(self, head=-1, rear=-1):
        """mainly return the \partial_wx, \partial_xx, \partial_xw.
        

        Args:
            head (int, optional): _description_. Defaults to -1.
            rear (int, optional): _description_. Defaults to -1.
        """
        def compute_loss(params, sample, label):
            output = func(params, sample)
            return criterion(output, label)
        
        if head + 1 != rear:
            raise Exception('The head, rear value not satisfy the requirement !')

        classifier = self.basic_neter.net.module.fc.to(self.basic_neter.device) #get the last layer

        func, classifier_params = make_functional(classifier)
        criterion = nn.CrossEntropyLoss(reduction='mean')

        loader = self.dataer.get_loader( # get the inner output loader for the last layer
            head=head, 
            rear=rear, 
            batch_size=1,  # using the total_size and sum_loss to get the pure_hessian
            isAdv=self.isDelta,
            isInner=True
        )
        get_partial_xx = jacrev(jacrev(compute_loss, argnums=1), argnums=1)
        get_partial_xw = jacrev(jacrev(compute_loss, argnums=1), argnums=0)

        partial_xw = None
        partial_xx = None

        for (inner_out, label) in loader:
            inner_out = inner_out.to(self.basic_neter.device)
            inner_out.requires_grad = True
            label = label.to(self.basic_neter.device)

            partial_xx = get_partial_xx(classifier_params, inner_out, label)[0].detach().reshape(inner_out.shape[1], -1)
            partial_xw = get_partial_xw(classifier_params, inner_out, label)[0].detach().reshape(inner_out.shape[1], -1)

        # return like H12, H21, H22
        return partial_xw.t(), partial_xw, -partial_xx

    # buiding the block matrix
    def buliding_matrix(self, M, H_11, H_12, H_22, H_21):
        """
        version for torch
        using gpu without data change between gpu and cpu
        M: memory matrix [w_size, w_size]
        H_11: partial_ww [w_size, w_size]
        H_12: partial_wx [w_size, x_size]
        H_22: partial_xx [x_size, x_size]
        H_21: partial_xw [x_size, w_size]

        retrun | M-H_11    H_12 |
               |  H21      H_22 |
        """
        device = 'cuda'
        A = torch.cat([torch.cat([M-H_11, H_12], dim=1), torch.cat([H_21, H_22], dim=1)], dim=0).to(device)
        print('black matrix A shape {}, type {}'.format(A.shape, type(A)))
        return A.detach()


    def Unlearning(self, head=-1, rear=-1, mini_batch=128):
        """
        where the method here split into two part, 
        head-->rear have at least one point, if rear-head==1, just into part_2, else first into part_1 to do pre delete operation
        delete about (rear-head-1) points' influnce for M[D_{ww}]
        1) part_1: like MUterRemover Unlearning, remove the (rear-head-1) points' influence.
        2) part_2: using the block_matrix function to do unlearning, the update_net_param need to be rewrite.
        Args:
            head (int, optional): _description_. Defaults to -1.
            rear (int, optional): _description_. Defaults to -1.
            mini_batch (int, optional): _description_. Defaults to 128.
        """
        if head == -1:
            head = 0
        if rear == -1:
            rear = self.dataer.train_data_lenth
        
        self.Load_matrix()
        ## part 1
        if rear - head > 1: # if > 1, into part_1
            self.matrix -= (self.get_pure_hessian(head=head, rear=rear-1) - self.get_indirect_hessian(head=head, rear=rear-1, mini_batch=mini_batch))
        
        if self.grad == None:
            self.grad = self.get_sum_grad(head=head, rear=rear)
        else:
            self.grad += self.get_sum_grad(head=head, rear=rear)

        ## part 2 
        strat_time = time.time()

        H_11 = self.get_pure_hessian(head=rear-1, rear=rear)
        H_12, H_21, H_22 = self.get_indirect_second_order_info(head=rear-1, rear=rear)
        block_matrix = self.buliding_matrix(self.matrix, H_11, H_12, H_22, H_21)
        grad_extension = torch.cat((self.grad, torch.zeros(H_22.shape[0]).to(self.basic_neter.device)), dim=0)

        if self.neter != None:
            temp = self.neter
            self.neter = None
            del temp

        self.neter = Neter(dataer=self.dataer, args=self.args)
        self.neter.net_copy(self.basic_neter) # deepcopy the basic_net's model parameters, only need to [update_parameters, test] is ok.

        # damp_cof = 0.0001
        # damp_factor = damp_cof * torch.eye(self.matrix.shape[0]).cuda()
        delta_w = cg_solve(block_matrix, grad_extension)
        delta_w = delta_w[:H_11.shape[0]]
        # delta_w = torch.mv(torch.linalg.pinv(self.matrix), self.grad)
        self.neter.Reset_last_layer(delta_w)

        print('Update done !')

        end_time = time.time()
        # after the unealing stage, delete the last point influence.            
        self.matrix -= (self.get_pure_hessian(head=rear-1, rear=rear) - self.get_indirect_hessian(head=rear-1, rear=rear, mini_batch=mini_batch))
        self.Save_matrix()

        return end_time - strat_time

class NewtonRemover(Remover):

    def __init__(self, basic_neter, dataer, isDelta, remove_method, args):
        
        super(NewtonRemover, self).__init__(basic_neter, dataer, isDelta, remove_method, args)
        self.matrix = self.get_pure_hessian()
        self.Save_matrix()

    def Unlearning(self, head, rear):

        if self.neter != None:
            temp = self.neter
            self.neter = None
            del temp

        self.neter = Neter(dataer=self.dataer, args=self.args)
        self.neter.net_copy(self.basic_neter) # deepcopy the basic_net's model parameters, only need to [update_parameters, test] is ok.

        self.Load_matrix()

        self.matrix -= self.get_pure_hessian(head=head, rear=rear)
        if self.grad == None:
            self.grad = self.get_sum_grad(head=head, rear=rear)
        else:
            self.grad += self.get_sum_grad(head=head, rear=rear)

        self.Save_matrix()

        self.Update_net_parameters()

class InfluenceRemover(Remover):
    
    def __init__(self, basic_neter, dataer, isDelta, remove_method, args):
        
        super(InfluenceRemover, self).__init__(basic_neter, dataer, isDelta, remove_method, args)
        self.matrix = self.get_pure_hessian()
        self.Save_matrix()

    def Unlearning(self, head, rear): 

        if self.neter != None:
            temp = self.neter
            self.neter = None
            del temp

        self.neter = Neter(dataer=self.dataer, args=self.args)
        self.neter.net_copy(self.basic_neter) # deepcopy the basic_net's model parameters, only need to [update_parameters, test] is ok.

        self.Load_matrix()

        if self.grad == None:
            self.grad = self.get_sum_grad(head=head, rear=rear)
        else:
            self.grad += self.get_sum_grad(head=head, rear=rear)

        self.Update_net_parameters()

        self.matrix -= self.get_pure_hessian(head=head, rear=rear)

        self.Save_matrix()

class FisherRemover(Remover):

    def __init__(self, basic_neter, dataer, isDelta, remove_method, args, batch_size=4096):
        
        super(FisherRemover, self).__init__(basic_neter, dataer, isDelta, remove_method, args)
        self.matrix = self.get_fisher_matrix(batch_size=batch_size)
        self.Save_matrix()

    def Unlearning(self, head, rear):

        if self.neter != None:
            temp = self.neter
            self.neter = None
            del temp

        self.neter = Neter(dataer=self.dataer, args=self.args)
        self.neter.net_copy(self.basic_neter) # deepcopy the basic_net's model parameters, only need to [update_parameters, test] is ok.

        self.Load_matrix()

        self.matrix -= self.get_fisher_matrix(head=head, rear=rear)
        if self.grad == None:
            self.grad = self.get_sum_grad(head=head, rear=rear)
        else:
            self.grad += self.get_sum_grad(head=head, rear=rear)
            
        self.Save_matrix()

        self.Update_net_parameters()


