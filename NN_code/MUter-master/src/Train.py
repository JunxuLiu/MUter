from builtins import print
from ntpath import join
import os
import torch
import torch.nn as nn
from model.common_model import *
from model.wrn import WideResNet
from model.preactresnet import PreActResNet18
from model.resnet import ResNet18
from tqdm import tqdm
import numpy as np
import copy
import time
from torchattacks import FGSM, PGD
from utils import get_layers, DataPreProcess

class Neter:

    def __init__(self, dataer, args, criterion=nn.CrossEntropyLoss(), device='cuda', arch=None, isTuning=False, pretrain_param=None):
        """
        manage all the training ways.
        Args:
            dataer (_type_): _description_
            args (_type_): _description_
            criterion (_type_, optional): _description_. Defaults to nn.CrossEntropyLoss().
            device (str, optional): _description_. Defaults to 'cuda'.
            arch (_type_, optional): _description_. Defaults to None.
            isTuning (bool, optional): _description_. Defaults to False.
            pretrain_param (_type_, optional): _description_. Defaults to None, if not NOne, it is a dict, basiclly include(
                epochs, lr, root_path, the fine-tuning layer ways(linear, MLP ), and so on~.
            )
        """

        self.criterion = criterion
        self.dataer = dataer
        self.device = device
        self.args = args
        self.net = None
        self.isTuning = isTuning
        self.pretrain_param = pretrain_param
        self.dataPreprocess = DataPreProcess(self.args)
        self.default_path = './data'
        self.inner_output = []
        self.atk_info = {
            'Cifar10': (8/255, 2/255, 10),
            'Mnist': (2/255, 0.4/255, 20),
            'SVHN': (4/255, 2/255, 10),
            'Cifar100': (8/255, 2/255, 10),
            'ImageNet': (8/255, 2/255, 10),
            'Lacuna-100': (8/255, 2/255, 10),
        }
        if self.args.adv_type == 'FGSM':
            self.atk_info = {
                'Cifar10': (8/255, 8/255, 1),
                'MNist': (2/255, 2/255, 1),
                'SVHN': (4/255, 4/255, 1),
                'Cifar100': (8/255, 8/255, 1),
                'ImageNet': (8/255, 8/255, 1),
                'Lacuna-100': (8/255, 8/255, 1),
            }

        if self.isTuning == False:
            if dataer.dataset_name == 'Mnist':
                self.net = TestModel(784, 10).to(self.device)
            elif dataer.dataset_name == 'Cifar10':
                if arch == None:
                    self.net = ResNet(ResidualBlock).to(self.device) # default setting
                elif arch == 'vgg16':
                    self.net = vgg16().to(self.device)
                else:
                    raise Exception('No such arch called {} !'.format(arch))
            elif dataer.dataset_name == 'Cifar100':
                self.net = WideResNet(28, 100, 10, 0)
            elif dataer.dataset_name == 'Lacuna-100':
                self.net = WideResNet(28, 100, 10, 0)
            else:
                raise Exception('No suchh dataset called {}'.format(dataer.dataset_name))

            if args.ngpu > 0:
                self.net = torch.nn.DataParallel(self.net, device_ids=list(range(args.ngpu)))

        else: 
            # using pre train model
            if self.pretrain_param == None:
                raise Exception('Not get the pretrain infom !')
            if os.path.exists(self.pretrain_param['root_path']) == False:
                raise Exception('No such path for get the pretrain model in {}!'.format(self.pretrain_param['root_path']))

            if args.dataset == 'ImageNet':
                self.net = WideResNet(self.pretrain_param['layers'], 1000, self.pretrain_param['widen_factor'], dropRate=self.pretrain_param['droprate'])
            elif args.dataset in ['Cifar100', 'Lacuna-100']:
                # self.net = PreActResNet18(num_classes=100)
                self.net = WideResNet(self.pretrain_param['layers'], 100, self.pretrain_param['widen_factor'], dropRate=self.pretrain_param['droprate'])

            if args.ngpu > 0:
                self.net = torch.nn.DataParallel(self.net, device_ids=list(range(args.ngpu)))
            self.net.load_state_dict(torch.load(self.pretrain_param['root_path']))

            print('Pretrain model successfully load ({})!'.format(self.pretrain_param['root_path']))
            # freezing the pre layer
            for param in self.net.parameters():
                param.requires_grad = False
            self.net.module.fc = self.pretrain_param['new_last_layer'] # new_last_layer may be linear or MLP

        self.net = self.net.to(self.device)

    def net_copy(self, basic_neter):
        
        self.net = copy.deepcopy(basic_neter.net).to(self.device)
        self.default_path = None
        self.args = basic_neter.args


    def training(
        self, 
        epochs=100, 
        lr=0.001, 
        batch_size=128, 
        isAdv=False, 
        verbose=False, 
        head=-1, 
        rear=-1, 
        isSave=False, 
        isSISA=False, 
        SISA_info=None, 
        isFinaltest=False,
        ):

        # for training, learning from the way in 'using Pre-Training can improve model rebustness and uncertainly', using y=2x-1, $x \in [0, 1]$ into $y \in [-1, 1]$, changing the domain

        if isSISA:
            if SISA_info == None:
                raise Exception('The self Loader is None, please recheck !')
            train_loader = SISA_info['train_loader']
        else:
            train_loader = self.dataer.get_loader(batch_size=batch_size, isTrain=True, head=head, rear=rear)

        if isAdv:
            self.isAdv = True
            # here the PGD source code have been modified, we use adv_image * 2 - 1 replace the adv_image.
            atk = PGD(self.net, self.atk_info[self.args.dataset][0], self.atk_info[self.args.dataset][1], self.atk_info[self.args.dataset][2], data_process=self.dataPreprocess)

        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        ## for fine_tuing model
        if self.isTuning:
            epochs = self.pretrain_param['epochs']
            optimizer = torch.optim.SGD(self.net.parameters(), lr=self.pretrain_param['lr'])
        ## 
        self.net.train()
        start_time = time.time()
        for epoch in range(1, epochs+1):

            # self.update(optimizer=optimizer, epoch=epoch, isClose=(isSISA or self.isTuning))
            if epoch == 80 or epoch == 120:
                optimizer.param_groups[0]['lr'] *= 0.1

            lenth = len(train_loader)
            avg_loss = 0.0
            steps = 1
            with tqdm(total=lenth) as pbar:
                pbar.set_description('Epoch [{}/{}]  Lr {}'.format(epoch, epochs, optimizer.param_groups[0]['lr']))
                for (image, label) in train_loader:
                    image = image.to(self.device)
                    label = label.to(self.device)

                    if isAdv:
                        image = atk(image, label).to(self.device)

                    # output = self.net(image * 2 - 1)  # map domain from [0, 1] into [-1, 1]
                    output = self.net(self.dataPreprocess.processing(image))

                    loss = self.criterion(output, label)
                    avg_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix(loss='{:.4f}'.format(avg_loss / steps))
                    pbar.update(1)
                    steps += 1

            if epoch % 10 == 0 and isFinaltest == False:
                print('Train acc: {:.2f}%'.format(self.test(isTrainset=True) * 100))
                print('Test acc: {:.2f}%'.format(self.test(isTrainset=False) * 100))
                print('Adv Train test acc: {:.2f}%'.format(self.test(isTrainset=True, isAttack=True)*100))
                print('Adv Test acc: {:.2f}%'.format(self.test(isTrainset=False, isAttack=True)*100))
                # torch.save(self.net.state_dict(), f='Lacuna-100_wrn28_model_epoch{}'.format(epoch))
        
        end_time = time.time()

        if isFinaltest:
            print('Train acc: {:.2f}%'.format(self.test(isTrainset=True, self_loader=train_loader) * 100))
            print('Test acc: {:.2f}%'.format(self.test(isTrainset=False) * 100))
            print('Adv Train test acc: {:.2f}%'.format(self.test(isTrainset=True, isAttack=True, self_loader=train_loader)*100))
            print('Adv Test acc: {:.2f}%'.format(self.test(isTrainset=False, isAttack=True)*100))

        if isAdv and isSave:
            path = os.path.join(self.default_path, '{}'.format(self.args.dataset))
            if os.path.exists(path) == False:
                os.makedirs(path)
            
            atk.save(train_loader, save_path=os.path.join(path, 'sample.pt'), verbose=True)
        
        if isSISA:  # need save the slices model
            self.save_sisa_model(path=SISA_info['save_path'])

        return (end_time - start_time)
        
    def test(self, batch_size=128, isTrainset=False, isAttack=False, self_loader=None):
        
        loader = self.dataer.get_loader(batch_size=batch_size, isTrain=isTrainset)
        if self_loader != None:
            loader = self_loader

        if isAttack:
            atk = PGD(self.net, self.atk_info[self.args.dataset][0], self.atk_info[self.args.dataset][1], self.atk_info[self.args.dataset][2], data_process=self.dataPreprocess)

        total = 0
        correct = 0
        self.net.eval()
        for (image, label) in tqdm(loader):
            image = image.to(self.device)
            label = label.to(self.device)

            if isAttack:
                image = atk(image, label).to(self.device)

            # output = self.net(image * 2 - 1)
            output = self.net(self.dataPreprocess.processing(image))

            _, pred = torch.max(output.data, 1)
            total += image.shape[0]
            correct += (pred == label).sum()
        
        self.net.train()
        return float(correct) / total
    
    def get_pred(self, batch_size=128, isTrain=False, isAttack=False):

        loader = self.dataer.get_loader(batch_size=batch_size, isTrain=isTrain)
        
        if isAttack:
            atk = PGD(self.net, self.atk_info[self.args.dataset][0], self.atk_info[self.args.dataset][1], self.atk_info[self.args.dataset][2], data_process=self.dataPreprocess)

        arr = []
        self.net.eval()
        for (image, label) in loader:
            image = image.to(self.device)
            label = label.to(self.device)

            if isAttack:
                image = atk(image, label).to(self.device)

            # output = self.net(image * 2 - 1)
            output = self.net(self.dataPreprocess.processing(image))

            _, pred = torch.max(output.data, 1)
            arr.append(pred)
        self.net.train()
        return arr

    def update(self, optimizer, epoch, multipler=0.1, isClose=False):
        
        if isClose:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = self.learning_rate_schedule(epoch=epoch, current_lr=param_group['lr'], multipler=multipler)

    def learning_rate_schedule(self, epoch, current_lr, multipler=0.1):

        lr_dict = {
            'Cifar10': [180, 240],
            'Mnist': []
        }

        if self.args.dataset not in lr_dict.keys():
            raise Exception('No such dataset in lr update schedule dict !')
        else:
            if epoch in lr_dict[self.args.dataset]:
                current_lr *= multipler
        return current_lr

    def save_adv_sample(self, batch_size=128, isTrain=True, isCover=False):
        
        atk = PGD(self.net, self.atk_info[self.args.dataset][0], self.atk_info[self.args.dataset][1], self.atk_info[self.args.dataset][2], data_process=self.dataPreprocess)
        path = os.path.join(self.default_path, '{}'.format(self.args.dataset))
        if os.path.exists(path) == False:
            os.makedirs(path)
        
        if isTrain:
            str = 'sample.pt'
        else:
            str = 'test_sample.pt'
        
        if os.path.exists(os.path.join(path, '{}'.format(str))):
            print('The adv_sample exists.')
            if isCover:
                print('Now re-cover the file !')
            else:
                return
        
        train_loader = self.dataer.get_loader(batch_size=batch_size, isTrain=isTrain)
        
        if isTrain:
            atk.save(train_loader, save_path=os.path.join(path, '{}'.format(str)), verbose=True)
        else:
            atk.save(train_loader, save_path=os.path.join(path, '{}'.format(str)), verbose=True)    
    
    def hook(self, module, input, output):

        self.inner_output.append(input[0].detach().cpu())

    def save_inner_output(self, batch_size=128, isTrain=True, isAdv=True, isCover=False):
        """
        generate the inner output samples and labels, if get the clean sample, not need the adv_samples, else the 
        first step is load the adv_samples and labels. 
        Args:
            batch_size (int, optional): _description_. Defaults to 128.
            isTrain (bool, optional): _description_. Defaults to True.
            isAdv (bool, optional): _description_. Defaults to True.
        """

        self.inner_output.clear()
        label_list = []

        if isAdv:
            if isTrain:
                str = 'sample.pt'
            else:
                str = 'test_sample.pt'
            if os.path.exists(os.path.join(self.default_path, '{}'.format(self.args.dataset), str)) == False:
                print('Not save the adv_sample, now saving...')
                self.save_adv_sample(isTrain=isTrain)

        if isAdv:
            adv_str = 'adv'
        else:
            adv_str = 'clean'
        if isTrain:
            if os.path.exists(os.path.join(self.default_path, '{}'.format(self.args.dataset), '{}_inner_output.pt'.format(adv_str))):
                print('The inner output sample exits.')
                if isCover:
                    print('Now re-cover the file !')
                else:
                    return
        else:
            if os.path.exists(os.path.join(self.default_path, '{}'.format(self.args.dataset), 'test_{}_inner_output.pt'.format(adv_str))):
                print('The inner output sample exits.')
                if isCover:
                    print('Now re-cover the file !')
                else:               
                    return

        loader = self.dataer.get_loader(batch_size=batch_size, isTrain=isTrain, isAdv=isAdv)

        handle = self.net.module.fc.register_forward_hook(self.hook)
        
        correct = 0
        total = 0

        self.net.eval()

        for (image, label) in tqdm(loader):
            image = image.to(self.device)
            label = label.to(self.device)
            # output = self.net(image * 2 - 1)            
            output = self.net(self.dataPreprocess.processing(image))
            label_list.append(label)
            _, pred = torch.max(output.data, 1)
            total += image.shape[0]
            correct += (pred == label).sum()
        
        self.net.train()

        print('The acc is {:.2f}%'.format(float(correct) / total * 100))

        inner_output_cat = torch.cat(self.inner_output, 0)
        label_list_cat = torch.cat(label_list, 0)


        if isTrain:
            torch.save((inner_output_cat, label_list_cat), os.path.join(self.default_path, '{}'.format(self.args.dataset), '{}_inner_output.pt'.format(adv_str)))
        else:
            torch.save((inner_output_cat, label_list_cat), os.path.join(self.default_path, '{}'.format(self.args.dataset), 'test_{}_inner_output.pt'.format(adv_str)))
        
        print('save done !')

        handle.remove()
        self.inner_output.clear()

    def save_sisa_model(self, path):

        torch.save(self.net.state_dict(), f=path)
        print('save in <{}>'.format(path))

    def load_sisa_model(self, path):

        if os.path.exists(path) == False:
            print('Not find the corespnds file path <{}>, please recheck !'.format(path))
        
        self.net.load_state_dict(torch.load(f=path))
        print('load done !')

    def save_model(self, name=None):

        if os.path.exists(os.path.join(self.default_path, '{}'.format(self.args.dataset))) == False:
            os.makedirs(os.path.join(self.default_path, '{}'.format(self.args.dataset)))

        if name == None:
            torch.save(self.net.state_dict(), f=os.path.join(self.default_path, '{}'.format(self.args.dataset), 'model_{}.pt'.format(self.args.tuning_epochs)))
            print('save done !')
        else:
            torch.save(self.net.state_dict(), f=os.path.join(self.default_path, '{}'.format(self.args.dataset), '{}.pt'.format(name)))
            print('save done !')

    def load_model(self, name=None):
        
        if name == None and os.path.exists(os.path.join(self.default_path, '{}'.format(self.args.dataset), 'model_{}.pt'.format(self.args.tuning_epochs))) == False:
            raise Exception('Not save the model, please recheck !')

        if name == None:
            self.net.load_state_dict(torch.load(f=os.path.join(self.default_path, '{}'.format(self.args.dataset), 'model_{}.pt'.format(self.args.tuning_epochs))))
            print('load done !')
        else:
            self.net.load_state_dict(torch.load(f=os.path.join(self.default_path, '{}'.format(self.args.dataset), '{}.pt'.format(name))))
            print('load done !')

    def Reset_last_layer(self, delta_w):
        
        head = 0
        for param in self.net.module.fc.parameters():
            num_params = np.prod(param.shape)
            param.data += delta_w[head : head + num_params].view_as(param.data).to(self.device)
            head += num_params

    def Reset_model_parameters_by_layers(self, delta_w):
        """
        delta_w is a dict, have the same name like the net itself.
        """
        for name, param in self.net.named_parameters():
            param.data += delta_w[name].view_as(param.data)
        
        print('layers update done !')
    
    def Reset_model_parameters_by_vector(self, delta_w):

        head = 0
        
        for param in self.net.parameters():
            numbers = np.prod(param.data.shape)
            param.data += delta_w[head : head + head + numbers]
            head += numbers

        print('vector update done !')
    
    def initialization(self, isCover=False):
        """
        generate the adv_train/test_inner_sample and clean _train/test_inner_sample
        """

        self.save_adv_sample(isTrain=True, isCover=isCover)
        self.save_adv_sample(isTrain=False, isCover=isCover)

        self.save_inner_output(isTrain=True, isAdv=True, isCover=isCover)
        self.save_inner_output(isTrain=True, isAdv=False, isCover=isCover)
        self.save_inner_output(isTrain=False, isAdv=True, isCover=isCover)
        self.save_inner_output(isTrain=False, isAdv=False, isCover=isCover)

    def test_inner_out_acc(self, isTrain=True, isAdv=True):

        loader = self.dataer.get_loader(
            isTrain=isTrain,
            isAdv=isAdv,
            isInner=True,
        )

        classifier = self.net.module.fc
        total = 0
        correct = 0

        self.net.eval()

        for (inner_out, label) in loader:
            inner_out = inner_out.to(self.device)
            label = label.to(self.device)
            output = classifier(inner_out)
            _, pred = torch.max(output.data, 1)
            total += inner_out.shape[0]
            correct += (pred == label).sum()
        
        self.net.train()
        
        return float(correct) / total
    
    
