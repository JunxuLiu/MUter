import torch
from Train import Neter
import numpy as np


def get_distance(retrain_neter, neter):

    distance = torch.tensor(0.0).cuda()
    retrain_fc = retrain_neter.net.module.fc
    compared_fc = neter.net.module.fc
    for (paramA, paramB) in zip(retrain_fc.parameters(), compared_fc.parameters()):
        distance += (paramA.data - paramB.data).pow(2.0).sum().detach()
    return distance.sqrt().cpu().numpy()

def Get_metrics(
    dataer,
    args,
    remover,
    retrain_neter,
    noise_list=[0.1, ], 
):
    
    dicter = {
        'remove_method': remover.remove_method,
        'noise_list': noise_list,
        'clean_acc': [],
        'perturbed_acc': [],
        'distance': [],
    }

    for noise_std in noise_list:

        neter = Neter(dataer=dataer, args=args)
        neter.net_copy(remover.neter)
        neter.Adding_noise(noise_std)
        dicter['clean_acc'].append(neter.test(isAttack=False))
        dicter['perturbed_acc'].append(neter.test(isAttack=True))
        dicter['distance'].append(get_distance(retrain_neter, neter))

    return dicter

