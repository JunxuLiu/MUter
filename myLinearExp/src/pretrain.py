import torch
from torchattacks import PGD
from argument import argument
import time
from tqdm import tqdm, trange
from utils import LossFunction, get_featrue, training_param
from model import *
from dataloder import *

def train(train_loader, test_loader, args, desc='Pre-Adv Training', verbose=False, model_path=None, atk_path=None):
    
    if model_path and os.path.exists(model_path):
        saved = torch.load(model_path)
        return saved.get('atm'), saved.get('time')

    device = 'cuda'
    model = LogisticModel(input_featrue=get_featrue(args))

    model = model.to(device)
    lr, epochs, atk_info = training_param(args)

    if verbose:
        print('model information: ')
        print(model)
        print('training type: {}, epsilon: {:.5f}, alpha: {:.5f}, steps: {}'.format(args.adv, atk_info[0], atk_info[1], atk_info[2]))
        print('training hyperparameters  lr: {:.3f}, epochs: {} '.format(lr, epochs))    

    # loss function
    criterion = LossFunction(args.model).to(device)

    # setting optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # attack setting
    # PGDL2 could be regrade as FGSML2 when alpha=eps, steps=1
    atk = PGD(model, eps=atk_info[0], alpha=atk_info[1], steps=atk_info[2], lossfun=LossFunction(args.model), lam=args.lam)

    # atk training 
    start_time = time.time()
    model.train()
    with tqdm(range(epochs), desc=desc, bar_format="{l_bar}{bar:10}{r_bar}") as pbar:
        for epoch in pbar:
            # pbar.set_description(desc)
            # print('Epoch [{}/{}] training type {}, learning rate : {:.4f}'.format(epoch+1, epochs, args.adv, optimizer.param_groups[0]['lr']), end=' ')
            total_loss = 0.0
            step = 0
            for data, label in train_loader:
                label = label.to(device)
                if args.adv == 'PGD' or args.adv == 'FGSM':
                    data = atk(data, label).to(device)
                data = data.to(device)
                output = model(data)

                loss = criterion(output, label)

                if args.lam != 0.0:
                    lam = torch.tensor(0.5 * args.lam)
                    l2_reg = torch.tensor(0.0)
                    for param in model.parameters():
                        l2_reg = l2_reg + lam * param.pow(2.0).sum()
                    loss = loss + l2_reg
                
                optimizer.zero_grad()
                step = step + 1
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            
            pbar.set_postfix(adv_train_type=args.adv, model=args.model, lr=optimizer.param_groups[0]['lr'], loss=total_loss/step, times=args.times)
            # print('loss : {:.5f}  adv_type : {} model : {}  times : {}'.format(total_loss/step, args.adv, args.model, args.times))
            # time.sleep(0.1)

    end_time = time.time()
    training_time = end_time - start_time
    print('traning {} model spending {:.2f} seconds'.format(args.adv, training_time))
    
    if model_path:
        # Save adversarially trained model
        saved = dict(atm=model, time=training_time)
        torch.save(saved, model_path)

    if atk_path:
        # Save adversarial images as torch.tensor from given torch.utils.data.DataLoader
        atk.save(test_loader, atk_path, verbose=False)
    
    return model, training_time


def Test_model(model, test_loader, args):
    """
    return clean/perturb test acc
    """
    device = 'cuda'
    model = model.to(device)
    _, _, atk_info = training_param(args, isAttack=True)

    atk = PGD(model, atk_info[0], atk_info[1], atk_info[2], lossfun=LossFunction(args.model), lam=args.lam)

    model.eval()
    correct = 0
    total = 0

    # clean test acc
    for data, label in test_loader:
        label = label.to(device)
        data = data.to(device)

        predict = model(data).round()
        total = total + data.shape[0]
        correct = correct + (predict == label).sum()
    
    clean_test_acc = float(correct) / total

    print('clean test acc : {:.2f}%'.format(clean_test_acc * 100))


    correct = 0
    total = 0
    # perturb test acc
    for data, label in test_loader:
        label = label.to(device)
        data = atk(data, label).to(device)

        predict = model(data).round()
        total = total + data.shape[0]
        correct = correct + (predict == label).sum()  

    perturb_test_acc = float(correct) / total

    print('perturb test acc : {:.2f}%'.format(perturb_test_acc * 100))

    return clean_test_acc, perturb_test_acc

def test(model, test_loader):
    device = 'cuda'
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    for image, label in test_loader:
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        predict = output.sign()
        total = total + image.shape[0]
        correct = correct + (predict == label).sum()
    
    return float(correct) / total

if __name__ == "__main__":
    args = argument()
    train_data, test_data, _ = Load_Data(args)
    train_loader = make_loader(train_data, batch_size=128, head=100)
    test_loader = make_loader(test_data, batch_size=128)
    model, training_time = train(train_loader, test_loader, args, verbose=True)

