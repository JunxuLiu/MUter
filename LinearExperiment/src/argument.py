import argparse

def argument():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='binaryMnist', help='the dataset', choices=['binaryMnist', 'covtype', 'epsilon', 'gisette', 'ijcnn1', 'higgs', 'madelon', 'phishing', 'splice'])
    parser.add_argument('--lam', type=float, default=1e-4, help='the regular lamda')
    parser.add_argument('--adv', type=str, default='PGD', help='the adv type:FGSM, PGD, CLEAN', choices=['PGD', 'FGSM', 'CLEAN'])
    parser.add_argument('--model', type=str, default='logistic', help='logistic, ridge', choices=['logistic', 'ridge'])
    parser.add_argument('--times', type=int, default=0, help='repeat the experiments, a.k.a random seeds for reproduction.')
    parser.add_argument('--parllsize', type=int, default=128, help='for accelerate the calculate of partial_(wx@xx^{-1}@xw) by parll, for large feature data, the parllsize be small since the CUDA memory could not support')
    parser.add_argument('--batchsize', type=int, default=128, help='the adversial training batch size')
    parser.add_argument('--deletenum', type=int, default=0, help='point the total number for forget, which is depend on the dataset')
    parser.add_argument('--deletebatch', type=int, default=1, help='batch delete number')
    parser.add_argument('--iterneumann', type=int, default=3, help='int neumann approximate iterations')
    parser.add_argument('--isbatch', type=bool, default=False, help='decide what MUter method used for unlearning, block or just cg')
    parser.add_argument('--remove_type', type=int, default=2, help='0:one step one point, 1:one step multiple points, 2: multiple step one point, 3:multiple step multiple points')



    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    return args


def get_remove_list(dataset):

    ## compare with removal list [1, 2, 3, 4, 5, ~1%, ~2%, ~3%, ~4%, ~5%] 
    remove_list = None
    if dataset == 'binaryMnist':
        remove_list = [1, 2, 3, 4, 5, 120, 240, 360, 480, 600]  # for mnist
    elif dataset == 'phishing':
        remove_list = [1, 2, 3, 4, 5, 100, 200, 300, 400, 500]  # for phsihing
    elif dataset == 'madelon':
        remove_list = [1, 2, 3, 4, 5, 20, 40, 60, 80, 100]  # for madelon
    elif dataset == 'covtype':
        remove_list = [1, 2, 3, 4, 5, 5000, 10000, 15000, 20000, 25000]
    elif dataset == 'epsilon':
        remove_list = [1, 2, 3, 4, 5, 4000, 8000, 12000, 16000, 20000]
    else:
        remove_list = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]  # for splice

    return remove_list