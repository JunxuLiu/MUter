!\bin\bash
# python utils.py --dataset Cifar10 --adv_type FGSM --isBatchRemove 0
# python utils.py --dataset Cifar10 --adv_type FGSM --isBatchRemove 1
# python utils.py --dataset Cifar10 --adv_type PGD --isBatchRemove 0
# python utils.py --dataset Cifar10 --adv_type PGD --isBatchRemove 1

python utils.py --dataset Cifar100 --adv_type FGSM --isBatchRemove 0
# python utils.py --dataset Cifar100 --adv_type FGSM --isBatchRemove 1
# python utils.py --dataset Cifar100 --adv_type PGD --isBatchRemove 0
# python utils.py --dataset Cifar100 --adv_type PGD --isBatchRemove 1

# python utils.py --dataset Lacuna-100 --adv_type FGSM --isBatchRemove 0
# python utils.py --dataset Lacuna-100 --adv_type FGSM --isBatchRemove 1
# python utils.py --dataset Lacuna-100 --adv_type PGD --isBatchRemove 0
# python utils.py --dataset Lacuna-100 --adv_type PGD --isBatchRemove 1