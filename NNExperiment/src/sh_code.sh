!\bin\bash

# Quick verification
## The forgetting script of the NN model is divided into two steps, and the following is the demo script.
## step1: python Pre_retrain_model.py; step2: python main.py

# first step
python Pre_retrain_model.py --times 0 --seed 666 --isBatchRemove 0 --adv_type PGD --dataset ImageNet --layers 28 --tuning_lr 0.001 --tuning_epochs 10 --gpu_id 0

# second step
python main.py --times 0 --seed 666 --isBatchRemove 0 --adv_type PGD --dataset ImageNet --layer 28 --tuning_lr 0.001 --gpu_id 0


## unlearning process

### for Lacuna-100->Lacuna-10 PGD

# python main.py --times 0 --seed 998 --isBatchRemove 0 --adv_type PGD --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 0
# python main.py --times 1 --seed 999 --isBatchRemove 0 --adv_type PGD --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 0
# python main.py --times 2 --seed 1000 --isBatchRemove 0 --adv_type PGD --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 0


### for Lacuna-100->Lacuna-10 FGSM

# python main.py --times 0 --seed 998 --isBatchRemove 0 --adv_type FGSM --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 0
# python main.py --times 1 --seed 999 --isBatchRemove 0 --adv_type FGSM --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 0
# python main.py --times 2 --seed 1000 --isBatchRemove 0 --adv_type FGSM --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 0

### for ImageNet->Cifar10 PGD

# python main.py --times 0 --seed 666 --isBatchRemove 0 --adv_type PGD --dataset ImageNet --layer 28 --tuning_lr 0.001 --gpu_id 0
# python main.py --times 1 --seed 777 --isBatchRemove 0 --adv_type PGD --dataset ImageNet --layer 28 --tuning_lr 0.001 --gpu_id 0
# python main.py --times 2 --seed 888 --isBatchRemove 0 --adv_type PGD --dataset ImageNet --layer 28 --tuning_lr 0.001 --gpu_id 0


### for ImageNet->Cifar10 FGSM

# python main.py --times 0 --seed 666 --isBatchRemove 0 --adv_type FGSM --dataset ImageNet --layer 28 --tuning_lr 0.001 --gpu_id 0
# python main.py --times 1 --seed 777 --isBatchRemove 0 --adv_type FGSM --dataset ImageNet--layer 28 --tuning_lr 0.001 --gpu_id 0
# python main.py --times 2 --seed 888 --isBatchRemove 0 --adv_type FGSM --dataset ImageNet --layer 28 --tuning_lr 0.001 --gpu_id 0



## pre training retrain model


### for Lacuna-100-->Lacuna-10

#### for BatchRemove-0 PGD
# python Pre_retrain_model.py --times 0 --seed 998 --isBatchRemove 0 --adv_type PGD --dataset Lacuna-100 --layers 28 --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 0
# python Pre_retrain_model.py --times 1 --seed 999 --isBatchRemove 0 --adv_type PGD --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 0
# python Pre_retrain_model.py --times 2 --seed 1000 --isBatchRemove 0 --adv_type PGD --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 0
 
#### for BatchRemove-0 FGSM
# python Pre_retrain_model.py --times 0 --seed 998 --isBatchRemove 0 --adv_type FGSM --dataset Lacuna-100 --layers 28 --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 0
# python Pre_retrain_model.py --times 1 --seed 999 --isBatchRemove 0 --adv_type FGSM --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 0
# python Pre_retrain_model.py --times 2 --seed 1000 --isBatchRemove 0 --adv_type FGSM --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 0
 

### for downsampled ImageNet-->Cifar10

#### for BatchRemove-0 PGD
# python Pre_retrain_model.py --times 0 --seed 666 --isBatchRemove 0 --adv_type PGD --dataset ImageNet --layers 28 --tuning_lr 0.001 --tuning_epochs 10 --gpu_id 0
# python Pre_retrain_model.py --times 1 --seed 777 --isBatchRemove 0 --adv_type PGD --dataset ImageNet --layers 28  --tuning_lr 0.001 --tuning_epochs 10 --gpu_id 0
# python Pre_retrain_model.py --times 2 --seed 888 --isBatchRemove 0 --adv_type PGD --dataset ImageNet --layers 28  --tuning_lr 0.001 --tuning_epochs 10 --gpu_id 0
 
#### for BatchRemove-0 FGSM
# python Pre_retrain_model.py --times 0 --seed 666 --isBatchRemove 0 --adv_type FGSM --dataset ImageNet --layers 28 --tuning_lr 0.001 --tuning_epochs 10 --gpu_id 0
# python Pre_retrain_model.py --times 1 --seed 777 --isBatchRemove 0 --adv_type FGSM --dataset ImageNet --layers 28  --tuning_lr 0.001 --tuning_epochs 10 --gpu_id 0
# python Pre_retrain_model.py --times 2 --seed 888 --isBatchRemove 0 --adv_type FGSM --dataset ImageNet --layers 28  --tuning_lr 0.001 --tuning_epochs 10 --gpu_id 0
 
