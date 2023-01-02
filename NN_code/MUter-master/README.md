# MUter  

#### Dependencies
torch, torchvision, torchattacks, functorch

#### Pretrain model
see sh_code.sh for more detial.

`
python Pre_retrain_model.py --times 0 --seed 998 --isBatchRemove 0 --adv_type PGD --dataset Cifar100 --layers 34 --tuning_lr 0.1
`

#### unlearning process
see sh_code for more detail.

`
python main.py --times 0 --seed 998 --isBatchRemove 1 --adv_type PGD --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 3
`


#### plot see util.py

