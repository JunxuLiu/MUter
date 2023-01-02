!\bin\bash

## for FGSM Batch

echo "python main.py --adv_type FGSM --isBatchRemove 1 --gpu_id 3 --seed 666 --times 1"
python main.py --adv_type FGSM --isBatchRemove 1 --gpu_id 3 --seed 666 --times 1

# echo "python main.py --adv_type FGSM --isBatchRemove 1 --gpu_id 3 --seed 888 --times 2"
# python main.py --adv_type FGSM --isBatchRemove 1 --gpu_id 3 --seed 888 --times 2

# ## for PGD Schur, before running, see if the load mode right!
# echo "python main.py --adv_type PGD --isBatchRemove 0 --gpu_id 3 --seed 666 --times 1"
# python main.py --adv_type PGD --isBatchRemove 0 --gpu_id 3 --seed 666 --times 1

# echo "python main.py --adv_type PGD --isBatchRemove 0 --gpu_id 3 --seed 888 --times 2"
# python main.py --adv_type PGD --isBatchRemove 0 --gpu_id 3 --seed 888 --times 2


### for PGD Batch

# echo "python main.py --adv_type PGD --isBatchRemove 1 --gpu_id 1 --seed 666 --times 1"
# python main.py --adv_type PGD --isBatchRemove 1 --gpu_id 1 --seed 666 --times 1

# echo "python main.py --adv_type PGD --isBatchRemove 1 --gpu_id 1 --seed 888 --times 2"
# python main.py --adv_type PGD --isBatchRemove 1 --gpu_id 1 --seed 888 --times 2