\bin\bash

## PGD_0

# echo "python main.py --times 0 --adv_type PGD --isBatchRemove 0 --seed 777"
# python main.py --times 0 --adv_type PGD --isBatchRemove 0 --seed 777

# echo "python main.py --times 1 --adv_type PGD --isBatchRemove 0 --seed 666"
# python main.py --times 1 --adv_type PGD --isBatchRemove 0 --seed 666

# echo "python main.py --times 2 --adv_type PGD --isBatchRemove 0 --seed 888"
# python main.py --times 2 --adv_type PGD --isBatchRemove 0 --seed 888

## PGD_1

# echo "python main.py --times 0 --adv_type PGD --isBatchRemove 1 --seed 777 --gpu_id 2"
# python main.py --times 0 --adv_type PGD --isBatchRemove 1 --seed 777 --gpu_id 2

# echo "python main.py --times 1 --adv_type PGD --isBatchRemove 1 --seed 666 --gpu_id 2"
# python main.py --times 1 --adv_type PGD --isBatchRemove 1 --seed 666 --gpu_id 2

echo "python main.py --times 2 --adv_type PGD --isBatchRemove 1 --seed 888 --gpu_id 2"
python main.py --times 2 --adv_type PGD --isBatchRemove 1 --seed 888 --gpu_id 2
