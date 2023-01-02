!\bin\bash

# echo "python Pre_retrain_model.py --times 1 --gpu_id 2 --adv_type FGSM --isBatchRemove 0 --seed 666"
# python Pre_retrain_model.py --times 1 --gpu_id 2 --adv_type FGSM --isBatchRemove 0 --seed 666

# echo "python Pre_retrain_model.py --times 1 --gpu_id 2 --adv_type FGSM --isBatchRemove 1 --seed 666"
# python Pre_retrain_model.py --times 1 --gpu_id 2 --adv_type FGSM --isBatchRemove 1 --seed 666

# echo "python Pre_retrain_model.py --times 1 --gpu_id 3 --adv_type PGD --isBatchRemove 0 --seed 666"
# python Pre_retrain_model.py --times 1 --gpu_id 3 --adv_type PGD --isBatchRemove 0 --seed 666

# echo "python Pre_retrain_model.py --times 1 --gpu_id 1 --adv_type PGD --isBatchRemove 1 --seed 666"
# python Pre_retrain_model.py --times 1 --gpu_id 1 --adv_type PGD --isBatchRemove 1 --seed 666

###

# echo "python Pre_retrain_model.py --times 2 --gpu_id 2 --adv_type FGSM --isBatchRemove 0 --seed 888"
# python Pre_retrain_model.py --times 2 --gpu_id 2 --adv_type FGSM --isBatchRemove 0 --seed 888

# echo "python Pre_retrain_model.py --times 2 --gpu_id 2 --adv_type FGSM --isBatchRemove 1 --seed 888"
# python Pre_retrain_model.py --times 2 --gpu_id 2 --adv_type FGSM --isBatchRemove 1 --seed 888

# echo "python Pre_retrain_model.py --times 2 --gpu_id 3 --adv_type PGD --isBatchRemove 0 --seed 888"
# python Pre_retrain_model.py --times 2 --gpu_id 3 --adv_type PGD --isBatchRemove 0 --seed 888

echo "python Pre_retrain_model.py --times 2 --gpu_id 1 --adv_type PGD --isBatchRemove 1 --seed 888"
python Pre_retrain_model.py --times 2 --gpu_id 1 --adv_type PGD --isBatchRemove 1 --seed 888