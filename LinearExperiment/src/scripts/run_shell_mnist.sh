!/bin/bash

###################
### MNIST-b successive deletion code
###################

# for model in logistic ridge
for model in ridge
do
    for adv in PGD FGSM
    do 
        for times in {0..4}
        do
        {  
            if [ $model == "logistic" ]; 
            then
                echo "python batch_muter_new.py --dataset binaryMnist --model $model --adv $adv --times $times --deletenum 600 --deletebatch 1 --isbatch False --remove_type 2"
                nohup python -u batch_muter_new.py --dataset binaryMnist --model $model --adv $adv --times $times --deletenum 600 --deletebatch 1 --isbatch False --remove_type 2 >logs/binaryMnist-$model-$adv-$times 2>&1 &
            else 
                echo "python batch_muter_new.py --dataset binaryMnist --model $model --adv $adv --times $times --deletenum 600 --deletebatch 1 --isbatch False --remove_type 2"
                nohup python -u batch_muter_new.py --dataset binaryMnist --model $model --adv $adv --times $times --deletenum 600 --deletebatch 1 --isbatch False --remove_type 2 >logs/binaryMnist-$model-$adv-$times 2>&1 & 
            fi 
        }
        done
        wait
    done
    wait
done 
wait
