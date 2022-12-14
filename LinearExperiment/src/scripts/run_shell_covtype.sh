!/bin/bash

###################
### Quick verification, demo code
###################

## for logistic model 
# python batch_muter_new.py --dataset binaryMnist --model logistic --adv PGD --times 0 --deletenum 600 --deletebatch 1  --isbatch False --remove_type 2

## for ridge model
# python batch_muter_new.py --dataset binaryMnist --model ridge --adv PGD --times 0 --deletenum 600 --deletebatch 1  --isbatch False --remove_type 2

###################
### COVTYPE successive deletation code
###################

for model in logistic ridge
do
    for adv in FGSM PGD
    do 
        for times in {0..4}
        do  
        {
            if [ $model == "logistic" ];                                                    
            then
                echo "python batch_muter_new.py --dataset covtype --model $model --adv $adv --times $times --deletenum 25000 --deletebatch 1 --iterneumann 3 --isbatch False --remove_type 2 --batchsize 128"
                nohup python -u batch_muter_new.py --dataset covtype --model $model --adv $adv --times $times --deletenum 25000 --deletebatch 1 --iterneumann 3 --isbatch False --remove_type 2 --batchsize 128 >logs/covtype-$model-$adv-iterneumann3-$times 2>&1 &
            else 
                echo "python batch_muter_new.py --dataset covtype --model $model --adv $adv --times $times --deletenum 25000 --deletebatch 1 --iterneumann 20 --isbatch False --remove_type 2 --batchsize 128"
                nohup python -u batch_muter_new.py --dataset covtype --model $model --adv $adv --times $times --deletenum 25000 --deletebatch 1 --iterneumann 20 --isbatch False --remove_type 2 --batchsize 128 >logs/covtype-$model-$adv-iterneumann20-$times 2>&1 &
            fi
        }
        done
        wait
    done
    wait
done 
wait

