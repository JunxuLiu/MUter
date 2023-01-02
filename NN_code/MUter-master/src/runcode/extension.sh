!\bin\bash

# # for SISA shard=5
# for times in 11 12 13
# do
#     for batch in 500 1000 1500 2000 2500 3000
#     do
#         echo "python main.py --remove_batch $batch --times $times --seed $times --isBatchRemove 2"
#         python main.py --remove_batch $batch --times $times --seed $times --isBatchRemove 2
#     done
# done

# for SISA shard=10
# for times in 11 
# do
#     for batch in 500 1000 1500 2000 2500 3000
#     do
#         echo "python main.py --remove_batch $batch --times $times --seed $times --isBatchRemove 2 --shards 10"
#         python main.py --remove_batch $batch --times $times --seed $times --isBatchRemove 2 --shard 10
#     done
# done

# for SISA shard=20
# for times in 11
# do
#     for batch in 1500 2000 2500 3000
#     do
#         echo "python main.py --remove_batch $batch --times $times --seed $times --isBatchRemove 2 --shards 20"
#         python main.py --remove_batch $batch --times $times --seed $times --isBatchRemove 2 --shard 20
#     done
# done

## for SISA-DK shard=5
# for times in 11 
# do
#     for batch in 500 1000 1500 2000 2500 3000
#     do
#         echo "python main.py --remove_batch $batch --times $times --seed $times --isBatchRemove 2 --isDK 1"
#         python main.py --remove_batch $batch --times $times --seed $times --isBatchRemove 2 --isDK 1
#     done
# done

# # for SISA-DK shard=10
# for times in 11 
# do
#     for batch in 500 1000 1500 2000 2500 3000
#     do
#         echo "python main.py --remove_batch $batch --times $times --seed $times --isBatchRemove 2 --isDK 1 --shards 10"
#         python main.py --remove_batch $batch --times $times --seed $times --isBatchRemove 2 --isDK 1 --shards 10
#     done
# done

# # for SISA-DK shard=20
# for times in 11 
# do
#     for batch in 500 1000 1500 2000 2500 3000
#     do
#         echo "python main.py --remove_batch $batch --times $times --seed $times --isBatchRemove 2 --isDK 1 --shards 20"
#         python main.py --remove_batch $batch --times $times --seed $times --isBatchRemove 2 --isDK 1 --shards 20
#     done
# done


## for MUter and FMUter

for times in 11 12 13
do
    for batch in 500 1000 1500 2000 2500 3000
    do
        echo "python main.py --remove_batch $batch --times $times --seed $times --isBatchRemove 2"
        python main.py --remove_batch $batch --times $times --seed $times --isBatchRemove 2
    done
done