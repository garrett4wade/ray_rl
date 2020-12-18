pkill -9 ray
pkill -9 Main_Thread

batch_size=32
num_workers=1
num_env=1
env_names=("8m")
num_frames=200000000
seeds=(256780)
for env_name in ${env_names[@]}
do
    for seed in ${seeds[@]}
    do
        exp_name="rec"${env_name}"_env"${num_env}"worker"${num_workers}"_seed"${seed}
        echo "current experiment ${exp_name}"
        python run_async_ppo.py --exp_name ${exp_name} \
                                --env_name ${env_name} \
                                --batch_size ${batch_size} \
                                --total_frames ${num_frames} \
                                --seed ${seed} \
                                --env_num ${num_env} \
                                --num_workers ${num_workers} \
                                --gpu_id 0 \
                                --min_return_chunk_num 32 \
                                --no_summary
        pkill -9 ray
        pkill -9 Main_Thread
    done
done