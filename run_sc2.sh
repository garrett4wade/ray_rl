pkill -9 ray
pkill -9 Main_Thread

batch_size=512
num_workers=50
num_env=2
group_name="sc2"
env_names=("8m")
num_frames=200000000
seeds=(256780 975134)
for env_name in ${env_names[@]}
do
    job_name=env_name
    for seed in ${seeds[@]}
    do
        exp_name="rec"${env_name}"_env"${num_env}"worker"${num_workers}"_seed"${seed}
        echo "current experiment ${exp_name}"
        python main_sc2.py --exp_name ${exp_name} \
                                --env_name ${env_name} \
                                --wandb_group ${group_name} \
                                --wandb_job ${job_name} \
                                --batch_size ${batch_size} \
                                --total_frames ${num_frames} \
                                --seed ${seed} \
                                --env_num ${num_env} \
                                --num_workers ${num_workers} \
                                --gpu_id 0 \
                                --min_return_chunk_num 32
        pkill -9 ray
        pkill -9 Main_Thread
        # wait for port release
        sleep 300
    done
done