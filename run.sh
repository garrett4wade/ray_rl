pkill -9 ray
num_workers=48
cpu_per_worker=1
num_envs=(16)
num_returns=(1)
group_name="sample_speed"
job_name="sample_speed"
num_frames=10000000
seed=562789
for num_env in ${num_envs[@]}
do
    for num_return in ${num_returns[@]}
    do
        exp_name="env"${num_env}"worker"${num_workers}"return"${num_return}
        echo "current experiment ${exp_name}"
        python run_async_ppo.py --exp_name ${exp_name} \
                                --wandb_group ${group_name} \
                                --wandb_job ${job_name} \
                                --total_frames ${num_frames} \
                                --seed ${seed} \
                                --env_num ${num_env} \
                                --num_workers ${num_workers} \
                                --cpu_per_worker ${cpu_per_worker} \
                                --q_size 16 \
                                --gpu_id 1 \
                                --num_returns ${num_return}
        pkill -9 ray
    done
done