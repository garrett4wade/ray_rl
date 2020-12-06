num_workers=16
num_envs=(1)
group_name="env_per_worker"
job_name="speed_test_nobatchqueue"
num_frames=2000000
seed=562789
for num_env in ${num_envs[@]}
do
    exp_name="2M_worker${num_workers}singleenv${num_env}"
    echo "current experiment ${exp_name}"
    python run_async_ppo.py --exp_name ${exp_name} \
                            --wandb_group ${group_name} \
                            --wandb_job ${job_name} \
                            --total_frames ${num_frames} \
                            --seed ${seed} \
                            --env_num ${num_env} \
                            --num_workers ${num_workers} \
                            --q_size 16 \
                            --gpu_id 0
    pkill -9 ray
done