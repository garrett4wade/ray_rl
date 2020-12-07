# num_workers=32
cpu_per_worker=1
num_envs=(4 16)
group_name="memory"
job_name="circularbuffer_list"
num_frames=4000000
seed=562789
for num_env in ${num_envs[@]}
do
    num_workers=$((32*8/${num_env}))
    exp_name="10M_circularbuffer_list_256worker"${num_workers}
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
                            --gpu_id 0
    pkill -9 ray
done