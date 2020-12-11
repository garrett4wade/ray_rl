pkill -9 ray
pkill -9 Main_Thread

num_workers=(50)
num_env=2
group_name="sample_speed"
job_name="sample_speed"
num_frames=100000000
seed=562789
for num_worker in ${num_workers[@]}
do
    exp_name="100M3m_env"${num_env}"worker"${num_worker}
    echo "current experiment ${exp_name}"
    python run_async_ppo.py --exp_name ${exp_name} \
                            --wandb_group ${group_name} \
                            --wandb_job ${job_name} \
                            --total_frames ${num_frames} \
                            --seed ${seed} \
                            --env_num ${num_env} \
                            --num_workers ${num_worker} \
                            --gpu_id 1 \
                            --min_return_chunk_num 32 \
                            --q_size 16
    pkill -9 ray
    pkill -9 Main_Thread
done