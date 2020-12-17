pkill -9 ray
pkill -9 Main_Thread

num_workers=1
num_env=1
group_name="sc2"
env_name="3m"
job_name="3m"
num_frames=100000000
seeds=(562789)
for seed in ${seeds[@]}
do
    exp_name="rec_env"${num_env}"worker"${num_workers}"_seed"${seed}
    echo "current experiment ${exp_name}"
    python run_async_ppo.py --exp_name ${exp_name} \
                            --env_name ${env_name} \
                            --wandb_group ${group_name} \
                            --wandb_job ${job_name} \
                            --batch_size 32 \
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