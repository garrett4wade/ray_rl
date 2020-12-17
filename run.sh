pkill -9 ray
pkill -9 Main_Thread

batch_size=256
num_workers=40
num_env=2
group_name="sc2"
env_name="3m"
job_name="3m"
num_frames=100000000
seeds=(256780)
for seed in ${seeds[@]}
do
    exp_name="recpopart3m_env"${num_env}"worker"${num_workers}"_seed"${seed}
    echo "current experiment ${exp_name}"
    python run_async_ppo.py --exp_name ${exp_name} \
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
done