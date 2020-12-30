pkill -9 ray
num_workers=(48)
num_env=16
min_return_chunk_num=32
group_name="shmbuf"
job_name="shmbuf"
num_frames=10000000
seed=986258
for num_worker in ${num_workers[@]}
do
    exp_name="4writer_humanoid_env"${num_env}"worker"${num_worker}
    echo "current experiment ${exp_name}"
    python main_mujoco.py --exp_name ${exp_name} \
                            --wandb_group ${group_name} \
                            --wandb_job ${job_name} \
                            --total_frames ${num_frames} \
                            --seed ${seed} \
                            --env_num ${num_env} \
                            --num_workers ${num_worker} \
                            --gpu_id 1 \
                            --min_return_chunk_num ${min_return_chunk_num} \
                            --batch_size 10240
    pkill -9 ray
done