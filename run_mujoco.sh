pkill -9 ray
pkill -9 python3.8
rm -rf /dev/shm/*
num_workers=(48)
num_env=16
group_name="postprocessor"
job_name="postprocessor"
num_frames=20000000
seed=89276
for num_worker in ${num_workers[@]}
do
    exp_name="baseline_humanoid_"${num_env}"*"${num_worker}
    echo "current experiment ${exp_name}"
    python3.8 main_mujoco.py --exp_name ${exp_name} \
                            --wandb_group ${group_name} \
                            --wandb_job ${job_name} \
                            --total_frames ${num_frames} \
                            --seed ${seed} \
                            --env_num ${num_env} \
                            --num_workers ${num_worker} \
                            --min_return_chunk_num 64 \
                            --batch_size 2560 \
                            --num_writers 1 \
                            --num_gpus 4
    pkill -9 ray
    pkill -9 python3.8
    rm -rf /dev/shm/*
done