pkill -9 ray
pkill -9 python3.8
rm -rf /dev/shm/*
num_workers=(48)
num_env=16
group_name="sc2"
job_name="sc2"
num_frames=200000000
seed=678446
env_name="PongNoFrameskip-v4"
for num_worker in ${num_workers[@]}
do
    exp_name=${env_name}"_"${num_env}"*"${num_worker}
    echo "current experiment ${exp_name}"
    python3.8 main_atari.py --exp_name ${exp_name} \
                            --wandb_group ${group_name} \
                            --wandb_job ${job_name} \
                            --total_frames ${num_frames} \
                            --seed ${seed} \
                            --env_num ${num_env} \
                            --env_name ${env_name} \
                            --num_workers ${num_worker} \
                            --min_return_chunk_num 16 \
                            --batch_size 512 \
                            --num_gpus 1 \
                            --q_size 8 \
                            --num_writers 8 \
                            --push_period 2
    pkill -9 ray
    pkill -9 python3.8
    rm -rf /dev/shm/*
done