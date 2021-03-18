pkill -9 ray
sleep 2
pkill -9 Main_Thread
sleep 2
pkill -9 python3.8
sleep 2
rm -rf /dev/shm/*

batch_size=512
num_workers=56
num_env=2
group_name="sc2"
env_names=("6h_vs_8z" "3s5z_vs_3s6z")
num_frames=50000000
seeds=(6846875)
for env_name in ${env_names[@]}
do
    job_name=env_name
    for seed in ${seeds[@]}
    do
        exp_name="hbr_othrnn_"${env_name}"_"${num_env}"*"${num_workers}"_seed"${seed}
        echo "current experiment ${exp_name}"
        python3.8 -u main_sc2.py --exp_name ${exp_name} \
                                --env_name ${env_name} \
                                --wandb_group ${group_name} \
                                --wandb_job ${job_name} \
                                --batch_size ${batch_size} \
                                --total_frames ${num_frames} \
                                --seed ${seed} \
                                --env_num ${num_env} \
                                --num_workers ${num_workers} \
                                --min_return_chunk_num 32 \
                                --push_period 2 \
                                --num_writers 4 \
                                --num_gpus 1 \
                                --q_size 6 \
                                --num_returns 1
        pkill -9 ray
        sleep 2
        pkill -9 Main_Thread
        sleep 2
        pkill -9 python3.8
        sleep 2
        rm -rf /dev/shm/*
        # wait for port release
        sleep 30
    done
done