pkill -9 ray
sleep 2
pkill -9 Main_Thread
sleep 2
pkill -9 python3.8
sleep 2
# rm -rf /dev/shm/*

batch_size=512
num_workers=54
num_env=2
group_name="sc2"
env_names=("5m_vs_6m")
num_frames=200000000
seeds=(58026)
# seeds=(189486847)
for env_name in ${env_names[@]}
do
    job_name=env_name
    for seed in ${seeds[@]}
    do
        exp_name="dpck_"${env_name}"_"${num_env}"*"${num_workers}"_seed"${seed}
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
                                --num_writers 8 \
                                --num_gpus 1 \
                                --q_size 6
        pkill -9 ray
        pkill -9 Main_Thread
        pkill -9 python3.8
        # rm -rf /dev/shm/*
        # wait for port release
        sleep 300
    done
done