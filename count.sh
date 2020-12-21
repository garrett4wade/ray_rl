num_idle=$(pgrep ray::IDLE | wc -l)
num_worker_get=$(pgrep ray::Worker.get | wc -l)
num_worker=$(pgrep ray::Worker | wc -l)
echo "-----------------------------------------"
echo "number of idle processes is: "${num_idle}
echo "number of worker.get processes is: "${num_worker_get}
echo "number of worker processes is: "$((${num_worker}-${num_worker_get}))
echo "sum of worker, idle, worker.get is: "$((${num_worker}+${num_idle}))
echo "total number of ray processes is: "$(pgrep ray | wc -l)
echo "-----------------------------------------"
