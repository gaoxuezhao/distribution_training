bash kill.sh

python -u trainer.py \
--ps_hosts=localhost:2222 \
--worker_hosts=localhost:2223,localhost:2224 \
--job_name=ps \
--checkpoint_dir=./ckpt/0 \
--task_index=0>log_pssa_0 2>&1 & 
echo 'ps 0'

python -u trainer.py \
--ps_hosts=localhost:2222 \
--worker_hosts=localhost:2223,localhost:2224 \
--job_name=worker \
--checkpoint_dir=./ckpt/1 \
--task_index=0 >log_worker_0 2>&1 &
echo 'worker 0'

python -u trainer.py \
--ps_hosts=localhost:2222 \
--worker_hosts=localhost:2223,localhost:2224 \
--job_name=worker \
--checkpoint_dir=./ckpt/2 \
--task_index=1 >log_worker_1 2>&1 &
echo 'worker 1'
