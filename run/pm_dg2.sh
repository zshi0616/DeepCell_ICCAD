NUM_PROC=2
GPUS=0,1
PORT=25672

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port $PORT train_pm.py \
 --exp_id pm_train_dg \
 --gpus $GPUS \
 --batch_size 8 --max_token_size 4096 \
 --pm_aggr dg2
