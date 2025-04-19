NUM_PROC=2
GPUS=0,1
PORT=25666

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port $PORT train_dg.py \
 --exp_id aig_train_dg \
 --gpus $GPUS \
 --batch_size 32 --max_token_size 4096 \
 --aig_encoder dg
