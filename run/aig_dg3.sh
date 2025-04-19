NUM_PROC=4
GPUS=0,1,2,3
PORT=25667

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port $PORT train_aig.py \
 --exp_id aig_train_dg3 \
 --gpus $GPUS \
 --batch_size 4 --max_token_size 4096 \
 --aig_encoder dg3 \
 --resume
