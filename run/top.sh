NUM_PROC=4
GPUS=0,1,2,3
MASK=0.01
HOP=4
PORT=25889

PM=dg2
AIG=dg3
REFINE=aig

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port $PORT top_train.py \
 --exp_id top_${REFINE}_${PM}_${AIG} \
 --batch_size 4 --num_epochs 60 --max_token_size 4096 \
 --mask_ratio $MASK \
 --k_hop $HOP \
 --linformer \
 --gpus $GPUS \
 --refine $REFINE \
 --pm_aggr $PM --aig_encoder $AIG 

