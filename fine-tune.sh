DATASET=$1
NGPU=$2

source ./args.sh $DATSET fives_L512_patches $NGPU 1

NCCL_P2P_LEVEL=NVL mpiexec --use-hwthread-cpus --oversubscribe -n $NGPU python fine-tune.py \
    --workdir $WORK_DIR --data_dir $DATA_DIR --image_size $IMAGE_SIZE \
    --microbatch $MICRO_BATCH --global_batch_size $BATCH_SIZE \
    --lr $LR --ema_rate $EMA_RATE --weight_decay $WEIGHT_DECAY --lr_anneal_steps $LR_ANNEAL_STEPS \
    --log_interval $LOG_INTERVAL --save_interval $SAVE_INTERVAL --total_training_steps $TOTAL_TRAINING_STEPS \
    --augment $AUG --num_workers $NGPU --fp16 $USE_16FP