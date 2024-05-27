# User launch paramters
DATASET=$1
NGPU=$2
# Default values
WORK_DIR=""
DATASET_DIR=""
IMAGE_SIZE=64
NUM_IMG_CH=3
MICRO_BATCH=2
BATCH_SIZE=8
LR=1e-4
EMA_RATE=0.9999
WEIGHT_DECAY=0.0
LR_ANNEAL_STEPS=0
USE_16FP=False
LOG_INTERVAL=500
SAMPLE_INTERVAL=10000
SAVE_INTERVAL=10000
TOTAL_TRAINING_STEPS=200000
AUG=0


# Arguments
if [[ $DATASET == "NAME" ]]; then
    DATA_DIR=YOUR_DATASET_PATH
    IMAGE_SIZE=64
    SAVE_INTERVAL=100000
elif [[ $DATASET == "fives_L512_patches" ]]; then
    WORK_DIR=/u6/sszabado/checkpoints/temp/
    DATA_DIR=/share/yaoliang/datasets/fives_L512_patches/train/images/
    IMAGE_SIZE=512
    USE_16FP=False
else
    echo "Not supported"
    exit 1    
fi


if [[ $IMAGE_SIZE == 512 ]]; then
    MICRO_BATCH=4
    BATCH_SIZE=8
elif  [[ $IMAGE_SIZE == 256 ]]; then
    BATCH_SIZE=16
elif  [[ $IMAGE_SIZE == 128 ]]; then
    BATCH_SIZE=14
elif  [[ $IMAGE_SIZE == 64 ]]; then
    BATCH_SIZE=30
else
    echo "Not supported"
    exit 1
fi