#!/bin/bash
batchs=5
DISPLAY_PORT=${2:-7000}
GPU=$1
lr=0.0002
loadSize=256
fineSize=256
L1=100
model=madain
G='mask'
checkpoint='checkpoints/'
datasetmode="shadowgt"
dataroot='/mnt/work/dataset/ShadowDatasets/ISTD_Dataset/train'
name='test'

NAME="${model}_G_${G}_${name}"

OTHER="--save_epoch_freq 100 --niter 50 --niter_decay 300 --test_epoch_freq 15"

CMD="python ../train.py --loadSize ${loadSize} \
    --randomSize
    --name ${NAME} \
    --dataroot  ${dataroot}\
    --checkpoints_dir ${checkpoint} \
    --fineSize $fineSize --model $model\
    --batch_size $batchs --display_port ${DISPLAY_PORT} --display_server http://localhost\
    --randomSize --keep_ratio --phase train_  --gpu_ids ${GPU} --lr ${lr} \
    --lambda_L1 ${L1} 
    --dataset_mode $datasetmode \
    --netG $G\
    $OTHER
"
echo $CMD
eval $CMD

