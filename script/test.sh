#!/bin/bash
batchs=1
GPU=$1
lr=0.0002
loadSize=256
fineSize=256
L1=100
model=madain
G='mask'
checkpoint='../checkpoints/'
datasetmode="shadowgttest"
dataroot='/mnt/work/dataset/ShadowDatasets/ISTD_Dataset/test/'
name='experiment_name'

NAME="${model}_G_${G}_${name}"


CMD="python ../test.py --loadSize ${loadSize} \
    --name ${NAME} \
    --dataroot  ${dataroot}\
    --checkpoints_dir ${checkpoint} \
    --fineSize $fineSize --model $model\
    --batch_size $batchs --keep_ratio --phase test_  --gpu_ids ${GPU} \
    --dataset_mode $datasetmode --epoch best_rmse\
    --netG $G\
    $OTHER
"
echo $CMD
eval $CMD

