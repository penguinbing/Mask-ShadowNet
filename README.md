# Mask-ShadowNet: Toward Shadow Removal via Masked Adaptive Instance Normalization
This is the Pytorch implementation of "**Mask-ShadowNet: Toward Shadow Removal via Masked Adaptive Instance Normalization**", ready to be published soon.  

> Note: The resolution of test shadow image should be 256*256 for our pre-trained model.  



Please first set the dataset path then start training or testing.
* Training
```bash
cd script
bash train.sh gpu_id display_port
```
where ***gpu_id*** is the ID of gpu, and ***display_port*** is the port number of visdom server.  

* Testing
```bash
cd script
bash test.sh gpu_id
```

## Prereuisites  
- Linux or macOS
- python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- PyTorch 1.2+


 


