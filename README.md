#  Progressiveiy Inpainting Image Based on a Forked-Then-Fused Decoder Network
##  Architecutre
![Network](https://github.com/yabg-shuai666/Inpainting/blob/main/Results/Network.png)
## Place2 Results
![Place2](https://github.com/yabg-shuai666/Inpainting/blob/main/Results/Place2.png)
## Paris Street-View Results
![Paris Street-View](https://github.com/yabg-shuai666/Inpainting/blob/main/Results/ParisStreetView.png)
## CelebA Results
![CelebA](https://github.com/yabg-shuai666/Inpainting/blob/main/Results/CelebA.png)

## Prerequisites
- Linux or Windows.
- Python Python 3.
- CPU or NVIDIA GPU + CUDA CuDNN.
- Tested on pytorch >= **1.2**

## Datasets
We use [Places2](http://places2.csail.mit.edu/), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [Paris Street-View](https://github.com/pathak22/context-encoder) datasets. To train a model on the full dataset, download datasets from official websites.

Our model is trained on the irregular mask dataset provided by [Liu et al](https://arxiv.org/abs/1804.07723). You can download publically available Irregular Mask Dataset from their [website](http://masc.cs.gmu.edu/wiki/partialconv).

## Acknowledgments
We benefit a lot from [Image Inpainting for Irregular Holes Using Partial Convolutions](https://github.com/NVIDIA/partialconv.git).     
We benefit a lot from [Shift-Net: Image Inpainting via Deep Feature Rearrangement](https://github.com/Zhaoyi-Yan/Shift-Net).   
We benefit a lot from [MUSICAL: Multi-Scale Image Contextual Attention Learning for Inpainting](https://github.com/wangning-001/MUSICAL.git).   
We benefit a lot from [Semantic Image Inpainting with Progressive Generative Networks](https://github.com/crashmoon/Progressive-Generative-Networks.git).   
We benefit a lot from [Generative Image Inpainting with Contextual Attention](https://github.com/JiahuiYu/generative_inpainting).    
We benefit a lot from [Coherent Semantic Attention for Image Inpaintin](https://github.com/KumapowerLIU/CSA-inpainting).    
We benefit a lot from [Region Normalization for Image Inpainting](https://github.com/geekyutao/RN.git).    
Finally, I sincerely my teachers Rong Hang and Fang Han for their great help and encouragement.
