# BSC-attack
Paper code: “Attacking Video Recognition Models with Bullet-Screen Comments” [paper](https://arxiv.org/pdf/2110.15629.pdf).

# Dataset
UCF-101 and HMDB-51 datasets are preprocessing by the methods in [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch).  
"dataset.py" file loads specified datasets.
## Dataset-C3D
Parameters "root_path", "video_path", "annotation_path" need to be customized in "datasets/c3d_dataset/c3d_opt.py".

# Model
C3D and LRCN models are from [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) and [video_adv](https://github.com/yanhui002/video_adv/tree/master/models/inception) respectively. I3D-Slow model is from [GluonCV](https://cv.gluon.ai/model_zoo/action_recognition.html)

## C3D
### C3D-UCF101
Parameter "pretrain_path" is the path of the pretrain model in "video_cls_models/c3d/ucf101_opts.py".    
Download [here](https://drive.google.com/open?id=1DmI6QBrh7xhme0jOL-3nEutJzesHZTqp).
* Generate the parameters file in pickle format
```bash
python ucf101_opts.py
```
* Use the path of the parameters file to specify the line 19 in 'video_cls_models/c3d/c3d.py'.
### C3D-HMDB51
Parameter "pretrain_path" is the path of the pretrain model in "video_cls_models/c3d/hmdb51_opts.py".  
Download [here](https://drive.google.com/open?id=1GWP0bAff6H6cE85J6Dz52in6JGv7QZ_u).
* Generate the parameters file in pickle format
```bash
python hmdb51_opts.py
```
* Use the path of the parameters file to specify the line 15 in 'video_cls_models/c3d/c3d.py'.
