# BSC-attack
Paper code: “[Attacking Video Recognition Models with Bullet-Screen Comments](https://arxiv.org/pdf/2110.15629.pdf)”.

# Dataset
UCF-101 and HMDB-51 datasets are preprocessing by the methods in [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch).  
"dataset.py" file loads specified datasets.
## Dataset-C3D
Parameters "root_path", "video_path", "annotation_path" need to be customized in "opts/c3d_opt.py".

# Model
C3D and LRCN models are from [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) and [video_adv](https://github.com/yanhui002/video_adv/tree/master/models/inception) respectively. I3D-Slow model is from [GluonCV](https://cv.gluon.ai/model_zoo/action_recognition.html)

## C3D
### C3D-UCF101
Parameter "pretrain_path" is the path of the pretrain model in "opts/c3d_opt.py/ucf101_parse_opts".  
Download [here](https://drive.google.com/open?id=1DmI6QBrh7xhme0jOL-3nEutJzesHZTqp).
* Use the path of the parameters file to specify the line 21 in 'opts/c3d_opt.py'.
### C3D-HMDB51
Parameter "pretrain_path" is the path of the pretrain model in "opts/c3d_opt.py/hmdb51_parse_opts".  
Download [here](https://drive.google.com/open?id=1GWP0bAff6H6cE85J6Dz52in6JGv7QZ_u).
* Use the path of the parameters file to specify the line 59 in 'opts/c3d_opt.py'.
### C3D-Kinetics400
Parameter "pretrain_path" is the path of the pretrain model in "opts/c3d_opt.py/kinetics_parse_opts".  
Download [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M).
* Use the path of the parameters file to specify the line 97 in 'opts/c3d_opt.py'.

# Citation
If you use the code or find this project helpful, please consider citing our paper.

```
@article{chen2021attacking,
  title={Attacking Video Recognition Models with Bullet-Screen Comments},
  author={Chen, Kai and Wei, Zhipeng and Chen, Jingjing and Wu, Zuxuan and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2110.15629},
  year={2021}
}
```
