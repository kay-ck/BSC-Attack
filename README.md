# Bullet-Screen Comment (BSC) Attack
Paper code: “[Attacking Video Recognition Models with Bullet-Screen Comments](https://arxiv.org/abs/2110.15629)”.

# Dataset
UCF-101 and HMDB-51 datasets are preprocessing by the methods in [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch).  
"dataset.py" file loads specified datasets.
## Dataset-C3D
Parameters "root_path", "video_path", "annotation_path" need to be customized in "opts/c3d_opt.py".

# Model
C3D and LRCN models are from [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) and [video_adv](https://github.com/yanhui002/video_adv/tree/master/models/inception) respectively. I3D-Slow model is from [GluonCV](https://cv.gluon.ai/model_zoo/action_recognition.html)

## C3D
### C3D-UCF101
Parameter "pretrain_path" is the path of the pretrain model in "opts/c3d_opt.py/".  
Download [here](https://drive.google.com/open?id=1DmI6QBrh7xhme0jOL-3nEutJzesHZTqp).

### C3D-HMDB51
Download [here](https://drive.google.com/open?id=1GWP0bAff6H6cE85J6Dz52in6JGv7QZ_u).

### C3D-Kinetics400
Download [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M).
* Use the path of the parameters file to specify the line 21 in 'opts/c3d_opt.py'.

# Attacks
```bash
python main.py --root_path <str> --video_path <str> --annotation_path <str> --dataset <kinetics/ucf101/hmdb51> --model <c3d/lrcn/i3d> --n_classes <400/101/51> --mean_dataset <kinetics/ucf101/hmdb51> --pretrain_path <str>
```
Our implementation inludes four black-box patch-based attack: BSC Attack based on Reinforcement Learning (RL), BSC Attack based on Basin Hopping (BH), BSC Attack based on random selection and Patch Attack with white square patch in our [paper](https://arxiv.org/abs/2110.15629). Patch Attack was originally proposed in [paper](https://arxiv.org/abs/2004.05682); BH is the baseline in [paper](https://arxiv.org/abs/2008.01919), which used to generate the adversarial watermark. Their implementation are in [PatchAttack](https://github.com/Chenglin-Yang/PatchAttack) and [Adv-watermark](https://github.com/jiaxiaojunQAQ/Adv-watermark) respectively. Besides, we use an image captioning model proposed in [paper](https://arxiv.org/abs/1502.03044) to generate different BSCs for each video, the code is in [Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).

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
