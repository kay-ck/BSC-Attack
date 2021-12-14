import argparse
import json
import os
import torch
import BSCAttack_c3d
from dataset import get_test_set
from mean import get_mean, get_std
from model import generate_model_c3d
from opts.c3d_opts import parse_opts
from spatial_transforms import spatial_Compose, Normalize, Scale, CornerCrop, ToTensor
from target_transforms import target_Compose, ClassLabel, VideoID
from temporal_transforms import LoopPadding

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.arch = '{}-{}'.format(opt.model_type, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value, dataset=opt.mean_dataset)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)
    torch.manual_seed(opt.manual_seed)
    if opt.model == 'c3d':
        model = generate_model_c3d(opt)
        model.eval()
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    if opt.model == 'c3d':
        spatial_transform = spatial_Compose([Scale(int(opt.sample_size / opt.scale_in_test)), CornerCrop(opt.sample_size, opt.crop_position_in_test), ToTensor(opt.norm_value), norm_method])
    temporal_transform = LoopPadding(opt.sample_duration)
    target_transform = target_Compose([VideoID(), ClassLabel()])
    test_data = get_test_set(opt, spatial_transform, temporal_transform, target_transform)
    BSCAttack_c3d.process(test_data, model, opt, test_data.class_names)
