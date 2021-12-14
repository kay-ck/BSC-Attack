from datasets.c3d.kinetics import Kinetics
from datasets.c3d.ucf101 import UCF101
from datasets.c3d.hmdb51 import HMDB51


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']
    assert opt.test_subset in ['val', 'test']
    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'kinetics':
        test_data = Kinetics(opt.video_path, opt.annotation_path, subset, 0, spatial_transform, temporal_transform, target_transform, sample_duration=opt.sample_duration)
    elif opt.dataset == 'activitynet':
        test_data = ActivityNet(opt.video_path, opt.annotation_path, subset, True, 0, spatial_transform, temporal_transform, target_transform, sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        test_data = UCF101(opt.video_path, opt.annotation_path, subset, 0, spatial_transform, temporal_transform, target_transform, sample_duration=opt.sample_duration)
    elif opt.dataset == 'hmdb51':
        test_data = HMDB51(opt.video_path, opt.annotation_path, subset, 0, spatial_transform, temporal_transform, target_transform, sample_duration=opt.sample_duration)
    return test_data
