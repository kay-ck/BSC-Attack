def get_mean(norm_value=255, dataset='ucf101'):
    assert dataset in ['hmdb51', 'kinetics', 'ucf101']
    if dataset == 'hmdb51':
        return [95.4070 / norm_value, 93.4680 / norm_value, 82.1443 / norm_value]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [110.63666788 / norm_value, 103.16065604 / norm_value, 96.29023126 / norm_value]
    elif dataset == 'ucf101':
        return [101.2198 / norm_value, 97.5751 / norm_value, 89.5303 / norm_value]


def get_std(norm_value=255, dataset='ucf101'):
    assert dataset in ['hmdb51', 'kinetics', 'ucf101']
    if dataset == 'hmdb51':
        return [51.674248 / norm_value, 50.311924 / norm_value, 49.48427 / norm_value]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [38.7568578 / norm_value, 37.88248729 / norm_value, 40.02898126 / norm_value]
    elif dataset == 'ucf101':
        return [62.08429 / norm_value, 60.398968 / norm_value, 59.187363 / norm_value]
