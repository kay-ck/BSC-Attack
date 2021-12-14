import os
from easydict import EasyDict as edict

# BSCAttack config
BSCA_cfg = edict()


# config BSCAttack
def configure_BSCA(n_occlu=4, rl_batch=500, steps=50, BSCA_color=False):
    # Attack's shared params
    BSCA_cfg.n_occlu = n_occlu  # num of BSCs can put on (default: 4)
    BSCA_cfg.lr = 0.03  # learning rate for RL agent (default: 0.03)
    BSCA_cfg.rl_batch = rl_batch  # batch number when optimizing a RL agent (default: 500)
    BSCA_cfg.steps = steps  # steps to optimize each RL agent (default: 50)
    BSCA_cfg.sigma = 1000  # sigam to control the IoU reward (default: 1000.)
    BSCA_cfg.sigma_sched = []  # sigma schedule for the multiple occlusions (default: n-occlu * sigma)
    if BSCA_cfg.sigma_sched == []:
        BSCA_cfg.sigma_sched = [BSCA_cfg.sigma] * BSCA_cfg.n_occlu
    BSCA_cfg.color = BSCA_color  # flag to use RGB
    BSCA_cfg.baseline_sub = True  # use baseline subtraction mode
    # attack dirs
    attack_dir = os.path.join(
        'n-occlu_{}_color_{}_lr_{}_rl-batch_{}_steps_{}'.format(BSCA_cfg.n_occlu, BSCA_cfg.color, BSCA_cfg.lr, BSCA_cfg.rl_batch, BSCA_cfg.steps),
        'sigma-sched_' + '-'.join([str(item) for item in BSCA_cfg.sigma_sched])
    )
    BSCA_cfg.attack_dir = attack_dir
