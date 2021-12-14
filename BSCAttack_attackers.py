import os
from easydict import EasyDict as edict
from BSCAttack_agents import agent
from BSCAttack_config import BSCA_cfg


class BSCA():
    def __init__(self, dir_title):
        # attack dirs
        self.attack_dir = os.path.join(BSCA_cfg.attack_dir, dir_title)

    def attack(self, text, font, model, input_tensor, label_tensor, input_name):
        # set up attack-dirs
        attack_dir = os.path.join(self.attack_dir, input_name)
        if not os.path.exists(attack_dir):
            os.makedirs(attack_dir)
        # set records
        rcd = edict()
        rcd.masks = []
        rcd.RGB_paintings = []
        rcd.combos = []
        rcd.areas = []
        rcd.salients = []
        rcd.non_target_success = []
        rcd.target_success = []
        rcd.queries = []
        rcd.time_used = []
        # attack
        mask_input_tensor, area, success, queries, time_used = agent.attack(
            model=model,
            input_tensor=input_tensor,
            target_tensor=label_tensor,
            text=text,
            font=font,
            sigma=BSCA_cfg.sigma,
            lr=BSCA_cfg.lr,
            baseline_subtraction=BSCA_cfg.baseline_sub,
            color=BSCA_cfg.color,
            num_occlu=BSCA_cfg.n_occlu,
            rl_batch=BSCA_cfg.rl_batch,
            steps=BSCA_cfg.steps
        )
        # update records
        rcd.masks.append(mask_input_tensor)
        if success:
            rcd.areas.append(((mask_input_tensor - input_tensor) != 0.).sum())
        else:
            rcd.areas.append(area)
        rcd.salients.append(area)
        rcd.success.append(success)
        rcd.queries.append(queries)
        rcd.time_used.append(time_used)
        # print records
        print('success: {} | queries: {:.4f} | occluded area: {:.4f} | occluded salient: {:.4f}'.format(rcd.success[0], rcd.queries[0], rcd.areas[0].item(), rcd.salients[0].item()))
        return mask_input_tensor, rcd, attack_dir
