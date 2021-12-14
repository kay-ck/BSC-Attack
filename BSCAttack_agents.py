import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.data import TensorDataset, DataLoader
import BSCAttack_c3d

# global variables
eps = np.finfo(np.float32).eps.item()


class robot():
    class p_pi(nn.Module):
        """
        policy network
        """
        def __init__(self, space, embedding_size=30, stable=True):
            super().__init__()
            self.embedding_size = embedding_size
            embedding_space = [space[-1]] + space[:-1]
            # create embedding space
            self.embedding_list = nn.ModuleList([nn.Embedding(embedding_space[i], self.embedding_size) for i in range(len(embedding_space))])
            if stable:
                self._stable_first_embedding()
            # create linear heads
            self.lstm = nn.LSTM(self.embedding_size, self.embedding_size, batch_first=True)  # (batch, seq, features)
            self.linear_list = nn.ModuleList([nn.Linear(self.embedding_size, space[i]) for i in range(len(space))])
            # set necessary parameters
            self.stage = 0
            self.hidden = None

        def forward(self, x):
            x = self.embedding_list[self.stage](x)
            # extract feature of current state
            self.lstm.flatten_parameters()
            x, self.hidden = self.lstm(x, self.hidden)  # hidden: hidden state plus cell state
            # get action prob given the current state
            prob = self.linear_list[self.stage](x.view(x.size(0), -1))
            return prob

        def increment_stage(self):
            self.stage += 1

        def _stable_first_embedding(self):
            target = self.embedding_list[0]
            for param in target.parameters():
                param.requires_grad = False

        def reset(self):
            """
            reset stage to 0
            clear hidden state
            """
            self.stage = 0
            self.hidden = None

    def __init__(self, space, rl_batch, gamma, lr, stable=True):
        # policy network
        self.mind = self.p_pi(space, stable=stable)
        # reward setting
        self.gamma = gamma  # back prop rewards
        # optimizer
        self.optimizer = optim.Adam(self.mind.parameters(), lr=lr)
        # useful parameters
        self.combo_size = len(space)
        self.rl_batch = rl_batch

    def select_action(self, state):
        """
        generate one parameter
        input:
        state: torch.longtensor with size (bs, 1), the sampled action at the last step
        return:
        action: torch.longtensor with size (bs, 1)
        log_p_action: torch.floattensor with size (bs, 1)
        value: [optional] torch.floattensor with size (bs, 1)
        """
        p_a = F.softmax(self.mind(state), dim=1)
        # select action with prob
        dist = Categorical(probs=p_a)
        action = dist.sample()
        log_p_action = dist.log_prob(action)
        return action.unsqueeze(-1), log_p_action.unsqueeze(-1)

    def select_combo(self):
        """
        generate the whole sequence of parameters
        return:
        combo: torch.longtensor with size (bs, space.size(0):
               (PREVIOUS STATEMENT) num_occlu * 4 or 7 if color==True)
        log_p_combo: torch.floattensor with size (bs, space.size(0))
        rewards_critic: torch.floatensor with size (bs, space.size(0))
        """
        state = torch.zeros((self.rl_batch, 1)).long().cuda()
        combo = []
        log_p_combo = []
        for _ in range(self.combo_size):
            action, log_p_action = self.select_action(state)
            combo.append(action)
            log_p_combo.append(log_p_action)
            state = action
            self.mind.module.increment_stage()
        combo = torch.cat(combo, dim=1)
        log_p_combo = torch.cat(log_p_combo, dim=1)
        return combo, log_p_combo


class agent(robot):
    def __init__(self, model, clip_tensor, target_tensor, text, font, num_occlu, color, sigma, shrink=1):
        """
        the __init__ function needs to create action space because this relates with
        the __init__ of the policy network
        """
        # build environment
        self.model = model
        self.clip_tensor = clip_tensor
        self.target_tensor = target_tensor
        self.text = text
        self.font = font
        # build action space
        self.num_occlu = num_occlu
        self.color = color
        self.space = self.create_searching_space(text, font, num_occlu, color, H=clip_tensor.size(-2), W=clip_tensor.size(-1))
        self.shrink = shrink
        # specific reward param
        self.sigma = sigma

    def build_robot(self, rl_batch, gamma, lr, stable=True):
        super().__init__(self.space, rl_batch, gamma, lr, stable)

    @staticmethod
    def create_searching_space(text, font, num_occlu, color=False, H=112, W=112, R=256, G=256, B=256, A=128, shrink=1):
        """
        input:
        num_occlu: the number of occlusion masks
        color: wheather to optimize the color, if it is true,
               7 parameters for each occlusion mask
        H, W: for futher decrease the color size
        notice: when parameterizing the mask, height comes first. e.g. c_dim. After consideration,
                I decide to use two coordinate pairs to parameterize the mask.
        return: list with size 7*num_occlu if color else 4*num_occlu, each item indicates the option number
        """
        # limit search space if H!=W which relates to the create_mask function
        if W > H:
            W = W // shrink
        elif H > W:
            H = H // shrink
        # create space
        search_space = []
        text_size_x, text_size_y = font.getsize(text)
        if color:
            for n in range(num_occlu):
                search_space += [int(H - text_size_y), int(W + text_size_x), int(A), int(R), int(G), int(B)]
        else:
            for n in range(num_occlu):
                search_space += [int(H - text_size_y), int(W + text_size_x), int(A)]
        return search_space

    @staticmethod
    def create_mask(clip, threshmap, points, text, font, distributed_mask=False, C=3, T=16, H=112, W=112, shrink=1):
        """
        clip: torch.floattensor with size (bs, 3, 16, 112, 112)
        points: the pixel coordinates in the image, torch.LongTensor with size (bs, num_occlu * 3 or 6 if color is true)
                if points.size(-1) is a multiple of 6, then distributed_mask=True
        distributed_mask: flag, if it is true, calculate the distributed masks
        return:
        mask_clip: torch.floattensor with size (bs, 3, 16, 112, 112)
        area: torch.floattensor with size (bs, 1)
        iou: torch.floattensor with size (bs, 1)
        """
        bs = points.size(0)
        total = points.size(-1)
        if ~distributed_mask and total % 3 == 0:
            p_l = 3
            num_occlu = total // 3
        elif total % 6 == 0:
            p_l = 6
            num_occlu = total // 6
            assert distributed_mask == True, 'accourding to num_occlu, distributed_mask should be true'
        else:
            assert False, 'occlusion num should be a multiple of 3 or 6'
        # post process combo
        p_combo = []
        a_combo = []
        text_size_x, text_size_y = font.getsize(text)
        if ~distributed_mask:
            for o in range(num_occlu):
                p_combo.append(torch.index_select(points, dim=1, index=torch.LongTensor([o * p_l + 1]).cuda()) - text_size_x)
                p_combo.append(torch.index_select(points, dim=1, index=torch.LongTensor([o * p_l + 0]).cuda()))
                a_combo.append(torch.index_select(points, dim=1, index=torch.LongTensor([o * p_l + 2]).cuda()) + 127)
        else:
            for o in range(num_occlu):
                p_combo.append(torch.index_select(points, dim=1, index=torch.LongTensor([o * p_l + 1]).cuda()) - text_size_x)
                p_combo.append(torch.index_select(points, dim=1, index=torch.LongTensor([o * p_l + 0]).cuda()))
                p_combo.append(torch.index_select(points, dim=1, index=torch.LongTensor([o * p_l + 3]).cuda()))
                p_combo.append(torch.index_select(points, dim=1, index=torch.LongTensor([o * p_l + 4]).cuda()))
                p_combo.append(torch.index_select(points, dim=1, index=torch.LongTensor([o * p_l + 5]).cuda()))
                a_combo.append(torch.index_select(points, dim=1, index=torch.LongTensor([o * p_l + 2]).cuda()) + 127)
        p_combo = torch.cat(p_combo, dim=1)
        a_combo = torch.cat(a_combo, dim=1)
        if distributed_mask:
            mask_clip = clip.clone().detach()
            dis_mask = torch.ones((bs, num_occlu, T, H, W))
            area = torch.zeros(bs, 1)
            iou = torch.zeros(bs, 1)
            # make masks
            for item in range(bs):
                for o in range(num_occlu):
                    for t in range(T):
                        frame = BSCAttack_c3d.tensor_to_pil(mask_clip[item, :, t, :, :])
                        mask_clip[item, :, t, :, :], dis_mask[item, o, t, :, :] = BSCAttack_c3d.mask_RGB_frame(frame, p_combo[item][o * 5 + 0], p_combo[item][o * 5 + 1], p_combo[item][o * 5 + 2], p_combo[item][o * 5 + 3], p_combo[item][o * 5 + 4], a_combo[item][o], t, H, W, text, font, text_size_x, text_size_y)
                area[item] = (((mask_clip[item] - clip[item]) * threshmap) != 0.).sum()
                inter_area = (dis_mask[item].sum(0) < (num_occlu - 1.)).sum()
                union_area = (dis_mask[item].sum(0) != num_occlu).sum()
                iou[item] = inter_area / union_area
            return mask_clip, area, iou
        else:
            mask_clip = clip.clone().detach()
            dis_mask = torch.ones((bs, num_occlu, T, H, W))
            area = torch.zeros(bs, 1)
            iou = torch.zeros(bs, 1)
            # make masks
            for item in range(bs):
                for o in range(num_occlu):
                    for t in range(T):
                        frame = BSCAttack_c3d.tensor_to_pil(mask_clip[item, :, t, :, :])
                        mask_clip[item, :, t, :, :], dis_mask[item, o, t, :, :] = BSCAttack_c3d.mask_frame(frame, p_combo[item][o * 2 + 0], p_combo[item][o * 2 + 1], a_combo[item][o], t, H, W, text, font, text_size_x, text_size_y)
                area[item] = (((mask_clip[item] - clip[item]) * threshmap) != 0.).sum()
                inter_area = (dis_mask[item].sum(0) < (num_occlu - 1.)).sum()
                union_area = (dis_mask[item].sum(0) != num_occlu).sum()
                iou[item] = inter_area / union_area
            for t in range(mask_clip[0].size(1)):
                BSCAttack_c3d.tensor_to_pil(mask_clip[0][:, t, :, :]).save(str(t) + '.png')
            return mask_clip, area, iou

    @staticmethod
    def get_reward(model, mask_input_tensor, target_tensor, area, iou, sigma=200):
        """
        input:
        model: utils.agent.model
        mask_input_tensor: torch.floattensor with size (bs, 3, 16, 112, 112)
        target_tensor: torch.longtensor with size (bs, 1)
        area: torch.floattensor with size (bs, 1)
        iou: torch.floattensor with size (bs, 1)
        sigma: controls penalization for the area, the smaller, the more powerful
        return:
        reward: torch.floattensor with size (bs, 1)
        acc: list of accs, label_acc and target_acc [default None]
        """
        with torch.no_grad():
            deal_dataset = TensorDataset(mask_input_tensor, target_tensor.cpu())
            deal_dataloader = DataLoader(deal_dataset, batch_size=deal_dataset.__len__(), shuffle=False, pin_memory=True)
            for deal_data in deal_dataloader:
                masked_input_tensor, target_tensor = deal_data
            if type(model) == list:
                output_tensor, _ = model[0](masked_input_tensor, bool_magic=True)
            else:
                output_tensor = model(masked_input_tensor)
            
            target_tensor, area, inter_area = target_tensor.cuda(), area.cuda(), iou.cuda()
            output_tensor = F.softmax(output_tensor, dim=1)
            pred = output_tensor.argmax(dim=1)
            
            label_filter = pred == target_tensor.view(-1)
            target_filter = None
            label_acc = label_filter.float().mean()
            target_acc = None
            p_cl = 1. - torch.gather(input=output_tensor, dim=1, index=target_tensor)
            reward = torch.log(p_cl + eps) + (- inter_area / sigma)
            acc = [label_acc, target_acc]
            filters = [label_filter, target_filter]
            return reward, acc, filters

    @staticmethod
    def reward_backward(rewards, gamma):
        """
        input:
        reward: torch.floattensor with size (bs, something)
        gamma: discount factor

        return:
        updated_reward: torch.floattensor with the same size as input
        """
        gamma = 1
        R = 0
        updated_rewards = torch.zeros(rewards.size()).cuda()
        for i in range(rewards.size(-1)):
            R = rewards[:, -(i + 1)] + gamma * R
            updated_rewards[:, -(i + 1)] = R
        return updated_rewards

    def reinforcement_learn(self, steps=150, baseline_subtraction=False):
        """
        input:
        steps: the steps to interact with the environment for the agent
        baseline_subtraction: flag to use baseline subtraction technique.
        return:
        floating_mask_clip: torch.floattensor with size (3, 16, 112, 112)
        area: torch.floattensor with size (1)
        """
        C = self.clip_tensor.size(-4)
        T = self.clip_tensor.size(-3)
        H = self.clip_tensor.size(-2)
        W = self.clip_tensor.size(-1)
        queries = 0
        threshmap = torch.ones((1, 1, T, H, W))
        for t in range(T):
            temp_frame = self.clip_tensor[:, t, :, :].permute(1, 2, 0).numpy().astype("uint8")
            (success, saliencymap) = cv2.saliency.StaticSaliencyFineGrained_create().computeSaliency(temp_frame)
            saliencymap = (saliencymap * 255).astype("uint8")
            threshmap[0, 0, t, :, :] = torch.from_numpy(cv2.threshold(saliencymap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] / 255)
        clip_batch = self.clip_tensor.expand(self.rl_batch, self.clip_tensor.size(-4), self.clip_tensor.size(-3), self.clip_tensor.size(-2), self.clip_tensor.size(-1)).contiguous()
        target_batch = self.target_tensor.expand(self.rl_batch, 1).contiguous()
        self.mind.cuda()
        self.mind = nn.DataParallel(self.mind, device_ids=None)
        self.mind.train()
        self.optimizer.zero_grad()
        # set up non-target attack records
        floating_mask_clip = None
        floating_area = torch.Tensor([C * T * H * W])
        # start learning, interacting with the environments
        for s in range(steps):
            # make combo and get reward
            combo, log_p_combo = self.select_combo()
            rewards = torch.zeros(combo.size()).cuda()
            mask_clip_batch, area, iou = self.create_mask(clip_batch, threshmap, combo, self.text, self.font, distributed_mask=self.color, T=T, H=H, W=W)
            r, acc, filters = self.get_reward(self.model, mask_clip_batch, target_batch, area, iou, sigma=self.sigma)
            queries += mask_clip_batch.size(0)
            rewards[:, -1] = r.squeeze(-1)
            rewards = self.reward_backward(rewards, self.gamma)
            # update records
            wrong_filter = ~filters[0]
            if acc[0] != 1 and iou[wrong_filter].min() == 0.:
                iou_filter = iou[wrong_filter].view(-1) == 0.
                area_candidate = area[wrong_filter][iou_filter]
                temp_floating_area, temp = area_candidate.min(dim=0)
                if temp_floating_area < floating_area:
                    floating_mask_clip = mask_clip_batch[wrong_filter][iou_filter][temp]
                    floating_area = temp_floating_area
                break
            # baseline subtraction
            if baseline_subtraction:
                rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
            # calculate loss
            loss = (-log_p_combo * rewards).sum(dim=1).mean()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.optimizer.zero_grad()
            # reset mind to continuously interact with the environment
            self.mind.module.reset()
        success = floating_mask_clip != None
        return floating_mask_clip, floating_area, success, queries

    @staticmethod
    def attack(model, input_tensor, target_tensor, text, font, sigma, lr=0.03, baseline_subtraction=True, 
               color=False, num_occlu=4, rl_batch=500, steps=50):
        """
        input:
        model: pytorch model
        input_tensor: torch.floattensor with size (3, 16, 112, 112)
        target_tensor: torch.longtensor
        sigma: scalar, contrain the area of the occlusion
        lr: learning rate for p_pi, scalar
        baseline_subtraction: flag to use reward normalization
        color: flag to search the RGB channel values
        return:
        mask_input_tensor: torch.floattensor with size (3, 16, 112, 112)
        area: scalar with size (1)
        """
        # time to start
        attack_begin = time.time()
        actor = agent(model, input_tensor, target_tensor, text, font, num_occlu, color, sigma)
        actor.build_robot(rl_batch=rl_batch, gamma=1, lr=lr, stable=True)
        mask_input_tensor, area, success, queries = actor.reinforcement_learn(steps=steps, baseline_subtraction=baseline_subtraction)
        return mask_input_tensor, area, success, queries, time.time() - attack_begin
