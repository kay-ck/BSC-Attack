import json
import os
import random
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import lgs
import PatchAttack_attackers as PA
from PatchAttack_config import configure_PA, PA_cfg
from spatial_transforms import spatial_Compose, Normalize, ToTensor


def pil_to_tensor(inputs):
    mean = [0, 0, 0]
    std = [1, 1, 1]
    if inputs.mode == 'L':
        transform = spatial_Compose([ToTensor(255)])
        inputs = transform(inputs)
    else:
        transform = spatial_Compose([ToTensor(255), Normalize(mean, std)])
        inputs = transform(inputs).permute(1, 2, 0)
    return inputs


def tensor_to_pil(inputs):
    mean = [0, 0, 0]
    std = [1, 1, 1]
    inv_transform = transforms.Compose([Normalize(mean, std), transforms.ToPILImage('RGB')])
    inputs = inv_transform(inputs.permute(2, 0, 1))
    return inputs


def mask_frame(frame, x_pos, y_pos, alpha, t, H, W, patch):
    rgba_frame = frame.convert('RGBA')
    patch_overlay = Image.new('RGBA', rgba_frame.size, (255, 255, 255, 0))
    patch_draw = ImageDraw.Draw(patch_overlay)
    patch_draw.rectangle((x_pos - t, y_pos, x_pos - t + patch, y_pos + patch), fill=(255, 255, 255, alpha))
    mask_frame = Image.alpha_composite(rgba_frame, patch_overlay).convert('RGB')
    rec_overlay = Image.new('L', (H, W), (255))
    rec_draw = ImageDraw.Draw(rec_overlay)
    rec_draw.rectangle((x_pos - t, y_pos, x_pos - t + patch, y_pos + patch), fill=(0))
    return pil_to_tensor(mask_frame), pil_to_tensor(rec_overlay)


def check_overlay(text, font, combo, n_occlu):
    combo = combo.squeeze(0)
    text_size_x, text_size_y = font.getsize(text)
    p_l = len(combo) // n_occlu
    for i in range(n_occlu):
        for j in range(i + 1, n_occlu):
            if (abs(combo[i * p_l + 0] - combo[j * p_l + 0]) < text_size_y) and (abs(combo[i * p_l + 1] - combo[j * p_l + 1]) < text_size_x):
                return True
    return False


def process(data_loader, model, opt, class_names):
    with open(os.path.join(opt.result_path, 'ucf101_patch_lrcn.txt'), 'r') as it:
        index_patch = json.load(it)
    configure_PA(target=False, n_occlu=1, rl_batch=500, steps=50, MPA_color=False)
    total_list = []
    count = 0
    query_num = 0
    total_area = 0
    total_salient = 0
    with open(os.path.join(opt.result_path, 'ucf101_196_lrcn.txt'), 'r') as f:
        content = f.readlines()
        index = []
        for ids in content:
            index.append(int(ids.strip('\n')))
    for i in index:
        input_tensor, labels = data_loader[i]
        patch = index_patch[str(i)]
        label_tensor = torch.LongTensor([labels[1]])
        predict = torch.nn.functional.softmax(model(torch.unsqueeze(input_tensor, dim=0))).argmax(dim=1)
        if predict == label_tensor:
            print('attack')
            dir_title = class_names[int(label_tensor)]
            MPA = PA.MPA(dir_title)
            adv_clip, rcd, attack_dir = MPA.attack(patch, model=model, input_tensor=input_tensor, label_tensor=label_tensor, target=45, input_name='{}'.format(labels[0]))
            if rcd.non_target_success[0]:
                print('success')
                count += 1
                query_num += rcd.queries[0]
                total_area += rcd.areas[0]
                total_salient += rcd.salients[0]
                for j in range(adv_clip[0].size(0)):
                    frame = tensor_to_pil(adv_clip[0][j, :, :, :])
                    frame.save(attack_dir + '/' + str(i * 16 + j) + '.png')
            total_list.append(labels[0])
        print('[{}/{}]\t'.format(i + 1, len(data_loader)))
    print('攻击之后：', count)
    print('攻击之前：', len(total_list))
    print('攻击成功率：', count / len(total_list))
    print('平均查询数', query_num / count)
    print('扰动像素率：', total_area / count)
    print('扰动显著率：', total_salient / count)
