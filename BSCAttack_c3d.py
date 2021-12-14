import json
import os
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import BSCAttack_attackers
from BSCAttack_config import configure_BSCA
from spatial_transforms import spatial_Compose, Normalize, ToTensor


def pil_to_tensor(inputs):
    mean = [101.2198, 97.5751, 89.5303]
    std = [1, 1, 1]
    if inputs.mode == 'L':
        transform = spatial_Compose([ToTensor(255)])
    else:
        transform = spatial_Compose([ToTensor(1), Normalize(mean, std)])
    inputs = transform(inputs)
    return inputs


def tensor_to_pil(inputs):
    mean = [-101.2198, -97.5751, -89.5303]
    std = [255, 255, 255]
    inv_transform = transforms.Compose([Normalize(mean, std), transforms.ToPILImage('RGB')])
    inputs = inv_transform(inputs)
    return inputs


def mask_frame(frame, x_pos, y_pos, alpha, t, H, W, text, font, text_size_x, text_size_y):
    rgba_frame = frame.convert('RGBA')
    text_overlay = Image.new('RGBA', rgba_frame.size, (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_overlay)
    text_draw.text((x_pos - t, y_pos), text, font=font, fill=(255, 255, 255, alpha))
    mask_frame = Image.alpha_composite(rgba_frame, text_overlay).convert('RGB')
    rec_overlay = Image.new('L', (H, W), (255))
    rec_draw = ImageDraw.Draw(rec_overlay)
    rec_draw.rectangle((x_pos - t, y_pos, x_pos - t + text_size_x, y_pos + text_size_y), fill=(0))
    return pil_to_tensor(mask_frame), pil_to_tensor(rec_overlay)


def mask_RGB_frame(frame, x_pos, y_pos, R, G, B, alpha, t, H, W, text, font, text_size_x, text_size_y):
    rgba_frame = frame.convert('RGBA')
    text_overlay = Image.new('RGBA', rgba_frame.size, (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_overlay)
    text_draw.text((x_pos - t, y_pos), text, font=font, fill=(R, G, B, alpha))
    mask_frame = Image.alpha_composite(rgba_frame, text_overlay).convert('RGB')
    rec_overlay = Image.new('L', (H, W), (255))
    rec_draw = ImageDraw.Draw(rec_overlay)
    rec_draw.rectangle((x_pos - t, y_pos, x_pos - t + text_size_x, y_pos + text_size_y), fill=(0))
    return pil_to_tensor(mask_frame), pil_to_tensor(rec_overlay)


def process(data_loader, model, opt, class_names):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', 9)
    configure_BSCA(n_occlu=4, rl_batch=500, steps=50, BSCA_color=False)
    total_list = []
    count = 0
    query_num = 0
    total_area = 0
    total_salient = 0
    for i in range(len(data_loader)):
        input_tensor, labels = data_loader[i]
        text = 'This is our adversarial BSC attack.'
        label_tensor = torch.LongTensor([labels[1]]).cuda()
        predict = torch.nn.functional.softmax(model(torch.unsqueeze(input_tensor, dim=0))).argmax(dim=1)
        if predict == label_tensor:
            print('attack')
            dir_title = class_names[int(label_tensor)]
            BSCA = BSCAttack_attackers.BSCA(dir_title)
            adv_clip, rcd, attack_dir = BSCA.attack(text, font, model=model, input_tensor=input_tensor, label_tensor=label_tensor, input_name='{}'.format(labels[0]))
            if rcd.success[0]:
                print('success')
                count += 1
                query_num += rcd.queries[0]
                total_area += rcd.areas[0]
                total_salient += rcd.salients[0]
                for j in range(adv_clip[0].size(1)):
                    frame = tensor_to_pil(adv_clip[0][:, j, :, :])
                    frame.save(attack_dir + '/' + str(j) + '.png')
            total_list.append(labels[0])
        print('[{}/{}]\t'.format(i + 1, len(data_loader)))
    print('攻击之后: ', count)
    print('攻击之前: ', len(total_list))
    print('攻击成功率: ', count / len(total_list))
    print('平均查询数: ', query_num / count)
    print('扰动像素率: ', total_area / count)
    print('扰动显著率: ', total_salient / count)
