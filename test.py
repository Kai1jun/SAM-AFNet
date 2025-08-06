import argparse
import os

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms
from mmcv.runner import load_checkpoint

import numpy as np
import cv2


def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 保存掩膜图像的路径
output_dir = "output/masks"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 修改eval_psnr函数以保存掩膜
def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')

    for batch_idx, batch in enumerate(pbar):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']
        # points = batch['points']
        labels = batch['labels']
        pred = torch.sigmoid(model.infer(inp, labels))

        # 使用detach()避免梯度计算
        pred_mask = pred.detach().squeeze(1).cpu().numpy()  # 断开梯度并转换为numpy数组

        # 保存掩膜图像
        for i in range(pred_mask.shape[0]):  # 遍历批次中的每个样本
            mask = pred_mask[i]
            mask = (mask * 255).astype(np.uint8)  # 将掩膜值转换为0-255之间的整数
            filename = os.path.join(output_dir, f"mask_{batch_idx * len(batch['inp']) + i:06d}.png")
            cv2.imwrite(filename, mask)

        result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
        val_metric1.add(result1.item(), inp.shape[0])
        val_metric2.add(result2.item(), inp.shape[0])
        val_metric3.add(result3.item(), inp.shape[0])
        val_metric4.add(result4.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
            pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
            pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
            pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()


# def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
#               verbose=False):
#     model.eval()
#     if data_norm is None:
#         data_norm = {
#             'inp': {'sub': [0], 'div': [1]},
#             'gt': {'sub': [0], 'div': [1]}
#         }

#     if eval_type == 'f1':
#         metric_fn = utils.calc_f1
#         metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
#     elif eval_type == 'fmeasure':
#         metric_fn = utils.calc_fmeasure
#         metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
#     elif eval_type == 'ber':
#         metric_fn = utils.calc_ber
#         metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
#     elif eval_type == 'cod':
#         metric_fn = utils.calc_cod
#         metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

#     val_metric1 = utils.Averager()
#     val_metric2 = utils.Averager()
#     val_metric3 = utils.Averager()
#     val_metric4 = utils.Averager()

#     pbar = tqdm(loader, leave=False, desc='val')

#     for batch in pbar:
#         for k, v in batch.items():
#             batch[k] = v.cuda()

#         inp = batch['inp']
#         # by Kaijun
#         points = batch['points']
#         labels = batch['labels']
#         pred = torch.sigmoid(model.infer(inp,points,labels))

#         result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
#         val_metric1.add(result1.item(), inp.shape[0])
#         val_metric2.add(result2.item(), inp.shape[0])
#         val_metric3.add(result3.item(), inp.shape[0])
#         val_metric4.add(result4.item(), inp.shape[0])

#         if verbose:
#             pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
#             pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
#             pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
#             pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))

#     return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8)

    model = models.make(config['model']).cuda()
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)
    
    metric1, metric2, metric3, metric4 = eval_psnr(loader, model,
                                                   data_norm=config.get('data_norm'),
                                                   eval_type=config.get('eval_type'),
                                                   eval_bsize=config.get('eval_bsize'),
                                                   verbose=True)
    print('metric1: {:.4f}'.format(metric1))
    print('metric2: {:.4f}'.format(metric2))
    print('metric3: {:.4f}'.format(metric3))
    print('metric4: {:.4f}'.format(metric4))
