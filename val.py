#! /data/cxli/miniconda3/envs/th200/bin/python
import argparse
import os
from glob import glob
import random
import numpy as np
from PIL import Image
from PIL import ImageFont, ImageDraw
import archs
import torch.nn.functional as F  # make sure this is imported
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import OrderedDict
from PIL import ImageFont, ImageDraw
import archs

from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time

from PIL import Image
# class SegGradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.activations = None
#         self.gradients = None

#         # Hook to save activations
#         self.fwd_hook = target_layer.register_forward_hook(self._forward_hook)
#         # Hook to save gradients
#         self.bwd_hook = target_layer.register_backward_hook(self._backward_hook)

#     def _forward_hook(self, module, inp, out):
#         self.activations = out.detach()

#     def _backward_hook(self, module, grad_in, grad_out):
#         self.gradients = grad_out[0].detach()

#     def remove_hooks(self):
#         self.fwd_hook.remove()
#         self.bwd_hook.remove()

#     def generate(self, x, class_idx=None, return_output=False):
#         """
#         x: input tensor of shape (1, C, H, W)
#         returns:
#             cam_np: (H, W) normalized to [0,1]
#             (optionally) logits: model raw output (1, num_classes, H, W)
#         """

#         self.model.zero_grad()
#         out = self.model(x)  # (B, num_classes, H, W)

#         # We assume binary segmentation (num_classes=1) or multi-channel mask.
#         if out.shape[1] == 1:
#             # foreground logit
#             score = out[:, 0, :, :].max()
#         else:
#             # choose class_idx or default to class=1 if exists
#             if class_idx is None:
#                 class_idx = 1 if out.shape[1] > 1 else 0
#             score = out[:, class_idx, :, :].max()

#         score.backward(retain_graph=True)

#         grads = self.gradients          # (B, C, H', W')
#         acts = self.activations         # (B, C, H', W')

#         # Global average pooling over H' and W'
#         weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

#         # Weighted sum
#         cam = (weights * acts).sum(dim=1, keepdim=True)  # (B, 1, H', W')
#         cam = F.relu(cam)

#         # Normalize CAM
#         cam_min = cam.min()
#         cam_max = cam.max()
#         if cam_max - cam_min > 1e-5:
#             cam = (cam - cam_min) / (cam_max - cam_min)
#         else:
#             cam = torch.zeros_like(cam)

#         # Resize to input size
#         cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
#         cam_np = cam[0, 0].cpu().numpy()  # (H, W)

#         if return_output:
#             return cam_np, out.detach()
#         else:
#             return cam_np

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='outputs', help='ouput dir')
            
    args = parser.parse_args()

    return args

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    args = parse_args()

    with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'], embed_dims=config['input_list'])

    model = model.cuda()
    # ---------- add this block right after `model = model.cuda()` ----------

    try:
        from thop import profile, clever_format
        flops_supported = True
    except:
        print("⚠️ THOP not installed. Install using: pip install thop")
        flops_supported = False

    def count_model_params(m):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    total_params, trainable_params = count_model_params(model)
    print(f"Total params: {total_params} ({total_params/1e6:.3f} M)")
    print(f"Trainable params: {trainable_params} ({trainable_params/1e6:.3f} M)")

        # ---- Compute GFLOPs using THOP ----
    if flops_supported:
        # create dummy input based on model input size
        dummy_input = torch.randn(1, config['input_channels'], 
                                  config['input_h'], config['input_w']).cuda()

        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")

        print(f"FLOPs: {flops}")
        print(f"Params (THOP): {params}")

        # save results
        try:
            os.makedirs(os.path.join(args.output_dir, config['name']), exist_ok=True)
            with open(os.path.join(args.output_dir, config['name'], 'model_params.txt'), 'w') as f:
                f.write(f"Total params: {total_params} ({total_params/1e6:.3f} M)\n")
                f.write(f"Trainable params: {trainable_params} ({trainable_params/1e6:.3f} M)\n")
                f.write(f"FLOPs: {flops}\n")
        except Exception as e:
            print("Could not save params file:", e)

    # ----------------------------------------------------------------------------------------------------------------------------

    dataset_name = config['dataset']
    img_ext = '.png'

    if dataset_name == 'busi':
        mask_ext = '.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name == 'isic':
        mask_ext = '.png'
    elif dataset_name == 'cvc':
        mask_ext = '.png'

    # Data loading code

    base='/content/drive/MyDrive/Akanksha/MIDL_2/U-KAN/Seg_UKAN/DATA/split_seed_43'
    test_img_dir = os.path.join(base, 'test', 'images')
    test_msk_dir = os.path.join(base, 'test', 'masks')


    test_img_ids = sorted(glob(os.path.join(test_img_dir, '*' + img_ext)))
    # img_ids.sort()
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_img_ids]

    # _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    ckpt = torch.load('/content/drive/MyDrive/Akanksha/MIDL_2/U-KAN/Seg_UKAN/outputs_busi_MIDL_seed44/busi/model_updated.pth')

    try:        
        model.load_state_dict(ckpt)
    except:
        print("Pretrained model keys:", ckpt.keys())
        print("Current model keys:", model.state_dict().keys())

        pretrained_dict = {k: v for k, v in ckpt.items() if k in model.state_dict()}
        current_dict = model.state_dict()
        diff_keys = set(current_dict.keys()) - set(pretrained_dict.keys())

        print("Difference in model keys:")
        for key in diff_keys:
            print(f"Key: {key}")

        model.load_state_dict(ckpt, strict=False)
    # print("\nGenerating Grad-CAM heatmaps ONLY (no overlay)...")
    # target_layer = model.decoder1  # D_ConvLayer: output (B, C, H, W)
    # grad_cam = SegGradCAM(model, target_layer)

    # gradcam_dir = os.path.join(args.output_dir, config['name'], 'gradcam_82')
    # os.makedirs(gradcam_dir, exist_ok=True)  
    # model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=test_img_dir,
        mask_dir=test_msk_dir,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    sensitivity_meter = AverageMeter()
    specificity_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()
         # We need gradients, so DO NOT use torch.no_grad here
    # with torch.enable_grad():
    #    for input, target, meta in tqdm(val_loader, total=len(val_loader)):
    #     input = input.cuda()
    #     target = target.cuda()
    #     input.requires_grad_()

    #     batch_size = input.shape[0]
    #     for b in range(batch_size):
    #         img_id = meta['img_id'][b]
    #         x = input[b].unsqueeze(0)     # (1, C, H, W)
    #         y = target[b].unsqueeze(0)    # (1, 1, H, W)

    #         # --- 1. Grad-CAM + logits ---
    #         cam_np, logits = grad_cam.generate(x, return_output=True)

    #         # --- 2. Compute IoU ---
    #         prob = torch.sigmoid(logits)[0, 0]
    #         pred_bin = (prob > 0.5).float()
    #         gt = y[0, 0]
    #         gt_bin = (gt > 0.5).float()
    #         inter = (pred_bin * gt_bin).sum()
    #         union = ((pred_bin + gt_bin) > 0).float().sum()
    #         iou_img = (inter + 1e-7) / (union + 1e-7)
    #         iou_val = float(iou_img.item())

    #         # --- 3. Load image ---
    #         orig_path = os.path.join(test_img_dir, img_id + img_ext)
    #         orig = cv2.imread(orig_path)
    #         if orig is None:
    #             print(f"Warning: could not read {orig_path}, skipping.")
    #             continue

    #         orig_resized = cv2.resize(orig, (config['input_w'], config['input_h']))

    #         # --- 4. Convert CAM to heatmap ---
    #         cam_uint8 = (cam_np * 255).astype(np.uint8)
    #         heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    #         # --- 5. Overlay ---
    #         overlay = cv2.addWeighted(orig_resized, 0.5, heatmap, 0.5, 0)

    #         # --- 6. Add IoU text with Times New Roman ---
    #         overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    #         draw = ImageDraw.Draw(overlay_pil)
    #         try:
    #             font_path = "/content/drive/MyDrive/fonts/times.ttf"
    #             font = ImageFont.truetype(font_path, size=18)
    #         except IOError:
    #             font = ImageFont.load_default()
    #         draw.text((10, 10), f"Plausibility IoU: {iou_val:.3f}", font=font, fill=(255, 255, 255))
    #         overlay = cv2.cvtColor(np.array(overlay_pil), cv2.COLOR_RGB2BGR)

    #         # --- 7. Save ---
    #         overlay_path = os.path.join(gradcam_dir, f"{img_id}_overlay_iou.jpg")
    #         cv2.imwrite(overlay_path, overlay)
    #         print("Saved:", overlay_path)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()

            # Compute model output
            output = model(input)

            # Compute IoU, Dice, and HD95
            iou, dice, hd95_ = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            hd95_avg_meter.update(hd95_, input.size(0))

            # Convert predictions to binary mask
            output = torch.sigmoid(output).cpu().numpy()
            target = target.cpu().numpy()

            output_bin = (output > 0.5).astype(np.float32)
            target_bin = (target > 0.5).astype(np.float32)

            # Calculate confusion matrix elements
            TP = ((output_bin == 1) & (target_bin == 1)).sum()
            TN = ((output_bin == 0) & (target_bin == 0)).sum()
            FP = ((output_bin == 1) & (target_bin == 0)).sum()
            FN = ((output_bin == 0) & (target_bin == 1)).sum()

            eps = 1e-7
            sensitivity = TP / (TP + FN + eps)
            specificity = TN / (TN + FP + eps)
            accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
            precision = TP / (TP + FP + eps)
            recall = TP / (TP + FN + eps)
            f1 = (2 * precision * recall) / (precision + recall + eps)

            # Update metric meters
            sensitivity_meter.update(sensitivity)
            specificity_meter.update(specificity)
            accuracy_meter.update(accuracy)
            precision_meter.update(precision)
            recall_meter.update(recall)
            f1_meter.update(f1)

            # Save predictions
            os.makedirs(os.path.join(args.output_dir, config['name'], 'out_val'), exist_ok=True)
            for pred, img_id in zip(output_bin, meta['img_id']):
                pred_np = pred[0].astype(np.uint8) * 255
                img = Image.fromarray(pred_np, 'L')
                img.save(os.path.join(args.output_dir, config['name'], f'out_val/{img_id}.jpg'))

            torch.cuda.empty_cache()

        # --- Print final test results ---
        print(config['name'])
        print('IoU: %.4f' % iou_avg_meter.avg)
        print('Dice: %.4f' % dice_avg_meter.avg)
        print('HD95: %.4f' % hd95_avg_meter.avg)
        print(f"Sensitivity: {sensitivity_meter.avg * 100:.2f}%")
        print(f"Specificity: {specificity_meter.avg * 100:.2f}%")
        print(f"Accuracy: {accuracy_meter.avg * 100:.2f}%")
        print(f"Precision: {precision_meter.avg:.4f}")
        print(f"Recall: {recall_meter.avg:.4f}")
        print(f"F1 Score: {f1_meter.avg:.4f}")
        print("Testing completed successfully!")


  # with torch.no_grad():
  #   for input, target, meta in tqdm(val_loader, total=len(val_loader)):
  #         input = input.cuda()
  #         target = target.cuda()
  #         model = model.cuda()

  #         # Compute model output
  #         output = model(input)

  #         # Compute IoU, Dice, and HD95
  #         iou, dice, hd95_ = iou_score(output, target)
  #         iou_avg_meter.update(iou, input.size(0))
  #         dice_avg_meter.update(dice, input.size(0))
  #         hd95_avg_meter.update(hd95_, input.size(0))

  #         # Convert predictions to binary mask
  #         output = torch.sigmoid(output).cpu().numpy()
  #         target = target.cpu().numpy()

  #         output_bin = (output > 0.5).astype(np.float32)
  #         target_bin = (target > 0.5).astype(np.float32)

  #         # Calculate confusion matrix elements
  #         TP = ((output_bin == 1) & (target_bin == 1)).sum()
  #         TN = ((output_bin == 0) & (target_bin == 0)).sum()
  #         FP = ((output_bin == 1) & (target_bin == 0)).sum()
  #         FN = ((output_bin == 0) & (target_bin == 1)).sum()

  #         eps = 1e-7
  #         sensitivity = TP / (TP + FN + eps)
  #         specificity = TN / (TN + FP + eps)
  #         accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
  #         precision = TP / (TP + FP + eps)
  #         recall = TP / (TP + FN + eps)
  #         f1 = (2 * precision * recall) / (precision + recall + eps)

  #         # Update metric meters
  #         sensitivity_meter.update(sensitivity)
  #         specificity_meter.update(specificity)
  #         accuracy_meter.update(accuracy)
  #         precision_meter.update(precision)
  #         recall_meter.update(recall)
  #         f1_meter.update(f1)

  #         # Save predictions
  #         os.makedirs(os.path.join(args.output_dir, config['name'], 'out_val'), exist_ok=True)
  #         for pred, img_id in zip(output_bin, meta['img_id']):
  #             pred_np = pred[0].astype(np.uint8) * 255
  #             img = Image.fromarray(pred_np, 'L')
  #             img.save(os.path.join(args.output_dir, config['name'], f'out_val/{img_id}.jpg'))

  #         torch.cuda.empty_cache()

  #   # --- Print final test results ---
  #   print(config['name'])
  #   print('IoU: %.4f' % iou_avg_meter.avg)
  #   print('Dice: %.4f' % dice_avg_meter.avg)
  #   print('HD95: %.4f' % hd95_avg_meter.avg)
  #   print(f"Sensitivity: {sensitivity_meter.avg * 100:.2f}%")
  #   print(f"Specificity: {specificity_meter.avg * 100:.2f}%")
  #   print(f"Accuracy: {accuracy_meter.avg * 100:.2f}%")
  #   print(f"Precision: {precision_meter.avg:.4f}")
  #   print(f"Recall: {recall_meter.avg:.4f}")
  #   print(f"F1 Score: {f1_meter.avg:.4f}")
  #   print("Testing completed successfully!")
# with torch.no_grad():
   
    # with torch.no_grad():
    #     for input, target, meta in tqdm(val_loader, total=len(val_loader)):
    #         input = input.cuda()
    #         target = target.cuda()
    #         model = model.cuda()
    #         # compute output
    #         output = model(input)

    #         iou, dice, hd95_ = iou_score(output, target)
    #         iou_avg_meter.update(iou, input.size(0))
    #         dice_avg_meter.update(dice, input.size(0))
    #         hd95_avg_meter.update(hd95_, input.size(0))

    #         output = torch.sigmoid(output).cpu().numpy()
    #         output[output>=0.5]=1
    #         output[output<0.5]=0



    #         os.makedirs(os.path.join(args.output_dir, config['name'], 'out_val'), exist_ok=True)
    #         for pred, img_id in zip(output, meta['img_id']):
    #             pred_np = pred[0].astype(np.uint8)
    #             pred_np = pred_np * 255
    #             img = Image.fromarray(pred_np, 'L')
    #             img.save(os.path.join(args.output_dir, config['name'], 'out_val/{}.jpg'.format(img_id)))

    
    # print(config['name'])
    # print('IoU: %.4f' % iou_avg_meter.avg)
    # print('Dice: %.4f' % dice_avg_meter.avg)
    # print('HD95: %.4f' % hd95_avg_meter.avg)



if __name__ == '__main__':
    main()


