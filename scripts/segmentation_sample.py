"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
import sys
import random
#sys.path.append("..")
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
import nibabel as nib
import pathlib
import warnings
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()
    seed = args.seed
   #result0=th.load('./Bratssliced/validation/000246/result0')
  #  print('loadedresult0', result0.shape)
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ds = BRATSDataset(args.data_dir, test_flag=False, 
                      normalize=(lambda x: 2*x - 1) if args.renormalize else None,
                      mode=args.data_mode,
                      half_resolution=(args.image_size == 128) and not args.half_res_crop,
                      random_half_crop=(args.image_size == 128) and args.half_res_crop,
                      concat_coords=args.concat_coords,
                      )
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False)
    all_images = []
    all_labels = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices
    if args.use_fp16:
        raise ValueError("fp16 currently not implemented")
        #model.convert_to_fp16()
    model.eval()
    for datal_data, ind in zip(datal, range(args.num_samples // args.batch_size)):
        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"reseeded (in for loop) to {seed}")
        raw_img, out_dict, weak_label, label, number = datal_data
        img = raw_img
        model_kwargs = {}
        output_shape = (args.batch_size, args.out_channels) + args.dims * (args.image_size,)
        sample_fn = diffusion.ddim_sample_loop_known
        sample, x_noisy, org = sample_fn(
            model,
            output_shape,
            img,
            mode='segmentation',
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            sampling_steps=args.sampling_steps,
        )
        print('done sampling')
        if sample.shape[0] != 1:
            warnings.warn('we are discarding batch>1 (should be implemented)')
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        if sample.shape[1] == 1:
            output_name = os.path.join(args.output_dir, f'segmentation_{number[0]}.nii.gz')
            img = nib.Nifti1Image(sample.detach().cpu().numpy()[0, 0, ...], np.eye(4))
            nib.save(img=img, filename=output_name)
        else:
            sample_save = sample.detach().cpu().numpy().squeeze()
            for k in range(sample_save.shape[0]):
                img = nib.Nifti1Image(sample_save[k, ...], np.eye(4))
                output_name = os.path.join(args.output_dir, f'segmentation_{number[0]}_{k}.nii.gz')
                nib.save(img=img, filename=output_name)
        print(f'saved to {output_name}')

        dice = lambda x, y: 2*(x * y).sum()/(x**2 + y**2).sum()
        binary_sample = sample > 0.5
        print(f'DSC = {dice(binary_sample, label.to(sample.device)>0).item()}')


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        data_mode='validation',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir='./results',
        mode='segmentation',
        renormalize=False,
        image_size=256,
        half_res_crop=False,
        concat_coords=False, # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
    )
    defaults.update({k:v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
