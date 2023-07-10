"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys
import random
sys.path.append(".")
import numpy as np
import torch as th
from guided_diffusion import dist_util
import torchvision as tv
from guided_diffusion.kvasirloader import KvasirTestset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from batchgenerators.utilities.file_and_folder_operations import *

def main():
    
    args = create_argparser().parse_args()
    dist_util.setup_dist(None, dv=args.dev)
    #logger.configure()

    seed=args.seed
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.fast is not None:
        args.timestep_respacing=args.fast
    
    ds= KvasirTestset(directory=args.in_dir, testsize=args.image_size)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)
    model.to(dist_util.dev())

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    for i in range(ds.size):

        image, guided_mask, name = ds.load_data()

        save_dir = os.path.join(args.out_dir, name)
        maybe_mkdir_p(save_dir)

        tv.utils.save_image(image, os.path.join(save_dir, f'gt.png'))
        tv.utils.save_image(guided_mask, os.path.join(save_dir, f'mask.png'))

        print(f"sampling {str(i)}")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        
        model_kwargs = {}
        start.record()
        sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
   
        imgs, _ = sample_fn(
                model,
                (args.batch_size, args.in_channel, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                progress=True,
                mask=guided_mask
            )

        end.record()
        th.cuda.synchronize()
        print(f'time for {args.batch_size} sample', start.elapsed_time(end) / 1000, 'seconds')  #time measurement for the generation of 1 sample
        print("saving")

        for j in range(imgs.shape[0]):
            img = imgs[j].data.cpu().numpy().transpose(1, 2, 0)
            img = img*255
            tv.utils.save_image(imgs[j], os.path.join(save_dir, f'fake{j}.png'))

def create_argparser():
    defaults = dict(
        in_dir="", # input dataset
        out_dir="./diffAug/generated_images/Kvasir_train_mask_induced_2/",#"./generated_images/Kvasir_256_v1_new/", # Monuseg_512_cvsave_v1_new
        clip_denoised=True, # if True, clip the denoised signal into [0, 1].
        batch_size=5, 
        image_size=256,
        seed=2333,
        use_ddim=False,
        model_path="./ckpts/emasavedmodel_0.9999_060000.pt",
        fast=None, # DDIM: "ddim100", TimeStepRespacing: "100"
        dev="0"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
