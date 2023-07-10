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

    #model = th.nn.DataParallel(model,device_ids=[int(id) for id in "1"])
    model.to(dist_util.dev())

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    image_path = os.path.join(args.out_dir, 'images')
    label_path = os.path.join(args.out_dir, 'masks')
    os.makedirs(image_path)
    os.makedirs(label_path)

    i = 0
    have = len(os.listdir(image_path))

    while(have + i*args.batch_size <= args.num_samples):

        guided_mask = None
        print(f"sampling {str(i)}")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        
        model_kwargs = {}
        start.record()
        sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
 
        imgs, masks = sample_fn(
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
            tv.utils.save_image(imgs[j], os.path.join(image_path, f'fake{have+i*args.batch_size + j}.png'))
            tv.utils.save_image(masks[j], os.path.join(label_path, f'fake{have+i*args.batch_size + j}.png'))

        print(f"Sample {have+i*args.batch_size + j} finished")
        i += 1
           

def create_argparser():
    defaults = dict(
        out_dir="./diffAug/generated_images/Monuseg_fake/",
        clip_denoised=True, # if True, clip the denoised signal into [0, 1].
        num_samples=1449, 
        batch_size=5, 
        seed=2333,
        use_ddim=False,
        model_path="./ckpts/emasavedmodel_0.9999_060000.pt", # some checkpoints
        fast="",
        dev="0"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
