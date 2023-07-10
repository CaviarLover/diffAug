"""
Train a diffusion model on images.
"""
import sys
import argparse
sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.kvasirloader import Dataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args.multi_gpu, dv=args.gpu_dev)
    logger.configure(dir=args.out_dir)

    logger.log("creating model and diffusion...")

    ds = Dataset(args.data_dir, args.image_size)
    datal= th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True)
    data = iter(datal)
    args.in_channel = 4 # 3 channels for rgb image, 1 for segmentation mask

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    logger.log("creating data loader...")

    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps
    ).run_loop()


def create_argparser():
    defaults = dict(
        # change the data directory here
        data_dir="./Data/MoNuSeg/Trainset", #"../Data/MoNuSeg/Trainset" "./Data/Kvasir/TrainDataset"
        out_dir="./ckpts", # save the checkpoint
        # how to sample from [0, T]
        schedule_sampler="uniform", # Improved DDPM "loss-second-moment"
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=5000, # number of iterations, -1 for endless iterations
        batch_size=10,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=5000,
        resume_checkpoint="",#"./results/savedmodel005000.pt",
        use_fp16=False,
        gpu_dev="0",
        fp16_scale_growth=1e-3,
        multi_gpu="" #"0,1,2"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
