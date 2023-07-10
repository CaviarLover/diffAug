# diffAug

Required package:
pytorch
torchvision
numpy
visdom

Please organe the data in this format:)
/Path_To_TrainSet/
    images/
        img1.png
        img2.png
        ...
    masks/
        img1.png
        img2.png
        ...

Training Steps:
1. Activate the visualization tool: python -m visdom.server -p 9000
2. Run the training file. python train.py --data_dir /Path_To_TrainSet/ --lr_anneal_steps 60000 --batch_size 10

Sampling:
python sample.py --out_dir /Path_to_Save/ --num_samples 1000 --model_path "./ckpts/emasavedmodel_0.9999_060000.pt" --dev "gpuindex"

Note:
If you want to change the image size, please change it in the "./guided_diffusion/script_util-model_and_diffusion_defaults"
