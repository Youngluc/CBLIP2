cd /home/chenghao03/CBLIP2
CUDA_VISIBLE_DEVICE=1,2 python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path ChineseBLIP2/projects/blip2/train/pretrain_stage2.yaml