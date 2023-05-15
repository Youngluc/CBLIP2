cd /home/chenghao03/CBLIP2
python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path ChineseBLIP2/projects/blip2/train/pretrain_stage1.yaml