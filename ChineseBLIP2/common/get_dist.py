import deepspeed
import torch.distributed as tr_dist
from deepspeed.comm import comm as ds_dist

class getDistributedLib:
    @property
    def dist(self):
        if deepspeed.utils.dist.is_initialized():
            return ds_dist
        else:
            return tr_dist

    def __getattr__(self, name):
        if name in self.dist.__dict__:
            return getattr(self.dist, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

dist = getDistributedLib()