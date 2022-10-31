
import torch as th
import torch.nn as nn


class IndexModel(nn.Module):

    def __init__(self,k=7):
        super().__init__()
        self.k = k
        self.misc = nn.Linear(10,10, bias = True)

    def forward(self,mem,dists,T_s,K_s):
        print("mem.shape: ",mem.shape)
        inds = th.zeros_like(mem)
        return inds
