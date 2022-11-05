
import torch as th
import torch.nn as nn
from einops import rearrange
from collections import OrderedDict


class IndexModel(nn.Module):

    def __init__(self,dim,ws,T_m,K_m,T_s,K_s,alpha=0.99):
        super().__init__()
        self.dim = dim
        self.ws = ws
        self.T_m = T_m
        self.K_m = K_m
        self.T_s = T_s
        self.K_s = K_s
        self.alpha = alpha

    def create_kde_map(self,inds):
        # -- unpack --
        B,H,T_m,nH,nW,K_m,_ = inds.shape
        inds = inds.view(B*H*T_m*nH*nW,K_m,3)
        print(inds[0])
        diff = inds[:,:,None,:] - inds[:,None,:,:]
        print("diff.shape: ",diff.shape)
        diff = th.exp(-th.mean(diff,-1))
        print("diff.shape: ",diff.shape)
        print(diff[0])


    def exp_smoothing(self,inds):

        trange = th.arange(inds.shape[2])
        alphas = self.alpha*th.pow((1-self.alpha),trange).to(inds.device)
        Z = th.sum(alphas)
        alphas = rearrange(alphas,'t -> 1 1 t 1 1 1 1')

        inds_pred = th.sum(alphas * inds,2,keepdim=True)/Z
        inds_pred = inds_pred.type(th.int32)
        inds_pred = inds_pred[...,:self.K_s,:].contiguous()

        T = th.max(inds[...,0])
        H = th.max(inds[...,1])
        W = th.max(inds[...,2])

        inds_pred[...,0] = 0#T+1
        inds_pred = th.clip(inds_pred,0)
        for i,V in enumerate([T,H,W]):
            inds_pred[...,i] = th.clip(inds_pred[...,i],0,V)
        return inds_pred

    def forward(self,inds_mem,dists):

        # -- unpack --
        B,H,T_m,nH,nW,K_m,_ = inds_mem.shape
        device = inds_mem.device
        dtype = inds_mem.dtype

        # -- allocate --
        inds_pred = self.exp_smoothing(inds_mem)
        # print("inds_mem.shape: ",inds_mem.shape)
        # exit(0)
        # inds_pred = th.zeros((B,H,T_s,nH,nW,K_s,3),device=device,dtype=dtype)
        # print("inds_pred.shape: ",inds_pred.shape)
        # inds_pred = inds_pred.view(B,H,-1,self.K_s,3)
        # print("inds_pred.shape: ",inds_pred.shape)
        print(inds_pred)

        return inds_pred
