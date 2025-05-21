import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from tms_train_simple import TMSModel, SparseFeatureDataset
from pathlib import Path
from typing import Dict

class LinearComponent(nn.Module):
    def __init__(self, d_in, d_out, C, n_instances, m=None):
        super().__init__()
        self.C = C
        self.n_instances = n_instances
        self.m = min(d_in, d_out) + 1 if m is None else m
        self.A = nn.Parameter(torch.randn(n_instances, C, d_in, self.m) * 0.02)
        self.B = nn.Parameter(torch.randn(n_instances, C, self.m, d_out) * 0.02)

    @property
    def component_weights(self):
        return einops.einsum(self.A, self.B, 'i C d_in m, i C m d_out -> i C d_in d_out')

    def forward(self, x, topk_mask=None, cache:Dict[str,torch.Tensor]=None, name=''):
        if cache is not None:
            cache[name+'.hook_pre'] = x
        inner = einops.einsum(x, self.A, 'b i d_in, i C d_in m -> b i C m')
        if topk_mask is not None:
            inner = inner * topk_mask.unsqueeze(-1)
        comp = einops.einsum(inner, self.B, 'b i C m, i C m d_out -> b i C d_out')
        if cache is not None:
            cache[name+'.hook_component_acts'] = comp
        out = comp.sum(dim=2)
        if cache is not None:
            cache[name+'.hook_post'] = out
        return out

class TransposedLinearComponent(nn.Module):
    def __init__(self, original_A, original_B):
        super().__init__()
        self.original_A = original_A
        self.original_B = original_B
        self.n_instances, self.C, _, self.m = original_A.shape

    @property
    def A(self):
        return einops.einsum(self.original_B, torch.ones(1), 'i C m d_out, b -> i C d_out m')

    @property
    def B(self):
        return einops.einsum(self.original_A, torch.ones(1), 'i C d_in m, b -> i C m d_in')

    @property
    def component_weights(self):
        return einops.einsum(self.A, self.B, 'i C d_out m, i C m d_in -> i C d_out d_in')

    def forward(self, x, topk_mask=None, cache=None, name=''):
        A = self.A
        B = self.B
        if cache is not None:
            cache[name+'.hook_pre'] = x
        inner = einops.einsum(x, A, 'b i d, i C d m -> b i C m')
        if topk_mask is not None:
            inner = inner * topk_mask.unsqueeze(-1)
        comp = einops.einsum(inner, B, 'b i C m, i C m k -> b i C k')
        if cache is not None:
            cache[name+'.hook_component_acts'] = comp
        out = comp.sum(dim=2)
        if cache is not None:
            cache[name+'.hook_post'] = out
        return out

class TMSSPDModel(nn.Module):
    def __init__(self, n_instances, n_features, n_hidden, C, bias_val=0.0, m=None, n_hidden_layers=0, device='cpu'):
        super().__init__()
        self.n_instances=n_instances
        self.n_features=n_features
        self.C=C
        self.linear1=LinearComponent(n_features,n_hidden,C,n_instances,m)
        self.linear2=TransposedLinearComponent(self.linear1.A, self.linear1.B)
        self.b_final=nn.Parameter(torch.zeros(n_instances,n_features)+bias_val)
        if n_hidden_layers>0:
            self.hidden_layers=nn.ModuleList([
                LinearComponent(n_hidden,n_hidden,C,n_instances,m)
                for _ in range(n_hidden_layers)
            ])
        else:
            self.hidden_layers=None
        self.device=device

    def forward(self,x,topk_mask=None,cache=None):
        h=self.linear1(x,topk_mask,cache, name='linear1')
        if self.hidden_layers is not None:
            for idx,layer in enumerate(self.hidden_layers):
                h=layer(h,topk_mask,cache,name=f'hidden_layers.{idx}')
        out=self.linear2(h,topk_mask,cache,name='linear2')+self.b_final
        return F.relu(out)

def run_with_cache(model,x,topk_mask=None):
    cache={}
    out=model(x,topk_mask,cache)
    return out,cache

def calc_grad_attributions(target_out,pre_acts,post_acts,component_weights,C):
    attr_shape=target_out.shape[:-1]+(C,)
    attrib=torch.zeros(attr_shape,device=target_out.device)
    comp_acts={}
    for name in pre_acts:
        comp_acts[name]=einops.einsum(pre_acts[name].detach(),component_weights[name],'... d_in, i C d_in d_out -> ... C d_out')
    for idx in range(target_out.shape[-1]):
        grads=torch.autograd.grad(target_out[...,idx].sum(), list(post_acts.values()), retain_graph=True)
        feat_attr=torch.zeros(attr_shape,device=target_out.device)
        for g,(name,comp) in zip(grads,comp_acts.items()):
            feat_attr+=einops.einsum(g,comp,'... d_out, ... C d_out -> ... C')
        attrib+=feat_attr**2
    return attrib

def calc_topk_mask(attributions, topk, batch_topk=True):
    batch_size=attributions.shape[0]
    k=int(topk*batch_size) if batch_topk else int(topk)
    if batch_topk:
        flat=attributions.reshape(-1, attributions.shape[-1])
        idx=flat.topk(k,dim=-1).indices
        mask=torch.zeros_like(flat,dtype=torch.bool)
        mask.scatter_(-1,idx,True)
        return mask.reshape_as(attributions)
    idx=attributions.topk(k,dim=-1).indices
    mask=torch.zeros_like(attributions,dtype=torch.bool)
    mask.scatter_(-1,idx,True)
    return mask

def calc_recon_mse(out1,out2,has_instance_dim):
    diff=(out1-out2)**2
    return diff.mean(dim=tuple(range(1,diff.ndim)))

def optimize(spd_model,target_model,dataset,steps=1000,lr=1e-3,batch_size=512,topk=2.0,C=40):
    device=spd_model.device
    opt=torch.optim.AdamW(spd_model.parameters(),lr=lr,weight_decay=0.0)
    for step in range(steps):
        batch,_=dataset.generate_batch(batch_size)
        batch=batch.to(device)
        target_out,target_cache=run_with_cache(target_model,batch)
        out,spd_cache=run_with_cache(spd_model,batch)
        mse=calc_recon_mse(out,target_out,True).mean()
        comp_weights={'linear1':spd_model.linear1.component_weights,'linear2':spd_model.linear2.component_weights}
        attrs=calc_grad_attributions(target_out,{k: v for k,v in target_cache.items() if k.endswith('hook_pre')},{k:v for k,v in target_cache.items() if k.endswith('hook_post')},comp_weights,C)
        mask=calc_topk_mask(attrs,topk,True)
        out_topk,_=run_with_cache(spd_model,batch,mask)
        topk_loss=calc_recon_mse(out_topk,target_out,True).mean()
        loss=topk_loss+mse
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step%100==0 or step==steps-1:
            print('step',step,'loss',loss.item())
    Path('simple_out').mkdir(exist_ok=True)
    torch.save(spd_model.state_dict(), Path('simple_out')/'spd_model.pth')
    print('saved to',Path('simple_out')/'spd_model.pth')

if __name__=='__main__':
    device='cuda' if torch.cuda.is_available() else 'cpu'
    target=TMSModel(n_instances=3,n_features=40,n_hidden=10,n_hidden_layers=0,device=device)
    target.load_state_dict(torch.load(Path('simple_out')/'tms_model.pth',map_location=device))
    spd=TMSSPDModel(n_instances=3,n_features=40,n_hidden=10,C=40,bias_val=0.0,device=device)
    dataset=SparseFeatureDataset(n_instances=3,n_features=40,feature_probability=0.05,device=device)
    optimize(spd,target,dataset,steps=2000,lr=1e-3,batch_size=2048,topk=2.0,C=40)
