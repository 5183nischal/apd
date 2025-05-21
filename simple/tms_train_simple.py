import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from pathlib import Path
from typing import Dict

class SparseFeatureDataset:
    def __init__(self, n_instances, n_features, feature_probability, device, generation="at_least_zero_active"):
        self.n_instances = n_instances
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device
        self.generation = generation

    def generate_batch(self, batch_size):
        if self.generation == "exactly_one_active":
            batch = torch.zeros(batch_size, self.n_instances, self.n_features, device=self.device)
            for inst in range(self.n_instances):
                idx = torch.randint(0, self.n_features, (batch_size,), device=self.device)
                vals = torch.rand(batch_size, device=self.device)
                batch[torch.arange(batch_size), inst, idx] = vals
        else:
            mask = torch.rand(batch_size, self.n_instances, self.n_features, device=self.device) < self.feature_probability
            vals = torch.rand(batch_size, self.n_instances, self.n_features, device=self.device)
            batch = mask.float() * vals
        return batch, batch.clone().detach()

class TMSModel(nn.Module):
    def __init__(self, n_instances, n_features, n_hidden, n_hidden_layers=0, device="cpu"):
        super().__init__()
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers
        self.device = device
        self.W1 = nn.Parameter(torch.empty(n_instances, n_features, n_hidden))
        nn.init.xavier_normal_(self.W1)
        self.b_final = nn.Parameter(torch.zeros(n_instances, n_features))
        if n_hidden_layers > 0:
            self.hidden = nn.ParameterList([
                nn.Parameter(torch.empty(n_instances, n_hidden, n_hidden))
                for _ in range(n_hidden_layers)
            ])
            for p in self.hidden:
                nn.init.xavier_normal_(p)
        else:
            self.hidden = None

    def forward(self, x, topk_mask=None, cache: Dict[str, torch.Tensor] | None = None):
        if cache is not None:
            cache["linear1.hook_pre"] = x
        h = einops.einsum(x, self.W1, "b i f, i f h -> b i h")
        if cache is not None:
            cache["linear1.hook_post"] = h
            cache["linear2.hook_pre"] = h
        if self.hidden is not None:
            for idx, w in enumerate(self.hidden):
                h = einops.einsum(h, w, "b i h, i h m -> b i m")
        out_pre = einops.einsum(h, self.W1.transpose(-1, -2), "b i h, i h f -> b i f") + self.b_final
        if cache is not None:
            cache["linear2.hook_post"] = out_pre
        return F.relu(out_pre)

def train(model, dataset, batch_size=2048, lr=1e-3, steps=2000, print_freq=100):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        batch, labels = dataset.generate_batch(batch_size)
        batch, labels = batch.to(model.device), labels.to(model.device)
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        loss = ((labels.abs() - out) ** 2).mean()
        loss.backward()
        opt.step()
        if step % print_freq == 0 or step == steps - 1:
            print(f"step {step} loss {loss.item():.4f}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TMSModel(n_instances=3, n_features=40, n_hidden=10, n_hidden_layers=0, device=device)
    dataset = SparseFeatureDataset(n_instances=3, n_features=40, feature_probability=0.05, device=device)
    train(model, dataset, batch_size=2048, lr=1e-3, steps=2000, print_freq=200)
    Path("simple_out").mkdir(exist_ok=True)
    torch.save(model.state_dict(), Path("simple_out")/"tms_model.pth")
    print("saved model to", Path("simple_out")/"tms_model.pth")
