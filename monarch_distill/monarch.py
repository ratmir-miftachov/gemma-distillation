import math

import torch
import torch.nn as nn


class MonarchLinear(nn.Module):
    def __init__(self, in_features, out_features, n_blocks, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_down_proj = out_features < in_features

        if not self.is_down_proj:
            self.n1 = n_blocks
            assert in_features % self.n1 == 0, f"in_features ({in_features}) must be divisible by n_blocks ({n_blocks})"
            self.n2 = in_features // self.n1
            assert out_features % self.n2 == 0, f"out_features ({out_features}) must be divisible by in_block_size ({self.n2})"
            self.n3 = out_features // self.n2
        else:
            self.n3 = n_blocks
            assert out_features % self.n3 == 0, f"out_features ({out_features}) must be divisible by n_blocks ({n_blocks})"
            self.n2 = out_features // self.n3
            assert in_features % self.n2 == 0, f"in_features ({in_features}) must be divisible by out_block_size ({self.n2})"
            self.n1 = in_features // self.n2

        self.blk1 = nn.Parameter(torch.empty(self.n1, self.n2, self.n2))
        self.blk2 = nn.Parameter(torch.empty(self.n2, self.n1, self.n3))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.blk1)
        nn.init.zeros_(self.blk2)

        for i in range(self.blk1.shape[0]):
            self.blk1.data[i].copy_(torch.eye(self.blk1.shape[1]))

        for i in range(self.blk2.shape[0]):
            min_dim = min(self.blk2.shape[1], self.blk2.shape[2])
            self.blk2.data[i, :min_dim, :min_dim].copy_(torch.eye(min_dim))

            if self.blk2.shape[1] != self.blk2.shape[2]:
                with torch.no_grad():
                    bound = 1.0 / (min_dim ** 0.5)
                    noise = torch.FloatTensor(self.blk2.shape[1], self.blk2.shape[2]).uniform_(-bound, bound)
                    mask = torch.ones(self.blk2.shape[1], self.blk2.shape[2])
                    mask[:min_dim, :min_dim] -= torch.eye(min_dim)
                    self.blk2.data[i] += (noise * mask).to(self.blk2.device)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        orig_shape = x.shape
        x = x.contiguous().view(-1, self.n1, self.n2)
        x = torch.einsum("bij, ijk -> bik", x, self.blk1)
        x = x.transpose(1, 2).contiguous()
        x = torch.einsum("bij, ijk -> bik", x, self.blk2)
        x = x.transpose(1, 2).contiguous()
        x = x.view(*orig_shape[:-1], self.out_features)

        if self.bias is not None:
            x = x + self.bias

        return x


class MonarchEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, n_blocks):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.n1 = n_blocks
        assert num_embeddings % self.n1 == 0, "num_embeddings must be divisible by n_blocks"
        self.n2 = num_embeddings // self.n1

        assert embedding_dim % self.n2 == 0, f"embedding_dim ({embedding_dim}) must be divisible by in_block_size ({self.n2})"
        self.n3 = embedding_dim // self.n2

        self.blk1 = nn.Parameter(torch.empty(self.n1, self.n2, self.n2))
        self.blk2 = nn.Parameter(torch.empty(self.n2, self.n1, self.n3))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.blk1, mean=0, std=1 / math.sqrt(self.n2))
        nn.init.normal_(self.blk2, mean=0, std=1 / math.sqrt(self.n1))

    def forward(self, x):
        blk1_flat = self.blk1.view(self.num_embeddings, self.n2)
        v1 = blk1_flat[x]

        block_idx = x // self.n2
        blk2_transposed = self.blk2.transpose(0, 1)
        w2_selected = blk2_transposed[block_idx]

        out_matrix = v1.unsqueeze(-1) * w2_selected
        out = out_matrix.transpose(-1, -2).contiguous()
        out = out.view(*x.shape, self.embedding_dim)
        return out


def replace_linear_with_monarch(module, blocks):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            device = child.weight.device
            dtype = child.weight.dtype
            monarch_layer = MonarchLinear(child.in_features, child.out_features, blocks, bias=child.bias is not None)
            setattr(module, name, monarch_layer.to(device=device, dtype=dtype))
        else:
            replace_linear_with_monarch(child, blocks)
    return module


def replace_with_monarch(student_model, module_path: str, blocks_weights: int, blocks_head: int):
    parent_path = ".".join(module_path.split(".")[:-1])
    child_name = module_path.split(".")[-1]

    parent_module = student_model if parent_path == "" else student_model.get_submodule(parent_path)
    old_module = getattr(parent_module, child_name)

    if module_path == "lm_head":
        device, dtype = old_module.weight.device, old_module.weight.dtype
        new_head = MonarchLinear(old_module.in_features, old_module.out_features, blocks_head, bias=old_module.bias is not None)
        setattr(parent_module, child_name, new_head.to(device=device, dtype=dtype))
        return getattr(parent_module, child_name)

    if "embed_tokens" in module_path:
        device, dtype = old_module.weight.device, old_module.weight.dtype
        new_embed = MonarchEmbedding(old_module.num_embeddings, old_module.embedding_dim, blocks_head)
        setattr(parent_module, child_name, new_embed.to(device=device, dtype=dtype))
        return getattr(parent_module, child_name)

    replace_linear_with_monarch(old_module, blocks_weights)
    return old_module
