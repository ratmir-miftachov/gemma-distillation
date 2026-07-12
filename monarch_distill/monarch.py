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
        # Hugging Face constructs modules on the meta device for low-memory loading.
        # The serialized factors materialize them later, so there is nothing to initialize yet.
        if self.blk1.is_meta:
            return

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

    @torch.no_grad()
    def initialize_from_dense(self, dense_layer: nn.Linear) -> float:
        """Project a dense linear map onto this rectangular Monarch layout."""
        if dense_layer.in_features != self.in_features or dense_layer.out_features != self.out_features:
            raise ValueError(
                "dense layer shape does not match Monarch layer: "
                f"got ({dense_layer.out_features}, {dense_layer.in_features}), "
                f"expected ({self.out_features}, {self.in_features})"
            )

        weight = dense_layer.weight.detach().to(device=self.blk1.device, dtype=torch.float32)
        slices = (
            weight.reshape(self.n3, self.n2, self.n1, self.n2)
            .permute(1, 2, 0, 3)
            .contiguous()
        )
        left_vectors, singular_values, right_vectors_h = torch.linalg.svd(slices, full_matrices=False)

        scales = singular_values[..., 0].clamp_min(0.0).sqrt()
        projected_blk2 = left_vectors[..., :, 0] * scales.unsqueeze(-1)
        projected_blk1 = right_vectors_h[..., 0, :] * scales.unsqueeze(-1)

        self.blk1.copy_(projected_blk1.permute(1, 2, 0).to(dtype=self.blk1.dtype))
        self.blk2.copy_(projected_blk2.to(dtype=self.blk2.dtype))

        if self.bias is not None:
            if dense_layer.bias is None:
                self.bias.zero_()
            else:
                self.bias.copy_(dense_layer.bias.detach().to(device=self.bias.device, dtype=self.bias.dtype))

        squared_singular_values = singular_values.square()
        total_energy = squared_singular_values.sum()
        residual_energy = squared_singular_values[..., 1:].sum()
        if total_energy.item() == 0.0:
            return 0.0
        return (residual_energy / total_energy).clamp_min(0.0).sqrt().item()

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


def replace_linear_with_monarch(module, blocks, init_method="identity_noise", module_path=""):
    for name, child in module.named_children():
        child_path = f"{module_path}.{name}" if module_path else name
        if isinstance(child, nn.Linear):
            device = child.weight.device
            dtype = child.weight.dtype
            monarch_layer = MonarchLinear(child.in_features, child.out_features, blocks, bias=child.bias is not None)
            monarch_layer = monarch_layer.to(device=device, dtype=dtype)
            if init_method == "dense_projection":
                relative_error = monarch_layer.initialize_from_dense(child)
                print(f"[Projection] {child_path} | relative Frobenius error: {relative_error:.6f}")
            elif init_method != "identity_noise":
                raise ValueError(f"unsupported Monarch initialization method: {init_method}")
            setattr(module, name, monarch_layer)
        else:
            replace_linear_with_monarch(child, blocks, init_method=init_method, module_path=child_path)
    return module


def replace_with_monarch(
    student_model,
    module_path: str,
    blocks_weights: int,
    blocks_head: int,
    init_method: str = "identity_noise",
):
    parent_path = ".".join(module_path.split(".")[:-1])
    child_name = module_path.split(".")[-1]

    parent_module = student_model if parent_path == "" else student_model.get_submodule(parent_path)
    old_module = getattr(parent_module, child_name)

    if module_path == "lm_head":
        device, dtype = old_module.weight.device, old_module.weight.dtype
        new_head = MonarchLinear(old_module.in_features, old_module.out_features, blocks_head, bias=old_module.bias is not None)
        new_head = new_head.to(device=device, dtype=dtype)
        if init_method == "dense_projection":
            relative_error = new_head.initialize_from_dense(old_module)
            print(f"[Projection] {module_path} | relative Frobenius error: {relative_error:.6f}")
        elif init_method != "identity_noise":
            raise ValueError(f"unsupported Monarch initialization method: {init_method}")
        setattr(parent_module, child_name, new_head)
        return getattr(parent_module, child_name)

    if "embed_tokens" in module_path:
        device, dtype = old_module.weight.device, old_module.weight.dtype
        new_embed = MonarchEmbedding(old_module.num_embeddings, old_module.embedding_dim, blocks_head)
        setattr(parent_module, child_name, new_embed.to(device=device, dtype=dtype))
        return getattr(parent_module, child_name)

    replace_linear_with_monarch(
        old_module,
        blocks_weights,
        init_method=init_method,
        module_path=module_path,
    )
    return old_module
