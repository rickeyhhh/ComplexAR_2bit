import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from .kernel import fairytoi_quant_block_V2


class QATLinearComplexReorder(nn.Linear):
    """
    Complex reorder QAT with two-sided output-channel reordering.

    Rule:
     1) Sort rows by absolute score (descending).
     2) Interleave ranked rows into upper/lower halves.
         Example: rank1->upper[0], rank2->lower[0], rank3->upper[1], rank4->lower[1].
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.out_features % 2 != 0 or self.in_features % 2 != 0:
            raise ValueError("QATLinearComplexReorder requires even in/out features")
        self.register_buffer("_row_perm", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("_row_inv", torch.empty(0, dtype=torch.long), persistent=True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        perm_key = prefix + "_row_perm"
        inv_key = prefix + "_row_inv"
        if perm_key not in state_dict:
            state_dict[perm_key] = torch.empty(0, dtype=torch.long)
        if inv_key not in state_dict:
            state_dict[inv_key] = torch.empty(0, dtype=torch.long)

        if state_dict[perm_key].shape != self._row_perm.shape:
            self._buffers["_row_perm"] = torch.empty(
                state_dict[perm_key].shape,
                dtype=self._row_perm.dtype,
                device=self._row_perm.device,
            )
        if state_dict[inv_key].shape != self._row_inv.shape:
            self._buffers["_row_inv"] = torch.empty(
                state_dict[inv_key].shape,
                dtype=self._row_inv.dtype,
                device=self._row_inv.device,
            )

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def _build_reorder_perm(self, A: torch.Tensor) -> torch.Tensor:
        num_channels = A.shape[0]
        half = num_channels // 2
        score = A.mean(dim=1).abs()
        sorted_src_idx = torch.argsort(score, descending=True)
        perm = torch.empty(num_channels, device=A.device, dtype=torch.long)
        perm[:half] = sorted_src_idx[0::2]
        perm[half:] = sorted_src_idx[1::2]
        return perm

    @torch.no_grad()
    def _get_cached_permutation(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        need_rebuild = (
            self._row_perm.numel() != A.shape[0]
            or self._row_inv.numel() != A.shape[0]
            or self._row_perm.device != A.device
            or self._row_inv.device != A.device
        )

        if need_rebuild:
            row_perm = self._build_reorder_perm(A)
            row_inv = torch.empty_like(row_perm)
            row_inv[row_perm] = torch.arange(row_perm.numel(), device=row_perm.device)
            self._row_perm = row_perm
            self._row_inv = row_inv
            print(f'  [reorder2 QAT] built new perm: out_features={A.shape[0]}')

        return self._row_perm, self._row_inv

    def forward(self, x):
        A = self.weight
        row_perm, row_inv = self._get_cached_permutation(A)

        A_reordered = A[row_perm, :]

        # Triton kernel returns bfloat16 quantized matrix
        B_quant_bf16 = fairytoi_quant_block_V2(A_reordered)
        # Cast back and preserve gradient via residual detach trick
        B_quant = B_quant_bf16.to(A_reordered.dtype)
        A_quant_reordered = (B_quant - A_reordered).detach() + A_reordered

        # Restore original row order and run linear
        A_quant = A_quant_reordered[row_inv, :]
        return F.linear(x, A_quant, self.bias)


METHOD_MAP = {
    'complex_reorder': QATLinearComplexReorder,
}


def replace_modules_for_qat(model: nn.Module, method: str, skip_output_layer: bool = False):
    """Recursively replace nn.Linear layers in the model with QAT layers"""
    if method not in METHOD_MAP:
        raise ValueError(f"Unknown method: {method}. Available methods: {list(METHOD_MAP.keys())}")

    TargetQATClass = METHOD_MAP[method]

    for name, module in model.named_children():
        logging.info(f"name: {name}, module: {module}")
        if len(list(module.children())) > 0:
            replace_modules_for_qat(module, method, skip_output_layer)

        if isinstance(module, nn.Linear):
            if skip_output_layer and name == 'output':
                print(f"  -> Skipping output layer: {name}")
                continue

            if module.in_features % 2 != 0 or module.out_features % 2 != 0:
                print(f"  -> Skipping replacement (non-even dimensions): {name} ({module.in_features}, {module.out_features})")
                continue

            print(f"  -> Replacing layer: {name} with {TargetQATClass.__name__}")
            new_module = TargetQATClass(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                dtype=module.weight.dtype,
                device=module.weight.device
            )
            new_module.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_module.bias.data.copy_(module.bias.data)

            setattr(model, name, new_module)


class InferenceOptimizedComplexReorder(nn.Linear):
    """Inference-optimized Complex Reorder linear layer."""

    def __init__(self, version="reorder2", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_features % 2 != 0 or self.out_features % 2 != 0:
            raise ValueError("Complex requires even in/out features.")
        self._is_quantized = False
        self._version = version.lower()
        self.register_buffer("_row_perm", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("_row_inv", torch.empty(0, dtype=torch.long), persistent=True)
        if self._version not in ["reorder"]:
            raise ValueError(f"Unsupported version: {version}. Only 'reorder' is supported.")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        perm_key = prefix + "_row_perm"
        inv_key = prefix + "_row_inv"
        if perm_key not in state_dict:
            state_dict[perm_key] = torch.empty(0, dtype=torch.long)
        if inv_key not in state_dict:
            state_dict[inv_key] = torch.empty(0, dtype=torch.long)

        if state_dict[perm_key].shape != self._row_perm.shape:
            self._buffers["_row_perm"] = torch.empty(
                state_dict[perm_key].shape,
                dtype=self._row_perm.dtype,
                device=self._row_perm.device,
            )
        if state_dict[inv_key].shape != self._row_inv.shape:
            self._buffers["_row_inv"] = torch.empty(
                state_dict[inv_key].shape,
                dtype=self._row_inv.dtype,
                device=self._row_inv.device,
            )

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    @torch.no_grad()
    def set_reorder2_permutation(self, row_perm: torch.Tensor, row_inv: torch.Tensor):
        if row_perm.numel() != self.out_features or row_inv.numel() != self.out_features:
            raise ValueError("reorder2 permutation shape mismatch")
        if row_perm.device != self.weight.device:
            row_perm = row_perm.to(self.weight.device)
        if row_inv.device != self.weight.device:
            row_inv = row_inv.to(self.weight.device)
        self._row_perm = row_perm.clone()
        self._row_inv = row_inv.clone()

    def _build_reorder_perm(self, A: torch.Tensor) -> torch.Tensor:
        num_channels = A.shape[0]
        half = num_channels // 2
        score = A.mean(dim=1).abs()
        sorted_src_idx = torch.argsort(score, descending=True)
        perm = torch.empty(num_channels, device=A.device, dtype=torch.long)
        perm[:half] = sorted_src_idx[0::2]
        perm[half:] = sorted_src_idx[1::2]
        return perm

    def _ensure_quantized(self):
        """Ensure weights are quantized, executed only once"""
        if not self._is_quantized:
            with torch.no_grad():
                A = self.weight

                if self._row_perm.numel() == A.shape[0] and self._row_inv.numel() == A.shape[0]:
                    row_perm = self._row_perm
                    row_inv = self._row_inv
                    print(f'  [reorder2 infer] using loaded perm: out_features={A.shape[0]}')
                else:
                    row_perm = self._build_reorder_perm(A)
                    row_inv = torch.empty_like(row_perm)
                    row_inv[row_perm] = torch.arange(row_perm.numel(), device=row_perm.device)
                    self._row_perm = row_perm
                    self._row_inv = row_inv
                    print(f'  [reorder2 infer] built new perm (no cached): out_features={A.shape[0]}')

                A_reordered = A[row_perm, :]

                # Quantize the reordered full matrix with the Triton kernel.
                A_q_reordered = fairytoi_quant_block_V2(A_reordered)
                self.weight.data = A_q_reordered[row_inv, :].to(self.weight.dtype)
                self._is_quantized = True

    def forward(self, x):
        self._ensure_quantized()
        return F.linear(x, self.weight, self.bias)


def convert_to_inference_mode(model):
    """Convert QAT modules to inference-optimized version (permanently modifies model weights)"""
    converted_count = 0

    def _convert_module(module, name_path=""):
        nonlocal converted_count

        for name, child in list(module.named_children()):
            full_name = f"{name_path}.{name}" if name_path else name

            if isinstance(child, QATLinearComplexReorder):
                version = "reorder"

                new_module = InferenceOptimizedComplexReorder(
                    version=version,
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype
                )
                new_module.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_module.bias.data.copy_(child.bias.data)
                row_perm = child._row_perm
                row_inv = child._row_inv
                if row_perm.numel() == child.out_features and row_inv.numel() == child.out_features:
                    new_module.set_reorder2_permutation(row_perm, row_inv)

                setattr(module, name, new_module)
                converted_count += 1
                print(f"  -> Converting Reorder2 layer: {full_name}")
            else:
                _convert_module(child, full_name)

    _convert_module(model)
    print(f"Converted {converted_count} QAT layers to inference-optimized version")
    return model
