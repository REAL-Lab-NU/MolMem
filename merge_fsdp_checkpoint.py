# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Merge a veRL FSDP actor checkpoint into a Hugging Face model directory."""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.distributed._tensor import Placement
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor


def _merge_tensors(tensors: list[torch.Tensor], placement: Placement) -> torch.Tensor:
    if placement.is_replicate():
        return tensors[0]
    if placement.is_partial():
        raise NotImplementedError("Partial DTensor placement is not supported")
    if placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    raise ValueError(f"Unsupported DTensor placement: {placement}")


def _discover_world_size(checkpoint_dir: Path) -> int:
    for path in checkpoint_dir.iterdir():
        match = re.fullmatch(r"model_world_size_(\d+)_rank_0\.pt", path.name)
        if match:
            return int(match.group(1))
    raise FileNotFoundError(
        f"No model_world_size_*_rank_0.pt checkpoint found in {checkpoint_dir}"
    )


def _load_shards(checkpoint_dir: Path, world_size: int) -> list[dict]:
    paths = [
        checkpoint_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing FSDP shard(s): {', '.join(missing)}")

    def load(path: Path):
        return torch.load(path, map_location="cpu", weights_only=False)

    with ThreadPoolExecutor(max_workers=min(32, world_size)) as executor:
        return list(executor.map(load, paths))


def merge_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    world_size = _discover_world_size(checkpoint_dir)
    shards = _load_shards(checkpoint_dir, world_size)
    keys = set(shards[0])
    if any(set(shard) != keys for shard in shards[1:]):
        raise ValueError("FSDP checkpoint shards do not contain identical parameter keys")

    pivot = shards[0][sorted(keys)[0]]
    if isinstance(pivot, DTensor):
        mesh = pivot.device_mesh.mesh
        mesh_names = pivot.device_mesh.mesh_dim_names
    else:
        mesh = np.array([world_size], dtype=np.int64)
        mesh_names = ("fsdp",)
    if mesh_names not in (("fsdp",), ("ddp", "fsdp")):
        raise NotImplementedError(f"Unsupported FSDP device mesh: {mesh_names}")

    total_shards = int(mesh.shape[-1])
    if total_shards != world_size:
        raise ValueError(
            f"Checkpoint names report {world_size} ranks but DTensor mesh has {total_shards}"
        )

    merged = {}
    for key in sorted(keys):
        values = [shard.pop(key) for shard in shards]
        if isinstance(values[0], DTensor):
            placements = tuple(values[0].placements)
            if mesh_names[0] == "ddp":
                placements = placements[1:]
            if len(placements) != 1:
                raise NotImplementedError("FSDP combined with tensor parallelism is unsupported")
            local_tensors = [value._local_tensor.bfloat16() for value in values]
            merged[key] = _merge_tensors(local_tensors, placements[0])
        else:
            tensors = [value.bfloat16() for value in values]
            merged[key] = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
    return merged


def merge_checkpoint(base_model: str, checkpoint_dir: Path, output_dir: Path) -> None:
    state_dict = merge_state_dict(checkpoint_dir)
    config = AutoConfig.from_pretrained(base_model)
    architectures = config.architectures or []
    if not any(name.endswith("ForCausalLM") for name in architectures):
        raise NotImplementedError(f"Expected a causal language model, got {architectures}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    try:
        model.generation_config = GenerationConfig.from_pretrained(base_model)
    except OSError:
        pass
    model.save_pretrained(output_dir, state_dict=state_dict)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    merge_checkpoint(args.base_model, args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
