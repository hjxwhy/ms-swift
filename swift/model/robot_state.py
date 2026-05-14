# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
from functools import wraps
from typing import Dict
from types import MethodType

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file

from swift.utils import deep_getattr
from .utils import get_llm_model

ROBOT_STATE_TOKEN = '<|robot_state|>'
ROBOT_STATE_PROJECTOR_NAME = 'robot_state_projector'
ROBOT_STATE_PROJECTOR_PREFIX = f'{ROBOT_STATE_PROJECTOR_NAME}.'
MODEL_MANAGED_MM_KEYS = [
    'pixel_values',
    'pixel_values_videos',
    'image_grid_thw',
    'video_grid_thw',
    'input_features',
    'feature_attention_mask',
    'input_values',
]


class RobotStateProjector(nn.Module):

    def __init__(self, state_dim: int, hidden_size: int, projector_hidden_size: int = None):
        super().__init__()
        projector_hidden_size = projector_hidden_size or hidden_size
        self.net = nn.Sequential(
            nn.Linear(state_dim, projector_hidden_size),
            nn.SiLU(),
            nn.Linear(projector_hidden_size, hidden_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


def get_llm_input_embeddings(model: nn.Module) -> nn.Module:
    if hasattr(model, 'get_input_embeddings'):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return embeddings
    llm_model = get_llm_model(model)
    if hasattr(llm_model, 'get_input_embeddings'):
        embeddings = llm_model.get_input_embeddings()
        if embeddings is not None:
            return embeddings
    for key in ['embed_tokens', 'language_model.embed_tokens', 'model.embed_tokens']:
        try:
            return deep_getattr(llm_model, key)
        except AttributeError:
            pass
    raise AttributeError('Cannot find input embeddings for robot state projector.')


def add_robot_state_projector(model: nn.Module,
                              state_dim: int,
                              projector_hidden_size: int = None,
                              *,
                              token: str = ROBOT_STATE_TOKEN,
                              model_dir: str = None) -> None:
    if model is None or state_dim is None:
        return
    if state_dim <= 0:
        raise ValueError(f'robot_state_dim must be positive, got: {state_dim}')
    if hasattr(model, ROBOT_STATE_PROJECTOR_NAME):
        patch_robot_state_forward(model)
        return
    input_embeddings = get_llm_input_embeddings(model)
    hidden_size = getattr(input_embeddings, 'embedding_dim', None)
    if hidden_size is None:
        hidden_size = input_embeddings.weight.shape[-1]
    projector = RobotStateProjector(state_dim, hidden_size, projector_hidden_size)
    projector.to(device=input_embeddings.weight.device, dtype=input_embeddings.weight.dtype)
    setattr(model, ROBOT_STATE_PROJECTOR_NAME, projector)
    model.config.robot_state_dim = state_dim
    model.config.robot_state_projector_hidden_size = projector_hidden_size
    model.config.robot_state_token = token
    if model_dir:
        state_dict = load_robot_state_projector_state_dict(model_dir)
        if state_dict:
            projector.load_state_dict(state_dict, strict=False)
    patch_robot_state_forward(model)


def _has_model_managed_mm_inputs(kwargs: Dict) -> bool:
    return any(kwargs.get(key) is not None for key in MODEL_MANAGED_MM_KEYS)


def _prepare_robot_state_inputs(model: nn.Module, kwargs: Dict) -> Dict:
    robot_states = kwargs.pop('robot_states', None)
    kwargs.pop('robot_state_counts', None)
    input_ids = kwargs.pop('_robot_state_input_ids', None)
    if input_ids is None:
        input_ids = kwargs.get('input_ids')
    inputs_embeds = kwargs.get('inputs_embeds')
    projector = getattr(model, ROBOT_STATE_PROJECTOR_NAME, None)

    if robot_states is None:
        if model.training and projector is not None:
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = get_llm_input_embeddings(model)(input_ids)
                kwargs['inputs_embeds'] = inputs_embeds
                if not _has_model_managed_mm_inputs(kwargs):
                    kwargs.pop('input_ids', None)
            if inputs_embeds is not None:
                kwargs['inputs_embeds'] = inputs_embeds + sum(p.sum() * 0 for p in projector.parameters())
        return kwargs

    if projector is None:
        raise ValueError('robot_states are provided, but model.robot_state_projector is not initialized.')
    if input_ids is None:
        raise ValueError('robot_states require input_ids to locate robot state token positions.')
    token_id = getattr(model.config, 'robot_state_token_id', None)
    if token_id is None:
        raise ValueError('model.config.robot_state_token_id is required when robot_states are provided.')

    if inputs_embeds is None:
        inputs_embeds = get_llm_input_embeddings(model)(input_ids)
        kwargs['inputs_embeds'] = inputs_embeds
        if not _has_model_managed_mm_inputs(kwargs):
            kwargs.pop('input_ids', None)

    state_mask = input_ids == token_id
    num_state_tokens = int(state_mask.sum().item())
    if num_state_tokens != robot_states.shape[0]:
        raise ValueError(
            f'robot state token count({num_state_tokens}) does not match robot_states count({robot_states.shape[0]}).')
    if num_state_tokens == 0:
        return kwargs

    projector_param = next(projector.parameters())
    state_embeds = projector(robot_states.to(device=projector_param.device, dtype=projector_param.dtype))
    state_embeds = state_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    state_mask = state_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    kwargs['inputs_embeds'] = inputs_embeds.masked_scatter(state_mask, state_embeds)
    return kwargs


def patch_robot_state_forward(model: nn.Module) -> None:
    if model is None or getattr(model, '_robot_state_forward_patched', False):
        return
    origin_forward = model.forward

    @wraps(origin_forward)
    def forward(self, *args, **kwargs):
        kwargs = _prepare_robot_state_inputs(self, kwargs)
        return origin_forward(*args, **kwargs)

    model.forward = MethodType(forward, model)
    model._robot_state_forward_patched = True


def load_robot_state_projector_state_dict(model_dir: str) -> Dict[str, torch.Tensor]:
    candidate_files = [
        'model.safetensors',
        'pytorch_model.bin',
        'adapter_model.safetensors',
        'adapter_model.bin',
    ]
    state_dict = {}
    for filename in candidate_files:
        path = os.path.join(model_dir, filename)
        if not os.path.isfile(path):
            continue
        if filename.endswith('.safetensors'):
            loaded = safe_load_file(path)
        else:
            loaded = torch.load(path, map_location='cpu')
        state_dict.update({
            k[len(ROBOT_STATE_PROJECTOR_PREFIX):]: v
            for k, v in loaded.items() if k.startswith(ROBOT_STATE_PROJECTOR_PREFIX)
        })
    for index_name in ['model.safetensors.index.json', 'pytorch_model.bin.index.json']:
        index_path = os.path.join(model_dir, index_name)
        if not os.path.isfile(index_path):
            continue
        with open(index_path, 'r') as f:
            index = json.load(f)
        shard_files = sorted({
            shard
            for key, shard in index.get('weight_map', {}).items() if key.startswith(ROBOT_STATE_PROJECTOR_PREFIX)
        })
        for shard_file in shard_files:
            shard_path = os.path.join(model_dir, shard_file)
            if shard_file.endswith('.safetensors'):
                loaded = safe_load_file(shard_path)
            else:
                loaded = torch.load(shard_path, map_location='cpu')
            state_dict.update({
                k[len(ROBOT_STATE_PROJECTOR_PREFIX):]: v
                for k, v in loaded.items() if k.startswith(ROBOT_STATE_PROJECTOR_PREFIX)
            })
    return state_dict
