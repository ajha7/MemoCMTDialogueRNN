from torch import nn, optim
from configs.base import Config
from typing import Union, List


def _build_param_groups(cfg: Config, network: nn.Module):
    """Build optimizer parameter groups with differential LRs when available.

    Groups:
    - encoders (text/audio): cfg.learning_rate
    - fusion layers in MemoCMT (non-encoder): cfg.fusion_learning_rate (fallback to cfg.learning_rate)
    - dialogue layers (ctx + dialogue classifier): cfg.dialogue_learning_rate (fallback to cfg.learning_rate)
    - others: cfg.learning_rate
    """
    try:
        all_trainable = [p for p in network.parameters() if p.requires_grad]
    except Exception:
        return None

    # If network has no hierarchy (e.g., plain MemoCMT), just return all params
    has_memo = hasattr(network, 'memo')
    has_ctx = hasattr(network, 'ctx') and isinstance(getattr(network, 'ctx'), nn.Module)

    if not has_memo and not has_ctx:
        return all_trainable

    enc_params: List[nn.Parameter] = []
    fusion_params: List[nn.Parameter] = []
    dialog_params: List[nn.Parameter] = []

    if has_memo:
        memo = getattr(network, 'memo')
        if hasattr(memo, 'text_encoder'):
            enc_params += [p for p in memo.text_encoder.parameters() if p.requires_grad]
        if hasattr(memo, 'audio_encoder'):
            enc_params += [p for p in memo.audio_encoder.parameters() if p.requires_grad]

        enc_ids = {id(p) for p in enc_params}
        fusion_params = [p for p in memo.parameters() if p.requires_grad and id(p) not in enc_ids]

    if has_ctx:
        dialog_params += [p for p in network.ctx.parameters() if p.requires_grad]
    if hasattr(network, 'classifier'):
        dialog_params += [p for p in network.classifier.parameters() if p.requires_grad]

    assigned_ids = {id(p) for p in enc_params + fusion_params + dialog_params}
    other_params = [p for p in all_trainable if id(p) not in assigned_ids]

    groups = []
    if enc_params:
        groups.append({'params': enc_params, 'lr': getattr(cfg, 'learning_rate', 1e-5)})
    if fusion_params:
        groups.append({'params': fusion_params, 'lr': getattr(cfg, 'fusion_learning_rate', getattr(cfg, 'learning_rate', 1e-5))})
    if dialog_params:
        groups.append({'params': dialog_params, 'lr': getattr(cfg, 'dialogue_learning_rate', getattr(cfg, 'learning_rate', 1e-5))})
    if other_params:
        groups.append({'params': other_params, 'lr': getattr(cfg, 'learning_rate', 1e-5)})

    # If grouping failed for some reason, fallback to flat params
    return groups if groups else all_trainable


def adamw(cfg: Config, network: nn.Module) -> optim.AdamW:
    params = _build_param_groups(cfg, network)
    return optim.AdamW(
        # params=network.parameters(),
        params=params,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta_1, cfg.adam_beta_2),
        eps=cfg.adam_eps,
        weight_decay=0.01,
    )


def adam(cfg: Config, network: nn.Module) -> optim.Adam:
    params = _build_param_groups(cfg, network)
    return optim.Adam(
        # params=network.parameters(),
        params=params,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta_1, cfg.adam_beta_2),
        eps=cfg.adam_eps,
        weight_decay=cfg.adam_weight_decay,
    )


def sgd(cfg: Config, network: nn.Module) -> optim.SGD:
    params = _build_param_groups(cfg, network)
    return optim.SGD(
        # params=network.parameters(),
        params=params,
        lr=cfg.learning_rate,
        momentum=cfg.momemtum,
        weight_decay=cfg.sdg_weight_decay,
    )


def get_optim(cfg: Config, network: nn.Module) -> Union[optim.SGD, optim.Adam]:
    optim_fn = {
        "SGD": sgd,
        "Adam": adam,
        "AdamW": adamw,
    }
    assert cfg.optimizer_type in optim_fn.keys(), (
        "Invalid optimizer_type. The valid optim is ["
        + " ".join(list(optim_fn.keys()))
        + "]"
    )

    return optim_fn[cfg.optimizer_type](cfg, network)
