from ray import tune
from tablebench.models.compat import OPTIMIZER_ARGS

# Superset of https://arxiv.org/pdf/2106.11959.pdf, Table 15,
# in order to cover hparams for other searches that derive from this space.
_DEFAULT_NN_SEARCH_SPACE = {
    "d_hidden": tune.choice([64, 128, 256, 512, 1024]),
    "lr": tune.loguniform(1e-5, 1e-1),
    "n_epochs": tune.qrandint(5, 100, 5),
    "num_layers": tune.randint(1, 8),
    "dropouts": tune.uniform(0., 0.5),
    "weight_decay": tune.loguniform(1e-6, 1.)
}

_dann_search_space = {
    **{k: v for k, v in _DEFAULT_NN_SEARCH_SPACE.items()
       if k not in OPTIMIZER_ARGS},
    # Below parameters all use the specified grid from DomainBed.
    # G (classifier) hyperparameters
    "lr_g": _DEFAULT_NN_SEARCH_SPACE["lr"],
    "weight_decay_g": _DEFAULT_NN_SEARCH_SPACE["weight_decay"],
    # D (discriminator) hyperparameters
    "lr_d": _DEFAULT_NN_SEARCH_SPACE["lr"],
    "weight_decay_d": _DEFAULT_NN_SEARCH_SPACE["weight_decay"],
    # Adversarial training parameters
    "d_steps_per_g_step": tune.loguniform(1, 2 ** 3, base=2),
    "grad_penalty": tune.loguniform(1e-2, 1e1),
    "loss_lambda": tune.loguniform(1e-2, 1e2),
}

_deepcoral_search_space = {
    **_DEFAULT_NN_SEARCH_SPACE,
    # Same range as DomainBed, see
    # https://github.com/facebookresearch/DomainBed/blob/main/domainbed
    # /hparams_registry.py#L72
    "mmd_gamma": tune.loguniform(1e-1, 1e1),
}

_dro_search_space = {
    **_DEFAULT_NN_SEARCH_SPACE,
    "geometry": ("cvar", "chi-square"),
    # Note: training is very slow for large values of uncertainty
    # set size (larger than ~0.5) for chi-square geometry, particularly
    # when the learning rate is small.
    # TODO(jpgard): adjust uncertainty set size grid based
    #  on discussion with authors.
    "size": tune.loguniform(1e-4, 1.),

}

_irm_search_space = {
    **_DEFAULT_NN_SEARCH_SPACE,
    # Same tuning space as  for IRM parameters as DomainBed; see
    # https://github.com/facebookresearch/DomainBed/blob
    # /2ed9edf781fe4b336c2fb6ffe7ca8a7c6f994422/domainbed/hparams_registry.py
    # #L61
    "irm_lambda": tune.loguniform(1e-1, 1e5),
    "irm_penalty_anneal_iters": tune.loguniform(1, 1e4)
}

# Similar to XGBoost search space; however, note that LightGBM is not
# use in the study from which the XGBoost space is derived.
_lightgbm_search_space = {
    "learning_rate": tune.loguniform(1e-5, 1.),
    "min_child_samples": tune.choice([1, 2, 4, 8, 16, 32, 64]),
    "min_child_weight": tune.loguniform(1e-8, 1e5),
    "subsample": tune.uniform(0.5, 1),
    "max_depth": tune.choice([-1] + list(range(1, 31))),
    "colsample_bytree": tune.uniform(0.5, 1),
    "colsample_bylevel": tune.uniform(0.5, 1),
    "reg_alpha": tune.loguniform(1e-8, 1e2),
    "reg_lambda": tune.loguniform(1e-8, 1e2),
}

_mixup_search_space = {
    **_DEFAULT_NN_SEARCH_SPACE,
    # Same range as DomainBed, see
    # https://github.com/facebookresearch/DomainBed/blob
    # /2ed9edf781fe4b336c2fb6ffe7ca8a7c6f994422/domainbed/hparams_registry.py
    # #L66
    "mixup_alpha": tune.uniform(10 ** -1, 10 ** 1)
}

_mmd_search_space = {
    **_DEFAULT_NN_SEARCH_SPACE,
    # Same range as DomainBed, see
    # https://github.com/facebookresearch/DomainBed/blob/main/domainbed
    # /hparams_registry.py#L72
    "mmd_gamma": tune.loguniform(1e-1, 1e1),
}

_wcs_search_space = {
    "C_domain": tune.choice([0.001, 0.01, 0.1, 1., 10., 100., 1000.]),
    "C_discrim": tune.choice([0.001, 0.01, 0.1, 1., 10., 100., 1000.]),

}
# Matches https://arxiv.org/pdf/2106.11959.pdf; see Table 16
_xgb_search_space = {
    "max_depth": tune.randint(3, 10),
    "min_child_weight": tune.loguniform(1e-8, 1e5),
    "subsample": tune.uniform(0.5, 1),
    "colsample_bytree": tune.uniform(0.5, 1),
    "colsample_bylevel": tune.uniform(0.5, 1),
    "learning_rate": tune.loguniform(1e-5, 1.),
    "gamma": tune.loguniform(1e-8, 1e2),
    "lambda": tune.loguniform(1e-8, 1e2),
    "alpha": tune.loguniform(1e-8, 1e2),
    "max_bin": tune.choice([128, 256, 512])
}

_expgrad_search_space = {
    **_xgb_search_space,
    "eps": tune.loguniform(1e-4, 1e0),
    "eta0": tune.choice([0.1, 0.2, 1.0, 2.0]),
}

_group_dro_search_space = {
    **_DEFAULT_NN_SEARCH_SPACE,
    "group_weights_step_size": tune.loguniform(1e-4, 1e0),
}

# Superset of https://arxiv.org/pdf/2106.11959.pdf, Table 14.
_resnet_search_space = {
    # Drop the key for d_hidden;
    **{k: v for k, v in _DEFAULT_NN_SEARCH_SPACE.items() if k != "d_hidden"},
    "n_blocks": tune.randint(1, 16),
    "d_main": tune.randint(64, 1024),
    "hidden_factor": tune.randint(1, 4),
    "dropout_first": tune.uniform(0., 0.5),  # after first linear layer
    "dropout_second": tune.uniform(0., 0.5),  # after second/hidden linear layer
}

# Superset of https://arxiv.org/pdf/2106.11959.pdf, Table 13
_ft_transformer_search_space = {
    # Drop the key for d_hidden;
    **{k: v for k, v in _DEFAULT_NN_SEARCH_SPACE.items() if k != "d_hidden"},
    "n_blocks": tune.randint(1, 4),
    "residual_dropout": tune.uniform(0, 0.2),
    "attention_dropout": tune.uniform(0, 0.5),
    "ffn_dropout": tune.uniform(0, 0.5),
    "ffn_factor": tune.uniform(2 / 3, 8 / 3),
    # This is feature embedding size in Table 13 above. Note that Table 13
    # reports this as a uniform (64, 512) parameter, but d_token *must* be a
    # multiple of n_heads, which is fixed at 8, so we use this
    # simpler/equivalent range instead.
    "d_token": tune.choice([64, 128, 256, 512])
}

_vrex_search_space = {
    **_DEFAULT_NN_SEARCH_SPACE,
    "vrex_lambda": tune.loguniform(1e-1, 1e5),
    "vrex_penalty_anneal_iters": tune.loguniform(1, 1e4),
}

search_space = {
    "dann": _dann_search_space,
    "deepcoral": _deepcoral_search_space,
    "dro": _dro_search_space,
    "expgrad": _expgrad_search_space,
    "ft_transformer": _ft_transformer_search_space,
    "group_dro": _group_dro_search_space,
    "irm": _irm_search_space,
    "lightgbm": _lightgbm_search_space,
    "mixup": _mixup_search_space,
    "mlp": _DEFAULT_NN_SEARCH_SPACE,
    "mmd": _mmd_search_space,
    "resnet": _resnet_search_space,
    "vrex": _vrex_search_space,
    "wcs": _wcs_search_space,
    "xgb": _xgb_search_space,
}
