from ray import tune

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

_deepcoral_search_space = {
    **_DEFAULT_NN_SEARCH_SPACE,
    # Tune s.t. CORAL loss is roughly of same order as the
    # overall loss.
    "loss_lambda": tune.loguniform(1e-5, 1),
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

_ft_transformer_search_space = {
    **_DEFAULT_NN_SEARCH_SPACE,
    "n_blocks": tune.randint(1, 4),
    # TODO(jpgard): tune the remaining parameters here; it is hard to parse
    #  how values from https://arxiv.org/pdf/2106.11959.pdf Table 13 map to
    #  ft-transformer params.
}

search_space = {
    "deepcoral": _deepcoral_search_space,
    "expgrad": _expgrad_search_space,
    "ft_transformer": _ft_transformer_search_space,
    "group_dro": _group_dro_search_space,
    "lightgbm": _lightgbm_search_space,
    "mlp": _DEFAULT_NN_SEARCH_SPACE,
    "resnet": _resnet_search_space,
    "wcs": _wcs_search_space,
    "xgb": _xgb_search_space,
}
