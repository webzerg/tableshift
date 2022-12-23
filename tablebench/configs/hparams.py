from ray import tune

_DEFAULT_NN_SEARCH_SPACE = {
    "d_hidden": tune.choice([64, 128, 256, 512, 1024]),
    "lr": tune.loguniform(1e-5, 1e-1),
    "n_epochs": tune.qrandint(5, 100, 5),
    "num_layers": tune.randint(1, 8),
    "dropouts": tune.uniform(0., 0.5),
    "weight_decay": tune.quniform(0., 1., 0.1),
}

_histgbm_search_space = {
    "learning_rate": tune.choice([0.1, 0.3, 1.0, 2.0]),
    "max_leaf_nodes": tune.choice([None, 2, 4, 8, 16, 32, 64]),
    "l2_regularization": tune.choice(
        [0., 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.]),
    "max_bins": tune.choice([32, 64, 128, 255]),
    "min_samples_leaf": tune.choice([1, 2, 4, 8, 16, 32, 64]),
}

_lightgbm_search_space = {
    "learning_rate": tune.choice([0.1, 0.3, 1.0, 2.0]),
    "num_iterations": tune.choice([64, 128, 256, 512, ]),
    "reg_lambda": tune.choice([0., 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.]),
    "min_child_samples": tune.choice([1, 2, 4, 8, 16, 32, 64]),
    "max_depth": tune.choice([-1, 2, 4, 8]),
    "colsample_bytree": tune.uniform(0.5, 1),
}

_wcs_search_space = {
    "C_domain": tune.choice([0.001, 0.01, 0.1, 1., 10., 100., 1000.]),
    "C_discrim": tune.choice([0.001, 0.01, 0.1, 1., 10., 100., 1000.]),

}

_xgb_search_space = {
    "max_depth": tune.randint(4, 8),
    "colsample_bytree": tune.uniform(0.5, 1),
    "colsample_bylevel": tune.uniform(0.5, 1),
    "learning_rate": tune.choice([0.1, 0.3, 1.0, 2.0]),
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

_resnet_search_space = {
    **_DEFAULT_NN_SEARCH_SPACE,
    "n_blocks": tune.randint(1, 8),  # TODO(jpgard): possibly expand to 16.
    # TODO(jpgard): use d_main and d_hidden via a hidden_factor; see
    #  https://arxiv.org/pdf/2106.11959.pdf Table 14.
    "d_main": tune.choice([64, 128, 256, 512, 1024]),
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
    "expgrad": _expgrad_search_space,
    "ft_transformer": _ft_transformer_search_space,
    "group_dro": _group_dro_search_space,
    "histgbm": _histgbm_search_space,
    "lightgbm": _lightgbm_search_space,
    "mlp": _DEFAULT_NN_SEARCH_SPACE,
    "resnet": _resnet_search_space,
    "wcs": _wcs_search_space,
    "xgb": _xgb_search_space,
}
