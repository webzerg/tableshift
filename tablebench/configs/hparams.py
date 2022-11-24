from frozendict import frozendict

from ray import tune

_DEFAULT_NN_SEARCH_SPACE = frozendict({
    "d_hidden": tune.choice([64, 128, 256, 512]),

    # Samples a float uniformly between 0.0001 and 0.1, while
    # sampling in log space and rounding to multiples of 0.00005
    "lr": tune.qloguniform(1e-4, 1e-1, 5e-5),

    "n_epochs": tune.randint(1, 2),
    "num_layers": tune.randint(1, 4),
    "weight_decay": tune.quniform(0., 1., 0.1),
})

_xgb_search_space = {
    "max_depth": tune.randint(4, 8),
    "colsample_bytree": tune.uniform(0.5, 1),
    "colsample_bylevel": tune.uniform(0.5, 1),
    "learning_rate": tune.choice([0.1, 0.3, 1.0, 2.0]),
    "max_bin": tune.choice([128, 256, 512])
}

_lightgbm_search_space = {
    "learning_rate": tune.choice([0.1, 0.3, 1.0, 2.0]),
    "n_estimators": tune.choice([64, 128, 256, 512, ]),
    "reg_lambda": tune.choice([0., 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.]),
    "min_child_samples": tune.choice([1, 2, 4, 8, 16, 32, 64]),
    "max_depth": tune.choice([-1, 2, 4, 8]),
    "colsample_bytree": tune.uniform(0.5, 1),
}

search_space = frozendict({
    # TODO(jpgard): update these with params specific to each model.
    "mlp": _DEFAULT_NN_SEARCH_SPACE,
    "ft_transformer": _DEFAULT_NN_SEARCH_SPACE,
    "lightgbm": _lightgbm_search_space,
    "resnet": _DEFAULT_NN_SEARCH_SPACE,
    "group_dro": _DEFAULT_NN_SEARCH_SPACE,
    "xgb": _xgb_search_space,
})
