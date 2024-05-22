sweep_configuration = {
    "method": "bayes",
    "name": "counterfactual_overfitting",
    "metric": {"goal": "minimize", "name": "train/loss"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64, 128, 256]},
        "epochs": {"values": [500]},
        "lr": {"values": [0.00001, 0.00002, 0.00005, 0.0001, 0.0002]},
        #"hidden_layers": {"values": [[30, 20, 5], [25, 15, 5], [20, 10, 5], [40, 20, 5]]},
        #"losses": {"values": ["normal"]},
        "radius": {"min": 0.0001, "max": 1.5},
        "samples": {"min": 100, "max": 10000},
        "alpha": {"min": 0.01, "max": 1.0}
    },
}


nosweep = {
    "batch_size": 256,
    "epochs": 500,
    "lr": 0.0001,
    "radius":  1.5,
    "samples": 100,
    "alpha": 0.1,
    "optimizer": "adam",
    "l2": 0.1
}
