sweep_configuration_dynamic_alpha = {
    "method": "bayes",
    "name": "counterfactual_overfitting",
    "metric": {"goal": "maximize", "name": "validation/accuracy"},
    "parameters": {
        "batch_size": {"values": [32, 64, 128, 256]},
        "epochs": {"values": [1000]},
        "lr": {"min": 0.00001, "max": 0.001}, #{"values": [0.00001, 0.00002, 0.00005, 0.0001, 0.0002]},
        #"hidden_layers": {"values": [[30, 20, 5], [25, 15, 5], [20, 10, 5], [40, 20, 5]]},
        #"losses": {"values": ["normal"]},
        "radius": {"min": 0.0001, "max": 1.5},
        "samples": {"min": 100, "max": 1000}
        #"alpha": {"min": 0.01, "max": 1.0}
    },
}


sweep_configuration_normal_no_l2 = {
    "method": "bayes",
    "name": "counterfactual_overfitting",
    "metric": {"goal": "maximize", "name": "validation/accuracy"},
    "parameters": {
        "batch_size": {"values": [32, 64, 128, 256]},
        "epochs": {"values": [1000]},
        "lr": {"min": 0.00001, "max": 0.001}, #{"values": [0.00001, 0.00002, 0.00005, 0.0001, 0.0002]},
        #"hidden_layers": {"values": [[30, 20, 5], [25, 15, 5], [20, 10, 5], [40, 20, 5]]},
        #"losses": {"values": ["normal"]},
        "radius": {"values": [0.0]}, #{"min": 0.0001, "max": 1.5},
        "samples": {"values": [0]}, #{"min": 100, "max": 1000}
        #"alpha": {"min": 0.01, "max": 1.0}
        "l2": {"values": [0.0]}
    }
}

sweep_configuration_normal = {
    "method": "bayes",
    "name": "counterfactual_overfitting",
    "metric": {"goal": "maximize", "name": "validation/accuracy"},
    "parameters": {
        "batch_size": {"values": [32, 64, 128, 256]},
        "epochs": {"values": [1000]},
        "lr": {"min": 0.00001, "max": 0.001}, #{"values": [0.00001, 0.00002, 0.00005, 0.0001, 0.0002]},
        #"hidden_layers": {"values": [[30, 20, 5], [25, 15, 5], [20, 10, 5], [40, 20, 5]]},
        #"losses": {"values": ["normal"]},
        "radius": {"values": [0.0]}, #{"min": 0.0001, "max": 1.5},
        "samples": {"values": [0]}, #{"min": 100, "max": 1000}
        "alpha": {"min": 0.01, "max": 1.0}, 
        "l2": {"min": 0.001, "max": 0.01}
    }
}


nosweep = {
    "batch_size": 256,
    "epochs": 1500,
    "lr": 0.0001,
    "radius":  1.5,
    "samples": 1000,
    "alpha": 0.5,
    "optimizer": "adam",
    "l2": 0.0
}
