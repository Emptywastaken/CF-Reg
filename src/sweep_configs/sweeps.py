sweep_configuration = {
    "method": "grid",
    "name": "counterfactual_overfitting",
    "metric": {"goal": "maximize", "name": "p_x"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64, 128, 256]},
        "epochs": {"values": [500, 1000, 1500]},
        "lr": {"values": [0.00001, 0.00002, 0.00005, 0.0001, 0.0002]},
        "hidden_layers": {"values": [[30, 20, 5], [25, 15, 5], [20, 10, 5], [40, 20, 5]]},
        "losses": {"values": ["regularized", "normal"]}
    },
}
