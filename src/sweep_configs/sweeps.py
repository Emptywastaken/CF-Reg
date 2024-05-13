sweep_configuration = {
    "method": "bayes",
    "name": "counterfactual_overfitting",
    "metric": {"goal": "maximize", "name": "p_x"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64, 128, 256]},
        "epochs": {"values": [500]},
        "lr": {"max": 0.1, "min": 0.00001},
    },
}
