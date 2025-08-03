"""
Configuration file for THItoGene training
"""

# Dataset configurations
GENE_LISTS = {
    "3CA": {
        "n_genes": 2977,
        "description": "3CA gene set"
    },
    "HER2ST": {
        "n_genes": 785,
        "description": "HER2ST gene set"
    },
    "CSCC": {
        "n_genes": 134,
        "description": "CSCC gene set"
    },
    "Hallmark": {
        "n_genes": 4376,
        "description": "Hallmark gene set"
    }
}

# Model configurations
MODEL_CONFIGS = {
    "default": {
        "learning_rate": 1e-5,
        "route_dim": 64,
        "caps": 20,
        "heads": [16, 8],
        "n_layers": 4
    },
    "small": {
        "learning_rate": 1e-4,
        "route_dim": 32,
        "caps": 10,
        "heads": [8, 4],
        "n_layers": 2
    },
    "large": {
        "learning_rate": 5e-6,
        "route_dim": 128,
        "caps": 40,
        "heads": [32, 16],
        "n_layers": 6
    }
}

# Training configurations
TRAINING_CONFIGS = {
    "quick_test": {
        "epochs": 10,
        "batch_size": 1,
        "patience": 5,
        "check_val_every_n_epoch": 1
    },
    "normal": {
        "epochs": 300,
        "batch_size": 1,
        "patience": 25,
        "check_val_every_n_epoch": 1
    },
    "long": {
        "epochs": 500,
        "batch_size": 1,
        "patience": 50,
        "check_val_every_n_epoch": 1
    }
}

# Paths
PATHS = {
    "models": "./models",
    "logs": "./logs",
    "data": "./data"
}
