from typing import Dict, List


SHUFFLE_SEEDS = list(range(10))

DATASET_TO_SIZES: Dict[str, List] = {
    "Kenya_maize": [None],
    "Brazil_coffee": [None],
    "United States of America_almond": [
        20,
        50,
        126,
        254,
        382,
        508,
        636,
        764,
        892,
        1020,
        1148,
        1316,
        None,
    ],
    "Togo_crop": [20, 50, 126, 254, 382, 508, 636, 764, 892, 1020, 1148, None],
}


# Model names
RANDOM_FOREST = "RF"
DL_RANDOM = "DL_RANDOM"
DL_PRETRAINED = "DL_PRETRAINED"
DL_MAML = "DL_MAML"


# LSTM model configurations
HIDDEN_VECTOR_SIZE = 128
NUM_CLASSIFICATION_LAYERS = 2
CLASSIFIER_DROPOUT = 0.2
CLASSIFIER_BASE_LAYERS = 1

# Pretrained LSTM configurations
PRETRAIN_VAL_RATIO = 0.1
