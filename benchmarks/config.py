from typing import Dict, List


SHUFFLE_SEEDS = list(range(10))

DATASET_TO_SIZES: Dict[str, List] = {
    "Kenya_1_maize": [None],
    "Brazil_0_coffee": [None],
    "Togo_crop": [20, 50, 126, 254, 382, 508, 636, 764, 892, 1020, 1148, None],
}


# Model names
RANDOM_FOREST = "RF_GRID_SEARCH"
DL_RANDOM = "DL_RANDOM"
DL_PRETRAINED = "DL_PRETRAINED"
DL_MAML = "DL_MAML"


# LSTM model configurations
HIDDEN_VECTOR_SIZE = 128
NUM_CLASSIFICATION_LAYERS = 2
CLASSIFIER_DROPOUT = 0.2
CLASSIFIER_BASE_LAYERS = 1
