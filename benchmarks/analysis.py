import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Optional, List


def load_combined_results(
    root, model_name: str, dataset_target: str, num_samples: Optional[int] = None
) -> DefaultDict:

    results_folder = Path(root) / model_name

    json_files = results_folder.glob(f"combined_{dataset_target}_{num_samples}_*.json")

    output_dict: DefaultDict[str, List] = defaultdict(list)
    for filepath in json_files:

        with filepath.open("r") as f:
            results = json.load(f)
        for key, val in results.items():
            output_dict[key].append(val)

    return output_dict
