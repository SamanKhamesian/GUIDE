import sys

from ablation_study.ablation_common import CARBOHYDRATE_ONLY, run_ablation
from config import DataConfig
from utils import set_seed


if __name__ == "__main__":
    patient_id = sys.argv[1]
    seed = int(sys.argv[2])

    set_seed(seed)
    run_ablation(case=CARBOHYDRATE_ONLY, dataset_name=DataConfig.DATASET, patient_id=patient_id, seed=seed, )
