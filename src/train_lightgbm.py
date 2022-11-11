import ray
from ray.data.dataset import Dataset
from ray.air.result import Result
from ray.train.lightgbm import LightGBMTrainer
from ray.air.config import ScalingConfig
from utils import get_wachttijden_categorizer, load_preprocessed_data


def train_lightgbm(target: str, train_dataset: Dataset, valid_dataset: Dataset) -> Result:
    trainer = LightGBMTrainer(
        preprocessor=get_wachttijden_categorizer(),
        scaling_config=ScalingConfig(
            # Whether to use GPU acceleration.
            use_gpu=False,
        ),
        label_column=target,
        num_boost_round=20,
        params={
            # LightGBM specific params
            "objective": "regression",
            "num_leaves": 31
        },
        datasets={"train": train_dataset, "valid": valid_dataset},
    )
    result = trainer.fit()
    return result


def main():
    target = 'WACHTTIJD'
    train_dataset, valid_dataset, _ = load_preprocessed_data()
    result = train_lightgbm(target, train_dataset, valid_dataset)

    print(f"L2 validation score: {result.metrics['valid-l2']}")


if __name__ == '__main__':
    main()
