import ray
from ray.data.dataset import Dataset
from ray.air.result import Result
from ray.train.lightgbm import LightGBMTrainer
from ray.air.config import ScalingConfig
from utils import get_wachttijden_categorizer


def load_data():
    train_dataset = ray.data.read_csv('./data/preprocessed/train/')
    test_dataset = ray.data.read_csv('./data/preprocessed/test/')

    return train_dataset, test_dataset


def train_lightgbm(target: str, train_dataset: Dataset, test_dataset: Dataset) -> Result:
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
        datasets={"train": train_dataset, "valid": test_dataset},
    )
    result = trainer.fit()
    return result


def main():
    target = 'WACHTTIJD'
    train_dataset, test_dataset = load_data()
    result = train_lightgbm(target, train_dataset, test_dataset)

    print(f"L2 validation score: {result.metrics['valid-l2']}")


if __name__ == '__main__':
    main()
