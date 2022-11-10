import ray
from ray.data.dataset import Dataset
from ray.air.result import Result
from ray.train.lightgbm import LightGBMTrainer
from ray.air.config import ScalingConfig
import tempfile
import pandas as pd
import re
import os


def alter_columnnames(dir):
    df = pd.read_csv(os.path.join(dir, os.listdir(dir)[0]))
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = os.path.join(tmpdir, 'data.csv')
        df.to_csv(tmppath, index=False)
        dataset = ray.data.read_csv(tmppath)

    return dataset


def load_data():
    # Because the columnnames of the data contain characters that LightGBM cannot handle,
    # columnsnames are altered for now using pandas. I have not found a way to do this
    # easily with Ray.
    train_dataset = alter_columnnames('./data/preprocessed/train/')
    test_dataset = alter_columnnames('./data/preprocessed/test/')

    return train_dataset, test_dataset


def train_sklearn(target: str, train_dataset: Dataset, test_dataset: Dataset) -> Result:
    trainer = LightGBMTrainer(
        scaling_config=ScalingConfig(
            # Whether to use GPU acceleration.
            use_gpu=False,
        ),
        label_column=target,
        num_boost_round=20,
        params={
            # LightGBM specific params
            "objective": "regression",
            "metric": "mean_squared_error"
        },
        datasets={"train": train_dataset, "valid": test_dataset},
    )
    result = trainer.fit()
    return result


def main():
    target = 'WACHTTIJD'
    train_dataset, test_dataset = load_data()
    result = train_sklearn(target, train_dataset, test_dataset)

    print(f"L2 validation score: {result.metrics['valid-l2']}")


if __name__ == '__main__':
    main()
