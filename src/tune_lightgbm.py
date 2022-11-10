import ray
from ray import tune
from ray.data.dataset import Dataset
from ray.air.result import Result
from ray.air import RunConfig, ScalingConfig
from ray.train.lightgbm import LightGBMTrainer
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner
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


def tune_lightgbm(target: str, train_dataset: Dataset, test_dataset: Dataset) -> Result:
    trainer = LightGBMTrainer(
        scaling_config=ScalingConfig(
            # Whether to use GPU acceleration.
            use_gpu=False,
        ),
        label_column=target,
        num_boost_round=20,
        params={
            "objective": "regression"
        },
        datasets={"train": train_dataset, "valid": test_dataset},
    )

    tuner = Tuner(
        trainer,
        run_config=RunConfig(verbose=1),
        param_space={
            "params": {
                "num_leaves": tune.randint(25, 63), 
            },
        },
        # max_concurrent_trials > 2 resulted in resource issues on local machine. This could be
        # incremented when running on a cluster. 
        tune_config=TuneConfig(num_samples=8, metric="valid-l2", mode="min", max_concurrent_trials=2),
    )
    result = tuner.fit()
    best_result = result.get_best_result()
    return best_result


def main():
    target = 'WACHTTIJD'
    train_dataset, test_dataset = load_data()
    result = tune_lightgbm(target, train_dataset, test_dataset)

    print(f"L2 validation score best model = {result.metrics['valid-l2']} with num_leaves = {result.metrics['config']['params']['num_leaves']}.")


if __name__ == '__main__':
    main()
