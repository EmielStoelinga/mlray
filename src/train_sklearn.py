import ray
from ray.data.dataset import Dataset
from ray.air.result import Result
from ray.train.sklearn import SklearnTrainer

from sklearn.tree import DecisionTreeRegressor


def load_data():
    train_dataset = ray.data.read_csv('./data/preprocessed/train/')
    test_dataset = ray.data.read_csv('./data/preprocessed/test/')

    return train_dataset, test_dataset


def train_sklearn(target: str, train_dataset: Dataset, test_dataset: Dataset) -> Result:
    trainer = SklearnTrainer(
        estimator=DecisionTreeRegressor(),
        label_column=target,
        datasets={"train": train_dataset, "valid": test_dataset},
        cv=5,
        scoring='r2'
    )
    result = trainer.fit()
    return result


def main():
    target = 'WACHTTIJD'
    train_dataset, test_dataset = load_data()
    result = train_sklearn(target, train_dataset, test_dataset)

    print(f"R2 validation score: {result.metrics['valid']['test_score']}")


if __name__ == '__main__':
    main()
