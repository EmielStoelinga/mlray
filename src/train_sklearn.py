import ray
from ray.data.dataset import Dataset
from ray.air.result import Result
from ray.train.sklearn import SklearnTrainer
from ray.data.preprocessors import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


def load_data():
    train_dataset = ray.data.read_csv('./data/preprocessed/train/')
    test_dataset = ray.data.read_csv('./data/preprocessed/test/')

    return train_dataset, test_dataset


def train_sklearn(target: str, train_dataset: Dataset, test_dataset: Dataset) -> Result:
    preprocessor = OneHotEncoder(columns=[
        'TYPE_WACHTTIJD',
        'SPECIALISME',
        'ROAZ_REGIO',
        'TYPE_ZORGINSTELLING',
    ])

    trainer = SklearnTrainer(
        preprocessor=preprocessor,
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

    print(f"R2 score: {result.metrics['valid']['test_score']}")


if __name__ == '__main__':
    main()
