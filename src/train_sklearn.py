import ray
from ray.data.dataset import Dataset
from ray.air.result import Result
from ray.train.sklearn import SklearnTrainer
from ray.data.preprocessors import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from utils import load_preprocessed_data


def train_sklearn(target: str, train_dataset: Dataset, valid_dataset: Dataset) -> Result:
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
        datasets={"train": train_dataset, "valid": valid_dataset},
        cv=5,
        scoring='r2'
    )
    result = trainer.fit()
    return result


def main():
    target = 'WACHTTIJD'
    train_dataset, valid_dataset, _ = load_preprocessed_data()
    result = train_sklearn(target, train_dataset, valid_dataset)

    print(f"R2 score: {result.metrics['valid']['test_score']}")


if __name__ == '__main__':
    main()
