import ray
import dask.dataframe as dd
from ray.util.dask import enable_dask_on_ray
from ray.data.preprocessors import OneHotEncoder


def load_data(filename):
    df = dd.read_csv(filename, encoding='iso-8859-1', sep=';', dtype={'WACHTTIJD': 'float64'})
    return df


def select_features(df: dd.DataFrame) -> dd.DataFrame:
    feature_names = [
        'TYPE_WACHTTIJD',
        'SPECIALISME',
        'ROAZ_REGIO',
        'TYPE_ZORGINSTELLING',
        'WACHTTIJD'
    ]

    return df[feature_names]



def fill_missing_values(df: dd.DataFrame) -> dd.DataFrame:
    df['TYPE_ZORGINSTELLING'] = df['TYPE_ZORGINSTELLING'].fillna('Kliniek')
    return df


def drop_invalid_records(df: dd.DataFrame) -> dd.DataFrame:
    return df.dropna(subset=['WACHTTIJD'])


def main():
    ray.init()

    enable_dask_on_ray()

    dataset_preprocessed = load_data('data/wachttijden.csv')
    dataset_preprocessed = select_features(dataset_preprocessed)
    dataset_preprocessed = fill_missing_values(dataset_preprocessed)
    dataset_preprocessed = drop_invalid_records(dataset_preprocessed)

    final_dataset = ray.data.from_dask(dataset_preprocessed)

    # split data into train and test sets.
    train_dataset, test_dataset = final_dataset.train_test_split(test_size=0.2)

    # write preprocessed data to csv
    train_dataset.write_csv('./data/preprocessed/train/')
    test_dataset.write_csv('./data/preprocessed/test/')

    ray.shutdown()

if __name__ == '__main__':
    main()
