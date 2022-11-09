import ray
from ray.data.preprocessors import OneHotEncoder
from pyarrow.csv import ParseOptions, ConvertOptions


def prepare_data(include_columns: list):
    # read data
    data_path = './data/raw/Dataset Wachttijden medisch-specialistische zorg 1 november 2022.csv'
    dataset = ray.data.read_csv(data_path, convert_options=ConvertOptions(
        include_columns=include_columns), parse_options=ParseOptions(delimiter=';'))

    # drop missing values
    dataset = dataset.filter(lambda row: row['WACHTTIJD'] is not None)

    return dataset


def main():
    # read data
    features = [
        'TYPE_WACHTTIJD',
        'SPECIALISME',
        'ROAZ_REGIO',
        'TYPE_ZORGINSTELLING'
    ]
    target = 'WACHTTIJD'
    dataset = prepare_data(features + [target])

    # preprocess data
    preprocessor = OneHotEncoder(features)
    dataset_preprocessed = preprocessor.fit_transform(dataset)

    # split data into train and validation.
    train_dataset, test_dataset = dataset_preprocessed.train_test_split(
        test_size=0.2)

    # write preprocessed data to csv
    train_dataset.write_csv('./data/preprocessed/train/')
    test_dataset.write_csv('./data/preprocessed/test/')


if __name__ == '__main__':
    main()
