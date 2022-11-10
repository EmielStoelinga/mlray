# MLRayDemo

## Ray
In this demo, [Ray](https://www.ray.io/) is used. Ray exists of different components, these are:
- Ray Core: foundation that enables scalable, distributed Python
- Ray Data: provide distributed data transformations
- Ray Train: enables distributed deep learning
- Ray Tune: hyperparameter tuning at scale
- Ray Serve: serve and scale ML models or ML pipelines
- Ray RLLib: reinforcement learning built on Ray

## Setup environment
Initialize and activate a conda environment using the following command:
```shell
conda create -n mlraydemo python=3.7
conda activate mlraydemo
```

Then install the required dependencies with the following command:
```shell
pip install -r requirements.txt
```

## Training with wachttijden data
### Data preparation
Download the data from [this website](https://puc.overheid.nl/PUC/Handlers/DownloadDocument.ashx?identifier=PUC_656543_22&versienummer=1) and paste it in the `./data/raw/` folder.
Data can be cleaned and pre-processed by running the `./src/prepare_dataset.py` Python script. In this script row without target values are removed and the features are one-hot encoded.

### SKLearn model training
A model can be trained by running the `./src/train_sklearn.py` script. First the data is loaded, then a decision tree is trained with cross validation (n=5) to predict the target value. A resulting L2 validation score is printed.

### LightGBM model training
A model can be trained by running the `./src/train_lightgbm.py` script. First the data is loaded in which columnnaes are altered slightly because LightGBM could not handle special characters in the columnnames well. Then a LightGBM model is trained to predict the target value. A resulting L2 validation score is printed.

## Tuning with wachttijden data
A LightGBM model can be tuned by running the `./src/tune_lightgbm.py` script. In the script, 8 runs with 8 different values for the paramter `num_leaves` are run of which the validation metrics of the best model are printed. 

## Feedback
- It is sometimes hard to perform elementary alterations such as changing columnnames. You are then limited to the functionalities of the ray.data.Dataset class and functions in ray.data which seem less mature than for example pandas.
- Once data has been fitted in a ray.data.Dataset object, it is relatively easy to change between different models due to the wrappers in ray.train.