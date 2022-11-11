import ray
from ray.train.lightgbm import LightGBMPredictor
from ray import serve
from ray.serve import PredictorDeployment
from ray.serve.http_adapters import pandas_read_json
import requests
from utils import load_preprocessed_data

def serve_lightgbm(checkpoint):
    serve.run(
        PredictorDeployment.options(name="LightGBMService").bind(
            LightGBMPredictor, checkpoint, http_adapter=pandas_read_json
        )
    )


def do_inference():
    _, _, test_dataset = load_preprocessed_data()

    # take first 5 rows of test dataset
    sample_input = test_dataset.take(5)
    sample_input = [dict(sample_input[n]) for n in range(len(sample_input))]

    output = requests.post("http://localhost:8000/", json=sample_input).json()
    print(output)


def main():
    model_checkpoint = '<checkpoint-uri>'
    serve_lightgbm(model_checkpoint)

    do_inference()


if __name__ == '__main__':
    main()
