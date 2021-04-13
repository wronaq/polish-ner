"""Script to run an experiment."""
import argparse
import json
import importlib
import time
import logging

logging.basicConfig(level=logging.INFO)


import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def run_experiment(experiment_config):
    """
    Run a training experiment.

    Parameters
    ----------
    experiment_config (dict)
        Of the form
        {
            "dataset": {
                "name": "NerDataset",
                "train_path": "train.txt",
                "valid_path": "valid.txt",
                "test_path": "test.txt",
            },
            "model": "NerModel",
            "network": {
                "args": {
                    "architecture": "allegro/herbert-base-cased",
                    "freeze": true
                }
            },
            "train_args": {
                "epochs": 3
            }
        }
    """
    print(f"Running experiment with config {experiment_config}")

    # dataset
    datasets_module = importlib.import_module("polish_ner.datasets")
    dataset_class_ = getattr(datasets_module, experiment_config["dataset"]["name"])
    dataset_args = experiment_config.get("dataset_args", {})
    try:
        architecture = experiment_config["network"]["args"]["architecture"]
        dataset_args["architecture"] = architecture
    except KeyError:
        pass
    dataset_train = dataset_class_(
        path=experiment_config["dataset"]["train_path"], **dataset_args
    )
    dataset_valid = dataset_class_(
        path=experiment_config["dataset"]["valid_path"], **dataset_args
    )
    dataset_test = dataset_class_(
        path=experiment_config["dataset"]["test_path"], **dataset_args
    )
    logging.info("DATASETS LOADED")

    # dataloaders
    dataloader_class_ = getattr(datasets_module, "DataLoaders")
    dataloaders = dataloader_class_(dataset_train, dataset_valid, dataset_test)
    logging.info("DATALOADERS CREATED")

    # network
    networks_module = importlib.import_module("polish_ner.networks")
    network_class_ = getattr(networks_module, experiment_config["network"]["name"])
    network_args = experiment_config["network"].get("args", {})
    network = network_class_(**network_args)
    logging.info("NETWORK LOADED")

    # model
    models_module = importlib.import_module("polish_ner.models")
    model_class_ = getattr(models_module, experiment_config["model"])
    model_args = experiment_config.get("train_args", {})
    model = model_class_(network, experiment_config["device"], **model_args)
    logging.info("MODEL CREATED")

    t = time.monotonic()
    model.fit(dataloaders=dataloaders)
    duration = int(time.monotonic() - t)
    print(
        f"Training took {duration//86400} days "
        f"{duration % 86400 // 3600} hours "
        f"{duration % 86400 % 3600 // 60} minutes.\n"
    )

    _, score = model.evaluate(model._dataloaders.test_loader)
    print(f"Test evaluation (F1 score (micro) ): {score:.5f}")


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_config",
        type=str,
        help='Path to experiment JSON like: \'{"dataset": "NerDataset", "model": "NerModel", "network": "allegro/herbert-base-cased"}\'',
    )
    args = parser.parse_args()
    return args


def main():
    """Run experiment."""
    args = _parse_args()

    with open(args.experiment_config, "r") as f:
        config = f.read()
    experiment_config = json.loads(config)
    run_experiment(experiment_config)


if __name__ == "__main__":
    main()
