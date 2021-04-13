"""Predictor class"""
import importlib
import argparse
import json


class NerPredictor:
    """Returns text with ner tags"""

    def __init__(self, model, path_to_weights):
        self.model = model
        self.model.load_weights(path_to_weights)

    def predict_ner_tags(self, input_text):
        """Predict on text"""
        output = self.model.predict_on_text(input_text)
        print(self._format(output))
        return output

    def _format(self, output):
        text_with_tags = ""
        for el in output:
            if el[1] == "O":
                text_with_tags += f"{el[0]} "
            else:
                text_with_tags += f"\033[91m{el[0]} ({el[1]})\033[0m "
        return text_with_tags


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_config",
        type=str,
        help="Path to experiment JSON used for training",
    )
    parser.add_argument("weights", type=str, help="Path to file with weights")
    parser.add_argument(
        "text",
        nargs="+",
        type=str,
        help="Text to predict",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Whether to use GPU",
    )
    args = parser.parse_args()

    args.device = "cuda" if args.gpu else "cpu"

    return args


def main():

    args = _parse_args()
    with open(args.experiment_config, "r") as f:
        config = f.read()
    experiment_config = json.loads(config)

    networks_module = importlib.import_module("polish_ner.networks")
    network_class_ = getattr(networks_module, experiment_config["network"]["name"])
    network_args = experiment_config["network"].get("network_args", {})
    network = network_class_(**network_args)

    models_module = importlib.import_module("polish_ner.models")
    model_class_ = getattr(models_module, experiment_config["model"])
    model = model_class_(network=network, device=args.device)

    predictor = NerPredictor(model, args.weights)
    predictor.predict_ner_tags(" ".join(args.text))


if __name__ == "__main__":
    main()
