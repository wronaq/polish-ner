from flask import Flask, request, render_template
import importlib
import json


app = Flask(__name__)

# load components
with open("tasks/config.json", "r") as f:
    config = f.read()
    experiment_config = json.loads(config)

networks_module = importlib.import_module("polish_ner.networks")
network_class_ = getattr(networks_module, experiment_config["network"]["name"])
network_args = experiment_config["network"].get("network_args", {})
network = network_class_(**network_args)

models_module = importlib.import_module("polish_ner.models")
model_class_ = getattr(models_module, experiment_config["model"])
model = model_class_(network=network, device="cpu")

predictor_module = importlib.import_module("polish_ner.predictor")
predictor_class_ = getattr(predictor_module, "NerPredictor")
predictor = predictor_class_(
    model,
    "weights/NerModel_NerDataset_allegro-herbert-base-cased_2021-04-12_18:55_weights.pt",
)


# root endpoint
@app.route("/", methods=["GET", "POST"])
def index():
    preds = None
    if request.method == "POST":
        text = request.form["content"]
        preds = predictor.predict_ner_tags(text)
    return render_template("index.html", preds=preds)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
