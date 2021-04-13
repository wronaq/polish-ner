"""Model class, to be extended by specific types of models."""
from pathlib import Path
import torch
import time
from .early_stopping import EarlyStopping
from importlib import import_module


class Model:
    """Base class, to be subclassed by predictors for specific type of data."""

    def __init__(self, network, device="cpu", **kwargs):

        self.device = device
        self.network = network.to(self.device)
        self._dataloaders = None
        self.name = ""
        self.kwargs = kwargs

        self._early_stopping = EarlyStopping(**self.kwargs.get("early_stopping", {}))

    @property
    def weights_filename(self):
        p = Path(__file__).resolve().parents[2] / "weights"
        p.mkdir(parents=True, exist_ok=True)
        return str(p / f"{self.name}_weights.pt")

    def fit(self, dataloaders, testing=False):

        if not testing:
            try:
                epochs = self.kwargs["epochs"]
            except KeyError:
                epochs = 5
        else:
            epochs = 50

        self._dataloaders = dataloaders
        self.name = (
            f"{self.__class__.__name__}_{self._dataloaders._datasets.get('train').__class__.__name__}_"
            f"{self.network.architecture.replace('/', '-')}_{time.strftime('%Y-%m-%d_%H:%M', time.gmtime())}"
        )

        criterion = self.criterion()
        train_loader = (
            self._dataloaders.train_loader
            if not testing
            else self._dataloaders._testing_batch
        )

        for epoch in range(epochs):
            self.network.train()
            running_loss = 0.0
            running_f1 = 0.0
            for i, batch in enumerate(train_loader):
                # forward and backward propagation
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["target"].to(self.device)
                outputs = self.network(input_ids, attention_mask).transpose(1, 2)
                loss = criterion(outputs, targets)
                self.optimizer().zero_grad()
                loss.backward()
                self.optimizer().step()

                # save results
                running_loss += loss.item()
                running_f1 += self.f1_micro(targets, outputs)
                if i > 0 and i % 50 == 0:
                    stats = (
                        f"Epoch: {epoch+1}/{epochs}, batch: {i}/{len(train_loader)}, "
                        f"train_loss: {running_loss/i:.5f}, train_f1_score: {running_f1/i:.5f}"
                    )
                    print(stats, flush=True)
                    with open("stats.log", "a") as f:
                        print(stats, file=f)
            if testing:
                continue
            # calculate loss and accuracy on validation dataset
            with torch.no_grad():
                val_loss, val_f1 = self.evaluate(self._dataloaders.valid_loader)
            stats = (
                f"Epoch: {epoch+1}/{epochs}, "
                f"train_loss: {running_loss/i:.5f}, train_f1_score: {running_f1/i:.5f}, "
                f"valid_loss: {val_loss:.5f}, valid_f1_score: {val_f1:.5f}"
            )
            print(stats)
            with open("stats.log", "a") as f:
                print(stats, file=f)

            # save after each epoch
            self.save_weights()

            # check for early stopping
            self._early_stopping(val_loss, self.network)
            if self._early_stopping.early_stop:
                print("Early stopping.")
                break

        if testing:
            return running_loss  # , running_cs
        else:
            self.load_weights(path_to_weights=self._early_stopping.path)
            self.save_weights()
            print("\nFinished training")

    def parametrize(self, obj, def_obj, def_params):
        try:
            kwargs = self.kwargs[obj]
            string_obj = kwargs.get("object", def_obj)
            module_path, class_name = string_obj.rsplit(".", 1)
            module = import_module(module_path)
            obj = getattr(module, class_name)
            params = kwargs.get("params", {})
        except KeyError:
            module_path, class_name = def_obj.rsplit(".", 1)
            module = import_module(module_path)
            obj = getattr(module, class_name)
            params = def_params
        finally:
            return obj, params

    def criterion(self):
        criterion, params = self.parametrize(
            "criterion", "torch.nn.NLLLoss", {"reduction": "mean"}
        )
        return criterion(**params).to(self.device)

    def optimizer(self):
        optimizer, params = self.parametrize(
            "optimizer", "torch.optim.AdamW", {"lr": 3e-4}
        )
        return optimizer(self.network.parameters(), **params)

    def load_weights(self, path_to_weights):
        self.network.load_state_dict(torch.load(path_to_weights))

    def save_weights(self):
        torch.save(self.network.state_dict(), self.weights_filename)

    def evaluate(self, dataloader):
        criterion = self.criterion()
        self.network.eval()
        valid_loss = 0.0
        valid_f1 = 0.0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["target"].to(self.device)
            outputs = self.network(input_ids, attention_mask).transpose(1, 2)
            loss = criterion(outputs, targets)
            # results
            valid_loss += loss.item()
            valid_f1 += self.f1_micro(targets, outputs)

        return valid_loss / len(dataloader), valid_f1 / len(dataloader)

    @staticmethod
    def f1_micro(true, pred):
        predicted_tag = pred.argmax(dim=1)
        true_positive = torch.eq(true, predicted_tag).sum().float()
        f1_score = torch.div(true_positive, true.numel())
        return f1_score
