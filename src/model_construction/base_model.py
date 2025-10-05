from typing import Any
from eegdash.api import BaseConcatDataset
import copy
import torch
import tqdm
import consts.model as cm
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import AdamW, Optimizer
from braindecode.models import EEGNeX

class BaseModel:
    def __init__(self):
        self.__device: str = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._train_set: BaseConcatDataset
        self._valid_set: BaseConcatDataset
        self._test_set:  BaseConcatDataset
        self._train_loader: DataLoader
        self._valid_loader: DataLoader
        self._test_loader:  DataLoader
        self._model: Module
        self._loss_fn: Module
        self._optimizer: Optimizer
        self._epoch: int = 10
        self._n_batches: int = 10
        self._print_batch_stats: bool = True
        self._scheduler = LRScheduler

        self._best_rmse: float = float("inf");
        self._best_state: Any = None
        self._best_epoch: int | None = None
        self._epoch_no_improv: int = 0

    def trainOneEpoch(self):
        total_loss: float = 0.0
        sum_sq_err: float = 0.0
        n_samples: int = 0

        progress_bar = tqdm.tqdm(
            enumerate(self._train_loader),
            total=len(self._train_loader),
            disable=not self._print_batch_stats
        )

        for batch_idx, batch in progress_bar:
            X, y = batch[0], batch[1]
            X, y = X.to(self.__device).float(), y.to(self.__device).float()

            self._optimizer.zero_grad(set_to_none=True)

            preds = self._model(X)
            loss = self._loss_fn(preds, y)
            loss.backward()
            self._optimizer.step()
            total_loss += loss.item()

            preds_flat = preds.detach().view(-1)
            y_flat = y.detach().view(-1)
            sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
            n_samples += y_flat.numel()

            if self._print_batch_stats:
                running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
                progress_bar.set_description(
                    f"Epoch {self._epoch}, Batch {batch_idx + 1}/{len(self._train_loader)}, "
                    f"Loss: {loss.item():.6f}, RMSE: {running_rmse:.6f}"
                )

        avg_loss = total_loss / len(self._train_loader)
        rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
        return avg_loss, rmse

    def validateModel(self):
        self._model.eval()

        total_loss: float = 0.0
        sum_sqr_err: float = 0.0
        n_samples = 0

        iterator = tqdm.tqdm(
            enumerate(self._test_loader),
            total=self._n_batches,
            disable=not self._print_batch_stats
        )

        for batch_idx, batch in iterator:
            X, y = batch[0], batch[1]
            X, y = X.to(self.__device).float(), y.to(self.__device).float()

            preds = self._model(X)
            batch_loss = self._loss_fn(preds, y).item()
            total_loss += batch_loss

            preds_flat = preds.detach().view(-1)
            y_flat = y.detach().view(-1)
            sum_sqr_err += torch.sum((preds_flat - y_flat) ** 2).item()
            n_samples += y_flat.numel()

            if self._print_batch_stats:
                running_rmse = (sum_sqr_err / max(n_samples, 1)) ** 0.5
                iterator.set_description(
                    f"Val Batch {batch_idx + 1}/{self._n_batches}, "
                        f"Loss: {batch_loss:.6f}, RMSE: {running_rmse:.6f}"
                )

        avg_loss = (total_loss / self._n_batches
            if self._n_batches
            else float("nan")
        )
        rmse = (sum_sqr_err / max(n_samples, 1)) ** 0.5

        print(f"Val RMSE: {rmse:.6f}, Val Loss: {avg_loss:.6f}\n")
        return avg_loss, rmse

    def trainModel(self):
        for epoch in range(1, self._epoch + 1):
            print(f"Epoch {epoch}/{self._epoch}")
            train_loss, train_rmse = self.trainOneEpoch()
            val_loss, val_rmse = self.validateModel()

            print(
                f"Train RMSE: {train_rmse:.6f}, "
                f"Average Train Loss: {train_loss:.6f}, "
                f"Val RMSE: {val_rmse:.6f}, "
                f"Average Val Loss: {val_loss:.6f}"
            )

            if val_rmse < self._best_rmse - cm.min_delta:
                self._best_rmse = val_rmse
                self._best_state = copy.deepcopy(self._model.state_dict())
                self._best_epoch = epoch
            else:
                self._epoch_no_improv += 1
                if self._epoch_no_improv >= cm.patience:
                    print(
                        f"Early stoping at epoch {epoch}"
                        f"Best Val RMSE: {self._best_rmse:.6f}"
                        f"Best Epoch: {self._best_epoch}"
                    )
                    break

    def saveModel(self, name: str = "weights_challenge_1"):
        if self._best_state is None:
            self._model.load_state_dict(self._best_state)

        torch.save(
            self._model.state_dict(), 
            name + ".pt"
        )
        print(f"Model saves as {name}.pt")
        pass

class BaseModelBuilder:
    def __init__(self):
        self.__model: BaseModel = BaseModel()

    def setTrainSet(self, train_set: BaseConcatDataset):
        self.__model._train_set = train_set
        return self

    def setValidationSet(self, valid_set: BaseConcatDataset):
        self.__model._valid_set = valid_set
        return self

    def setTestSet(self, test_set: BaseConcatDataset):
        self.__model._test_set = test_set
        return self

    def setModel(self, model: Module):
        self.__model._model = model
        return self

    def setLossFn(self, loss_fn: Module):
        self.__model._loss_fn = loss_fn
        return self

    def setEpoch(self, epoch: int):
        self.__model._epoch = epoch
        return self

    def setOptimizer(self, optimizer: Optimizer):
        self.__model._optimizer = optimizer
        return self

    def shouldPrintStats(self, opt: bool):
        self.__model._print_batch_stats = opt
        return self

    def buildModel(self) -> BaseModel:
        self.__model._train_loader = (
            DataLoader(self.__model._train_set)
        )
        self.__model._valid_loader = (
            DataLoader(self.__model._valid_set)
        )
        self.__model._test_loader  = (
            DataLoader(self.__model._test_set)
        )
        return self.__model



DefaultModel = EEGNeX(
    n_chans=129,
    n_outputs=129,
    n_times=200,
    sfreq=100
)

DefaultOptimzer = AdamW(
    DefaultModel.parameters(),
    lr=cm.lr,
    weight_decay=cm.weight_decay
)
