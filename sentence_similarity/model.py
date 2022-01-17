from typing import Type

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Metric, SpearmanCorrcoef


class SimilarityLitModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = nn.MSELoss(),
        metric: Type[Metric] = SpearmanCorrcoef,
        metric_name: str = "spearman",
    ) -> None:
        super().__init__()
        # initialize model
        self.model = model
        self.forward = self.model.forward
        self.criterion = criterion
        self.train_metric = metric()
        self.val_metric = metric()
        self.metric_name = metric_name

    def training_step(self, batch, batch_idx):
        features1, features2, y_hat = batch
        # run model
        y = self.model(features1, features2)
        # compute loss
        loss = self.criterion(y, y_hat)
        # compute metrics
        metric = self.train_metric(y, y_hat)
        self.log_dict({"train_loss": loss, f"train_{self.metric_name}": metric})
        return loss
    
    def validation_step(self, batch, batch_idx):
        features1, features2, y_hat = batch
        # run model
        y = self.model(features1, features2)
        # compute loss
        loss = self.criterion(y, y_hat)
        # compute metrics
        metric = self.val_metric(y, y_hat)
        self.log_dict({"train_loss": loss, f"val_{self.metric_name}": metric}, prog_bar=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0.01)
