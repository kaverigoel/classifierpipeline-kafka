from typing import Any, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from dataloader.dataloader import ClassificationDataset
from torch.utils.data import DataLoader
# from torchvision import transforms
import wandb
from torch.nn import functional as F
import torchvision.models as models
from metrics.metrics import ClassificationMetrics


class ResNet(pl.LightningModule):
    def __init__(self, params: Any):
        super(ResNet, self).__init__()
        self._model = models.resnet18(pretrained=True)
        self._model.fc = nn.Linear(512, params.out_channels)

        for param in self._model.parameters():
            param.requires_grad = True

        for param in self._model.fc.parameters():
            param.requires_grad = True
        self.params = params

    def forward(self, x: torch.Tensor):
        x = self._model(x)
        x = F.sigmoid(x)
        return x

    def loss_fn(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        Loss function for the classifier
        """
        # y = y.unsqueeze(1)
        y = F.one_hot(y, num_classes=self.params.out_channels)
        # print(y_pred.shape, y.shape)
        if self.params.loss == 'binary_cross_entropy':
            loss = nn.BCELoss()(y_pred.to(torch.float32), y.to(torch.float32))
        elif self.params.loss == 'cross_entropy':
            loss = nn.CrossEntropyLoss()(y_pred.to(torch.float32), y.to(torch.float32))
        else:
            raise NotImplementedError('Currently does not support this loss')
        return loss

    def configure_optimizers(self) -> torch.optim:
        opt = torch.optim.Adam(self._model.parameters(), lr=1e-4)
        return opt

    def train_dataloader(self) -> DataLoader:
        dataset = ClassificationDataset(
            data_dir=self.params.data_dir, mode='train', split=self.params.val_split, aug=self.params.aug)
        trainloader = DataLoader(dataset, batch_size=self.params.batch_size,
                                 shuffle=True, num_workers=self.params.num_workers)
        return trainloader

    def training_step(self, train_batch: list, batch_idx: int) -> dict:
        x, y = train_batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        logs = {
            'train_loss': loss
        }

        if batch_idx % 20 == 0:
            wandb.log(logs)
        self.log('train_loss', loss)
        return {'loss': loss}

    def val_dataloader(self) -> DataLoader:
        dataset = ClassificationDataset(
            data_dir=self.params.data_dir, mode="val", split=self.params.val_split, aug=self.params.aug)
        valloader = DataLoader(dataset, batch_size=self.params.batch_size,
                               shuffle=False, num_workers=self.params.num_workers)
        return valloader

    def validation_step(self, val_batch: list, batch_idx: int) -> dict:
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs: list) -> None:
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {
            'val_loss': val_loss
        }
        wandb.log(logs)

    def test_dataloader(self) -> DataLoader:
        dataset = ClassificationDataset(
            data_dir=self.params.data_dir, mode='test', split=self.params.val_split, aug=self.params.aug)
        testloader = DataLoader(dataset, batch_size=self.params.batch_size,
                                shuffle=False, num_workers=self.params.num_workers)
        return testloader

    def test_step(self, test_batch: list, batch_idx: torch.Tensor) -> dict:
        """
        Backward step of the model for testing
        """
        x, y = test_batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)

        # Calculate metrics
        y_pred_formatted = torch.argmax(y_pred, dim=1)
        metricObj = ClassificationMetrics(y_pred_formatted, y)
        acc = metricObj.acc

        return {
            'test_loss': loss,
            'test_acc': acc

        }

    def test_epoch_end(self, outputs: list) -> None:
        test_loss = torch.stack(
            [x['test_loss'] for x in outputs]).mean()
        test_acc = torch.stack([x['test_acc']
                                for x in outputs]).mean()

        logs = {
            'test_loss': test_loss,
            'test_acc': test_acc
        }

        wandb.log(logs)
        self.log_dict(logs)

    def load_model_weights_from_ckpt(self, path:str = None) -> None:
        """Load model weights to model on cpu"""
        ckpt = torch.load(path,
                          map_location=torch.device('cpu'))

        pretrained_dict = ckpt['state_dict']
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k,
                           v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
