import torch
import torchmetrics


class ClassificationMetrics():
    def __init__(self, y_pred: torch.Tensor, y: torch.Tensor, mode: str = 'binary'):
        self.mode = mode
        self._mode_check()
        self.y = y.detach().cpu()
        self.y_pred = y_pred.detach().cpu()
        self.confusion = self._get_confusion()
        self.accuracy = self._calc_accuracy()

    def _get_confusion(self):
        matrix = torchmetrics.ConfusionMatrix(num_classes = 10)
        matrix = matrix(self.y_pred, self.y)
        return matrix

    def _calc_accuracy(self):
        acc = torchmetrics.Accuracy(num_classes = 10)
        acc = acc(self.y_pred, self.y)
        return acc

    def _mode_check(self):
        if self.mode == 'binary':
            pass
        else:
            raise NotImplementedError('This mode has not been implemented yet')

    @property
    def acc(self):
        return self.accuracy

    @property
    def confusion_matrix(self):
        return self.confusion
