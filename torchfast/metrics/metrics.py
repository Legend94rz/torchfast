from abc import abstractmethod, ABCMeta
import torch as T
from . import functional as F
from ..misc.misc import all_reduce_mean, distributed
from sklearn.metrics import multilabel_confusion_matrix
from typing import Optional
import numpy as np


# 替换现有传入fit中的metrics参数为:[(0, 'acc', BinaryAccuracy())]。原有写法现在会自动转化为LambdaAverageMeter。Meter构造函数尽量不要有参数，保持简洁。


class BaseMeter(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.reset()

    def to(self, device):
        for k in self.__dict__.keys():
            if isinstance(self.__dict__[k], T.Tensor):
                self.__dict__[k] = self.__dict__[k].to(device)
        return self

    @abstractmethod
    def sync(self) -> None:
        # sync 本身并不返回值，只是同步一些统计变量。需要再用value属性获得度量值。
        raise NotImplementedError("Not Implememted `sync` method for your metric.")

    def reset(self) -> "BaseMeter":
        return self

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Not Implememted `__call__` method for your metric.")

    @property
    @abstractmethod
    def value(self) -> T.Tensor:
        raise NotImplementedError("Not Implememted `value` method for your metric.")


class AverageMeter(BaseMeter):
    """
    在forward里实现一步的计算，__call__里实现如移动平均等统计，并返回（平均后的）结果。
    """
    def __call__(self, *args, **kwargs):
        v = self.forward(*args, **kwargs)
        self._val = (self._val * self.n + v) / (self.n + 1)
        self.n += 1
        return self._val

    def reset(self):
        self.n = 0
        self._val = T.zeros(1)
        return self

    def sync(self):
        all_reduce_mean(self._val)

    @property
    def value(self):
        return self._val


class LambdaAverageMeter(AverageMeter):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class BinaryAccuracy(AverageMeter):
    def forward(self, input, target):
        return F.binary_accuracy(input, target)


class BinaryAccuracyWithLogits(AverageMeter):
    def forward(self, input, target):
        return F.binary_accuracy_with_logits(input, target)


class SparseCategoricalAccuracy(AverageMeter):
    def forward(self, input: T.Tensor, target: T.Tensor):
        return F.sparse_categorical_accuracy(input, target)


class MeanSquaredError(AverageMeter):
    def forward(self, input: T.Tensor, target: T.Tensor):
        return F.mean_squared_error(input, target)


class RootMeanSquaredError(AverageMeter):
    def forward(self, input: T.Tensor, target: T.Tensor):
        return F.root_mean_squared_error(input, target)


class MeanAbsoluteError(AverageMeter):
    def forward(self, input: T.Tensor, target: T.Tensor):
        return F.mean_absolute_error(input, target)


class _ConfusionMatrixBased(BaseMeter):
    def __init__(self, threshold=.0, average: Optional[str] = 'macro', pos_label=1):
        """
        用于二分类、多分类或多标记下的precision/recall/f1计算。

        Args:
            threshold (int, optional): 二值化inputs的阈值。>threshold为1，否则为0，形状需要兼容inputs。Defaults to `0`.
            average (str, optional): {'micro', 'macro', 'binary'} or None.  Defaults to 'macro'.
                - `macro`: 各个类分别计算f1，然后求平均
                - `micro`: 统一计算tp/fp/fn，然后直接求f1. 
                - `binary`: 仅计算由`pos_label`指定的类。
            pos_label: 仅当`average=="binary"`时有效。

        Inputs:
            input: (n_sample) or (n_sample x n_label). float array. logits / prob
            targets: same as input. 0-1 array.
        """
        super().__init__()
        assert average in {'macro', 'micro', 'binary', None}
        self.average = average
        self.threshold = T.tensor(threshold)
        self.pos_label = pos_label

    def reset(self):
        self.confusion_mat = T.zeros(1)
        return self

    def __call__(self, input: T.Tensor, target: T.Tensor):
        # binary/multi-label: input: [N, *], target: [N, *]
        # multi-class: input: [N, c], target: [N, ]
        if not (input.ndim - 1 == target.ndim or input.ndim == target.ndim):
            raise RuntimeError(f"The ndim is mismatch of input: {input.ndim} and target: {target.ndim}.")
        if input.ndim != target.ndim: # multi-class
            labels = list(range(input.shape[-1]))
            pred = input.argmax(-1).cpu().numpy().astype('int')
        else:  # binary/multi-label
            labels = [0, 1]
            pred = (input > self.threshold).cpu().numpy().astype('int')
        mcm = T.tensor(np.ascontiguousarray(multilabel_confusion_matrix(target.cpu().numpy().astype('int'), pred, labels=labels)),
                       device=self.confusion_mat.device)
        self.confusion_mat = (self.confusion_mat + mcm).contiguous()
        return self.value

    def sync(self):
        distributed.all_reduce(self.confusion_mat, op=distributed.ReduceOp.SUM)


class F1Score(_ConfusionMatrixBased):
    @classmethod
    def f1(cls, mcm: T.Tensor, average: Optional[str], pos_label: int):
        tp = mcm[:, 1, 1]
        fp = mcm[:, 0, 1]
        fn = mcm[:, 1, 0]
        if average == 'macro':
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            score = (2 * p * r / (p + r)).mean()
        elif average == 'micro':
            p = tp.sum() / (tp.sum() + fp.sum())
            r = tp.sum() / (tp.sum() + fn.sum())
            score = 2 * p * r / (p + r)
        elif average == 'binary':
            p = tp[pos_label] / (tp[pos_label] + fp[pos_label])
            r = tp[pos_label] / (tp[pos_label] + fn[pos_label])
            score = (2 * p * r / (p + r))
        elif average is None:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            score = (2 * p * r / (p + r))
        else:
            raise RuntimeError(f"Unknown average method: {average}")
        return score

    @property
    def value(self):
        return self.f1(self.confusion_mat, self.average, self.pos_label)


class Recall(_ConfusionMatrixBased):
    @classmethod
    def recall(cls, mcm: T.Tensor, average: Optional[str], pos_label: int):
        tp = mcm[:, 1, 1]
        fn = mcm[:, 1, 0]
        if average == 'macro':
            score = (tp / (tp + fn)).mean()
        elif average == 'micro':
            score = tp.sum() / (tp.sum() + fn.sum())
        elif average == 'binary':
            score = tp[pos_label] / (tp[pos_label] + fn[pos_label])
        elif average is None:
            score = tp / (tp + fn)
        else:
            raise RuntimeError(f"Unknown average method: {average}")
        return score

    @property
    def value(self):
        return self.recall(self.confusion_mat, self.average, self.pos_label)


class Precision(_ConfusionMatrixBased):
    @classmethod
    def precision(cls, mcm: T.Tensor, average: Optional[str], pos_label: int):
        tp = mcm[:, 1, 1]
        fp = mcm[:, 0, 1]
        if average == 'macro':
            score = (tp / (tp + fp)).mean()
        elif average == 'micro':
            score = tp.sum() / (tp.sum() + fp.sum())
        elif average == 'binary':
            score = tp[pos_label] / (tp[pos_label] + fp[pos_label])
        elif average is None:
            score = tp / (tp + fp)
        else:
            raise RuntimeError(f"Unknown average method: {average}")
        return score

    @property
    def value(self):
        return self.precision(self.confusion_mat, self.average, self.pos_label)


class ROCAUC(BaseMeter):
    @classmethod
    def auc(cls, mcmt: T.Tensor):
        tp, fp, fn, tn = mcmt
        x = fp / (fp + tn)  # [c, #thres]
        y = tp / (tp + fn)
        num_thres = x.shape[1]
        assert y.shape[1] == num_thres
        area = 0.5 * (x[:, :num_thres - 1] - x[:, 1:]) * (y[:, :num_thres - 1] + y[:, 1:])
        return area.sum(1).mean()

    def __init__(self, num_thresholds=200):
        """
        用于二分类或多标记下的ROC AUC计算。多标记下取各标记下的auc平均值，即macro方式。

        Args:
            num_thresholds (int, optional): [0, 1]分成多少个区间. Defaults to 200.

        Inputs:
            input: (n_sample) or (n_sample x n_label). float array (prob)
            targets: same as input. 0-1 array.

        Example:
            ```
                auc = ROCAUC(10)
                auc(T.tensor([0.9, 0.8, 0.7, 0.6, 0.5]), T.tensor([1, 1, 0, 1, 1]))
                print(auc.value) # 0.5
                auc(T.tensor([0.4, 0.3, 0.2, 0.1, 0]), T.tensor([0, 0, 1, 0, 0]))
                print(auc.value) # 0.8
            ```
        """
        super().__init__()
        self.threshold = T.linspace(0, 1, num_thresholds + 1)

    def reset(self):
        self.confusion_mat = T.zeros(1)

    def __call__(self, input: T.tensor, target: T.tensor):
        mcmt = F.multilab_confusion_matrix_at_threshold(input, target, self.threshold)  # [tp, fp, fn, tn]
        self.confusion_mat = self.confusion_mat + mcmt
        return self.value

    def sync(self):
        distributed.all_reduce(self.confusion_mat, op=distributed.ReduceOp.SUM)

    @property
    def value(self):
        return self.auc(self.confusion_mat)
