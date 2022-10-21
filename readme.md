# Fast Torch

[![PyPI version](https://badge.fury.io/py/torchfast.svg)](https://badge.fury.io/py/torchfast)

A keras-like library for pytorch.
Easy to use and more efficient.
In most cases, the only thing you need to do is to define a `nn.Module`,
write the `forward`, and call `Learner(module, optim, loss).fit()` with the help of Torchfast.


# Setup
## Install via `pip`

`pip install Torchfast`

## Install via source code
1. clone this repo:

   `git clone https://github.com/Legend94rz/torchfast`


2. setup by `setup.py`:

   `python setup.py install`

   or, you can build a `*.whl` package and then install it by `pip`:

   ```
   python setup.py bdist_wheel
   pip install -U (the-whl-file-name-generated-just-now).whl
   ```

# Features
Add one line code to:
* Stochastic weighted averaging (SWA)
* Automatic mixed precision (AMP)
* Data distributed parallel (DDP) training
* More precise f1/precision/recall score
* Torch profiling
* Ploting metrics to TensorBoard
* More efficient `DataLoader` when using `Tensor` as input

and more...

<!--

Todo:
* 如何用SWA来evaluate?如何每隔x个epoch验证?
* callback支持传入更多的变量,以支持更一般的场景?
* metric每隔x个step计算一次?
* 某些metric显示在控制台上，另外一些metric用其他方式显示，如tensorboard? 

-->


# Tutorial
## Example code

```python
from TorchFast import *
from torch import nn
import numpy as np


class SimpleMLP(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.ln = nn.Sequential(nn.Linear(20, 512), nn.SiLU(), nn.Dropout(0.2), nn.Linear(512, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.ln(x)


if __name__ == "__main__":
    # generate some data
    X = np.random.randn(500000, 20).astype('float32')
    y = (np.median(X, axis=1, keepdims=True)>0).astype('float32')
    print(y.mean())

    # fast torch:
    m = Learner(SimpleMLP(), AdaBelief, BinaryLabelSmoothLoss(0.05))
    m.fit(TensorDataLoader(X[:400000], y[:400000], batch_size=4096, shuffle=True), 1000, None,
          metrics=[(0, 'acc', BinaryAccuracyWithLogits())],   # compute `binary_accuracy_with_logits` w.r.t the 0-th forward output and the targets
          callbacks=[EarlyStopping(verbose=True, patience=7), ReduceLROnPlateau(verbose=True)],
          validation_set=TensorDataLoader(X[400000:], y[400000:], batch_size=4096), verbose=True)
```

## Work with [scikit-learn](https://github.com/scikit-learn/scikit-learn)

See [this](example/work_with_sklearn.py) and [this](example/work_with_optuna.py) examples for more details.


## About distributed training

Firstly, the following line should be added before initializing a learner (and the datasets):

`local_rank = Learner.init_distributed_training(dummy=False, seed=0)`

the `dummy` param is used to debug. If user want to disable parallel temporarily, set `dummy=True`.
This function will return the `LOCAL_RANK` mentioned by `torch.distributed.launch` tool. `seed` is the random seed
used by all the training process, which is optional. TorchFast will choose a random value when it is `None` and ensure
all the processes have same random seeds.

Then start parallel training with the help of the tool `torch.distributed.launch` offered by pytorch:

`python -m torch.distributed.launch --use_env [your script and parameters]`

or the newer tool:

`torchrun [your script and parameters]`

NOTE:
1. `--use_env` is **required** because TorchFast reads the `LOCAL_RANK` from `os.environ`,
   avoiding parses arguments from command line. (Which is default when using `torchrun` command.)

1. When using `ModelCheckpoint`,
   Only the process whose local_rank == 0 will save the checkpoint.

1. TorchFast will add `DistributedSampler` automatically when the values of `training_set` or `validation_set` is not `torch.DataLoader`.
   Besides, users needn't call `sampler.set_epoch` at every epoch beginning, TorchFast will do that for you.
   
1. Doesn't support distributed training in jupyter notebook now.

1. Ensure the `device` parameter is set to `cuda` *exactly*.

1. Currently, TorchFast will automatically take the average of the metric values from each node. However, some measurement methods do not meet this condition, such as `rmse`, i.e. the `rmse` on the whole batch is not equals to the mean of `rmse`s on each sub-batch. In this case, please consider using alternative methods that satisfy this condition, for example, one should use `mse` instead of `rmse` when distributed training.

## For more complex module

Overwrite `Learner.compute_forward`, `Learner.compute_losses`, `Learner.compute_metric`, and `Learner.compute_output` respectively
to custom the data flow.



# Reference

[1] [torch.distributed.launch.py](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py) and [its tutorial](https://pytorch.org/docs/stable/distributed.html#launch-utility)

