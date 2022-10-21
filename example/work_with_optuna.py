from torch import nn
import numpy as np
import torch as T
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from torchfast import NNRegressor, NNClassifier, SparseCategoricalAccuracy
import optuna
from functools import partial


class MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, x):
        return self.m(x)


def train(cfg, X, y, cv=True):
    h = cfg.suggest_int('hidden', 3, 13)
    clf = NNClassifier(MLP, T.optim.Adam, T.nn.CrossEntropyLoss(), batch_size=2, metrics=[(0, 'acc', SparseCategoricalAccuracy())], epochs=10, device='cpu', hidden=h)
    if cv:
        res = cross_validate(clf, X, y, cv=5, n_jobs=1, return_estimator=False, pre_dispatch=None, scoring='accuracy', error_score='raise')
        del clf
        return np.mean(res['test_score'])
    else:
        clf.fit(X, y)
        return clf


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X = X.astype('float32')
    sampler = optuna.samplers.TPESampler(n_startup_trials=4)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    # or using database to enable parallel hyper-params tuning:
    # study = optuna.create_study(direction='maximize', sampler=sampler, load_if_exists=True, storage='sqlite:///study.db')  # mysql is suggested: mysql+mysqlconnector://user:pasword@host/database
    study.optimize(partial(train, X=X, y=y, cv=True), n_trials=8, gc_after_trial=True)
    print(study.best_trial)
    print(train(study.best_trial, X, y, cv=False))
