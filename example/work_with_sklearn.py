from torch import nn
import torch as T
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from torchfast import NNRegressor, NNClassifier


class MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, x):
        return self.m(x)


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X = X.astype('float32')
    clf = NNClassifier(MLP, T.optim.Adam, T.nn.CrossEntropyLoss(), batch_size=2, epochs=10, device='cpu', hidden=15)
    # limitation:
    #  if one uses gpu, `n_jobs` must set to 1, and `pre_dispatch` must set to `None`.
    #  or using `threading` as parallel backend:
    #  e.g. `with parallel_backend('threading'):`
    res = cross_validate(clf, X, y, cv=3, n_jobs=1, return_estimator=True, pre_dispatch=None, scoring='accuracy')
    print(res['test_score'])

    res = GridSearchCV(clf, {'hidden': list(range(3, 10, 2)), 'verbose': [False]}, cv=3, verbose=1, pre_dispatch=None, refit=True)
    res.fit(X, y)
    print(res.cv_results_)
