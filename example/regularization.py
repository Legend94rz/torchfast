from torchfast import L2Regularization, Learner, T, nn, NNClassifier, BinaryAccuracy, BinaryAccuracyWithLogits
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, cross_validate
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from functools import partial


class MySVC(nn.Module):
    def __init__(self, n_feature=20, C=1.0):
        super().__init__()
        self.w = L2Regularization(nn.Linear(n_feature, 1, bias=False), C)
        self.b = nn.Parameter(T.tensor(0.))
        #self.w = nn.Linear(n_feature, 1)
    
    def forward(self, x):
        return self.w(x) + self.b

class SVCLearner(Learner):
    def compute_metric(self, idx, name, func, detached_results, batch_data):
        batx, baty = batch_data
        return func((detached_results[0] > 0).float(), baty)


class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, wx, target):
        remap_target = (target * 2) - 1     # [0, 1] -> [-1, 1]
        return T.clamp(1 - remap_target * wx, min=0).mean()


class MLP(nn.Module):
    def __init__(self, n_feature=20):
        super().__init__()
        self.w = nn.Sequential(nn.Dropout(0.1), nn.Linear(n_feature, 64), nn.SiLU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.SiLU(), nn.Dropout(0.1), nn.Linear(32, 1))
    
    def forward(self, x):
        return self.w(x)

    
if __name__ == "__main__":
    X, y = make_classification(10000, 20, random_state=0)
    y = y.reshape(-1, 1).astype('float32')
    kf = KFold(5)
    for f, (train_idx, val_idx) in enumerate(kf.split(range(len(X)))):
        print('============ fold {f} ===============')
        m = SVCLearner(MySVC(), partial(T.optim.SGD, lr=1e-3), HingeLoss())
        m.fit((X[train_idx].astype('float32'), y[train_idx]), 100, 64, [(0, 'acc', BinaryAccuracy())], (X[val_idx].astype('float32'), y[val_idx]), device='cpu', shuffle=True)
        #m = Learner(MLP(), T.optim.Adam, nn.BCEWithLogitsLoss())
        #m.fit((X[train_idx].astype('float32'), y[train_idx]), 100, 64, [(0, 'acc', BinaryAccuracyWithLogits())], (X[val_idx].astype('float32'), y[val_idx]), device='cpu', shuffle=True)
        m2 = SVC()
        m2.fit(X[train_idx], y[train_idx])
        print('sklearn svc acc:', accuracy_score(y[val_idx], m2.predict(X[val_idx])))
