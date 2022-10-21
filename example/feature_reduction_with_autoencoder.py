from sklearn.base import TransformerMixin
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

from torchfast import nn, Learner, T
from torchfast.learner.skwrapper import _SKWrapperBase


class Encoder(nn.Module):
    def __init__(self, input_dim, n_hidden=(10, 4)):
        super(Encoder, self).__init__()
        sz = (input_dim, ) + n_hidden
        self.top = nn.Linear(input_dim, n_hidden[0])
        self.m = nn.Sequential(*sum([[nn.ReLU(), nn.Linear(n_hidden[i], n_hidden[i+1])] for i in range(len(n_hidden)-1)], []))

    def forward(self, x):
        return self.m(self.top(x))


class Decoder(nn.Module):
    def __init__(self, output_dim, n_hidden=(4, 10)):
        super(Decoder, self).__init__()
        self.top = nn.SiLU()
        self.m = nn.Sequential(*sum([[nn.Linear(n_hidden[i], n_hidden[i+1]), nn.ReLU()] for i in range(len(n_hidden)-1)], []))
        self.bottom = nn.Sequential(nn.Linear(n_hidden[-1], output_dim))

    def forward(self, x):
        x = self.top(x)
        x = self.m(x)
        return self.bottom(x)


class EncoderDecoder(nn.Module):
    def __init__(self, fea_dim):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(fea_dim)
        self.decoder = Decoder(fea_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoEncoderTransformer(TransformerMixin, _SKWrapperBase):
    def fit(self, X, y=None):
        m = EncoderDecoder(self.input_dim)
        encdec = Learner(m, self.optimizer_fn, self.loss_fn)
        cv = ShuffleSplit(test_size=self.validation_fraction, random_state=self.random_state)
        idx_train, idx_val = next(cv.split(X))
        train_ds = (X[idx_train], X[idx_train])
        val_ds = (X[idx_val], X[idx_val])
        encdec.fit(train_ds, self.epochs, self.batch_size, self.metrics, val_ds, self.callbacks, self.device,
                          self.verbose, **self._kwargs)
        self.learner_ = Learner(m.encoder)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        return self.learner_.predict(X)


if __name__ == "__main__":
    # generate some data
    n_fea = 20
    X, y = make_classification(20000, n_fea)
    X = X.astype('float32')

    # make a pipe line
    pp = make_pipeline(AutoEncoderTransformer(optimizer_fn=T.optim.Adam, loss_fn=T.nn.MSELoss(), input_dim=20, device='cpu'), LogisticRegression())

    # fit & eval
    res = cross_validate(pp, X, y, cv=5, n_jobs=1, pre_dispatch=None)
    print(res['test_score'])
    res = cross_validate(LogisticRegression(), X, y, cv=5)
    print(res['test_score'])
