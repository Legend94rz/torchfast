from torchfast import *
import numpy as np
from sklearn.metrics import f1_score


class SimpleMLP(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.ln = nn.Sequential(nn.Linear(20, 512), LambdaLayer(lambda x: F.silu(x)), nn.Dropout(0.2), nn.Linear(512, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.ln(x)


if __name__ == "__main__":
    # generate some data
    X = np.random.randn(500000, 20).astype('float32')
    y = (np.median(X, axis=1, keepdims=True)>0).astype('float32')
    print(y.mean())

    # fast torch:
    m = Learner(SimpleMLP(), AdaBelief, BinaryLabelSmoothLoss(0.05), amp=True)
    m.fit((X, y), 2, 4096,
          metrics=[(0, 'acc', BinaryAccuracyWithLogits()), (0, 'f1', F1Score(0, average=None))],
          callbacks=[
              TensorBoard("/home/renzhen/clslog", ['loss', 'val_loss'], {'#': ['acc', 'val_acc']}, 10),
              GradClipper(10), 
              EarlyStopping(verbose=True, patience=7), 
              ReduceLROnPlateau(verbose=True), 
              TorchProfile(on_trace_ready=T.profiler.tensorboard_trace_handler('/home/renzhen/clslog'), schedule=T.profiler.schedule(wait=20, warmup=20, active=1)), 
          ],
          dataset_split=(0.8, 0.2), verbose=True, prefer_tensorloader=True, shuffle=False, device='cuda')

    print(m.predict(X))
