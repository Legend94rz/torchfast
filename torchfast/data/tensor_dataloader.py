import torch as T
from torch.utils.data import SequentialSampler, RandomSampler


class TensorDataLoader:
    """
    Warning:
        `TensorDataLoader` doesn't support distributed training now.
    
    Note:
        `shuffle` is mutually exclusive with `sampler`.
    """
    def __init__(self, *tensors, batch_size, shuffle=False, pin_memory=False, sampler=None):
        assert all(len(t) == len(tensors[0]) for t in tensors)
        assert sampler is None or not shuffle, "`shuffle` and `sampler` is exclusive."
        self.pin_memory = pin_memory and T.cuda.is_available()
        self.tensors = [T.as_tensor(t) for t in tensors]
        if self.pin_memory:
            self.tensors = [t.pin_memory() for t in self.tensors]
        self.batch_size = batch_size
        self.shuffle = shuffle
        if sampler is None:
            sampler = RandomSampler(self.tensors[0]) if shuffle else SequentialSampler(self.tensors[0])
        self.sampler = sampler
        self.dataset_len = len(sampler)
        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        self.i = None
        self.ref_tensors = None

    def __iter__(self):
        r = list(self.sampler)
        self.ref_tensors = [t[r] for t in self.tensors]
        del r
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i: self.i+self.batch_size] for t in self.ref_tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
