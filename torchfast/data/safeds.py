from functools import wraps


class let_it_safe:
    """
    Skip invalid index of a torch (custom) `Dataset` object.
    
    e.g.:
    ```
    @let_it_safe()
    class MyDataset(Dataset):
        ...
    ```
    """
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.next_cache = {}
    
    def __call__(self, cls):
        self.origin_get_fn = cls.__getitem__
        
        @wraps(cls)
        def safe_get(slf, idx):
            if idx >= len(slf):
                raise IndexError()
            
            start = idx
            it = None
            re_read = True
            if start not in self.next_cache:
                re_read = False
                while True:
                    try:
                        it = self.origin_get_fn(slf, idx)
                        if self.use_cache:
                            self.next_cache[start] = idx
                        break
                    except Exception as e:
                        idx += 1
                        if idx >= len(slf):
                            idx = 0
            idx = self.next_cache[start]
            if re_read:
                return self.origin_get_fn(slf, idx)
            return it

        cls.__getitem__ = safe_get
        return cls
