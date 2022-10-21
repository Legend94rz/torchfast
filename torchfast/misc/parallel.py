import multiprocessing as mp
from multiprocessing.pool import worker


class SmartPool(mp.pool.Pool):
    """扩展multiprocessing.pool.Pool。
    支持istarmap； initializer 第一个参数为进程序号。

    NOTE:
        可以通过 threading.current_thread().foo = bar 的方式保存 initializer 中的一些变量，以便在 worker 函数中获取。
    """
    @classmethod
    def _map_proxy(cls, args):
        print(args)
        func, *a = args
        return func(*a)

    def istarmap(self, func, iter, chunksize=1):
        return super().imap(self._map_proxy, ((func, ) + tuple(x) for x in iter), chunksize)

    @staticmethod
    def _repopulate_pool_static(ctx, Process, processes, pool, inqueue,
                                outqueue, initializer, initargs,
                                maxtasksperchild, wrap_exception):
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        for i in range(processes - len(pool)):
            w = Process(ctx, target=worker,
                        args=(inqueue, outqueue,
                              initializer,
                              (i, ) + initargs, maxtasksperchild,
                              wrap_exception))
            w.name = w.name.replace('Process', 'PoolWorker')
            w.daemon = True
            w.start()
            pool.append(w)
            mp.util.debug('added worker')
