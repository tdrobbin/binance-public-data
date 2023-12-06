from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

from tqdm import tqdm


class WorkerPool:

    def __init__(self, n=24, desc='Downloading'):
        self.n = n
        self.desc = desc
        self.pool = ThreadPoolExecutor(max_workers=self.n)
        self.futures = {}
        self.closed = False

    def submit(self, func, *args):
        assert not self.closed
        future = self.pool.submit(func, *args)
        self.futures[future] = args

    def work(self):
        for future in tqdm(as_completed(self.futures),
                           total=len(self.futures),
                           desc=self.desc):
            inputs = self.futures[future]
            future.result()
            print(f'Completed {inputs}')
        self.pool.shutdown()
        self.closed = True
        print(f'Completed {self.desc}')
