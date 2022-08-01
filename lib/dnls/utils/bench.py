import torch as th

class RecordIt:
    def __init__(self,gpu_rec,timer):
        self.gpu_rec = gpu_rec
        self.timer = timer
        self.curr_name = None
        self.use_synch = False

    def __call__(self,name,sync=False):
        self.curr_name = name
        self.use_synch = sync
        return self

    def __enter__(self):
        if self.use_synch:
            th.cuda.empty_cache()
            th.cuda.synchronize()
        self.gpu_rec.reset()
        self.timer.start(self.curr_name)

    def __exit__(self, type, value, traceback):
        if self.use_synch:
            th.cuda.synchronize()
        self.timer.stop(self.curr_name)
        self.gpu_rec.snap(self.curr_name)

    def __del__(self):
        self.curr_name = None
        self.use_synch = False

    def __str__(self):
        timer_str = str(self.timer)
        gpu_str = str(self.gpu_rec)
        msg = "--- RecordIt Summary ---\n\n"
        msg += timer_str + "\n"
        msg += gpu_str
        return msg
