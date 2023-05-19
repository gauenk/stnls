import time
import torch as th
import numpy as np


class ExpTimer():

    def __init__(self,use_timer=True):
        self.use_timer = use_timer
        self.times = []
        self.names = []
        self.start_times = []

    def __str__(self):
        msg = "--- Exp Times ---"
        for k,v in self.items():
            msg += "\n%s: %2.3e\n" % (k,v)
        return msg

    def __getitem__(self,name):
        idx = self.names.index(name)
        total_time = self.times[idx]
        return total_time

    def __setitem__(self,name,time):
        if name in self.names:
            raise KeyError(f"Already set key [{name}]")
        self.names.append(name)
        self.times.append(time)

    def keys(self):
        names = ["timer_%s" % name for name in self.names]
        return names

    def items(self):
        names = ["timer_%s" % name for name in self.names]
        return zip(names,self.times)

    def sync_start(self,name):
        if self.use_timer is False: return
        th.cuda.synchronize()
        self.start(name)

    def start(self,name):
        if self.use_timer is False: return
        th.cuda.synchronize()
        if name in self.names:
            print(self.names)
            raise ValueError("Name [%s] already in list." % name)
        self.names.append(name)
        self.times.append(-1)
        start_time = time.perf_counter()
        self.start_times.append(start_time)

    def sync_stop(self,name):
        if self.use_timer is False: return
        th.cuda.synchronize()
        self.stop(name)

    def stop(self,name):
        if self.use_timer is False: return
        end_time = time.perf_counter() # at start
        idx = self.names.index(name)
        start_time = self.start_times[idx]
        exec_time = end_time - start_time
        self.times[idx] = exec_time

class ExpTimerList(ExpTimer):

    def __init__(self,use_timer=True):
        super().__init__(use_timer)

    def __setitem__(self,name,time):
        assert isinstance(time,list)
        if name in self.names:
            idx = self.names.index(name)
            self.times[idx] = time
        else:
            self.names.append(name)
            self.times.append(time)

    def start(self,name):
        if self.use_timer is False: return
        th.cuda.synchronize()
        if not(name in self.names):
            self.names.append(name)
            start_time = time.perf_counter()
            self.start_times.append(start_time)
        else:
            idx = self.names.index(name)
            self.start_times[idx] = time.perf_counter()

    def stop(self,name):
        if self.use_timer is False: return
        end_time = time.perf_counter() # at start
        idx = self.names.index(name)
        start_time = self.start_times[idx]
        exec_time = end_time - start_time
        if idx < len(times):
            self.times[idx].append(exec_time)
        else:
            self.times.append([exec_time])

    def update_times(self,timer):
        # print(timer.names)
        if not(self.use_timer): return
        for key in timer.names:
            if key in self.times.names:
                self.times[key].append(timer[key])
            else:
                self.times[key] = [timer[key]]

    def __str__(self):
        msg = "--- Exp Times ---"
        for k,v in self.items():
            msg += "\n%s: %2.3f\n" % (k,np.sum(v))
        return msg


class AggTimer(ExpTimer):

    def __init__(self,use_timer=True):
        super().__init__(use_timer)

    def __str__(self):
        msg = "--- Exp Times ---"
        for k,v in self.items():
            msg += "\n%s: %2.3f\n" % (k,np.sum(v))
        return msg



class TimeIt():
    """

    Support using ExpTimer and "with"

    timer = ExpTimer()
    with TimeIt(timer,"name"):
       ...

    """

    def __init__(self,timer,name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.timer.sync_start(self.name)
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.timer.sync_stop(self.name)
