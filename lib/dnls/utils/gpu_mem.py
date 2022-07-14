import torch as th

def print_gpu_stats(gpu_stats,name):
    fmt_all = "[%s] Memory Allocated: %2.3f"
    fmt_res = "[%s] Memory Reserved: %2.3f"
    if gpu_stats:
        th.cuda.empty_cache()
        th.cuda.synchronize()
        mem = th.cuda.memory_allocated() / 1024**3
        print(fmt_all % (name,mem))
        mem = th.cuda.memory_reserved() / 1024**3
        print(fmt_res % (name,mem))

def print_peak_gpu_stats(gpu_stats,name,reset=True):
    th.cuda.empty_cache()
    th.cuda.synchronize()
    mem = th.cuda.max_memory_allocated(0)
    mem_gb = mem / (1024.**3)
    if reset: th.cuda.reset_peak_memory_stats()
    if gpu_stats:
        print("Max Mem (GB): %2.3f" % mem_gb)
    return mem_gb

def reset():
    th.cuda.reset_peak_memory_stats()

class GpuRecord():

    def __init__(self):
        self.mems = []
        self.names = []

    def __str__(self):
        msg = "--- Gpu Mems ---\n"
        for k,m in self.items():
            msg += "\n%s: %2.3f\n" % (k,m)
        return msg

    def __getitem__(self,name):
        idx = self.names.index(name)
        gpu_mem = self.mems[idx]
        return gpu_mem

    def items(self):
        return zip(self.names,self.mems)

    def reset(self):
        print_peak_gpu_stats(False,"",True)

    def snap(self,name):
        mem_gb = print_peak_gpu_stats(False,"",True)
        self.names.append(name)
        self.mems.append(mem_gb)
