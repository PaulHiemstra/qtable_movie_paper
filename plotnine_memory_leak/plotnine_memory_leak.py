import matplotlib.pyplot as plt
import sys
try:
    if sys.argv[1] == 'ioff':
        plt.ioff()
except:
    plt.ion()

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from plotnine import *
import psutil 
import os

# Functions based on https://www.geeksforgeeks.org/how-to-get-current-cpu-and-ram-usage-in-python/
def get_mem_usage():
    return psutil.virtual_memory()[3]

def get_cpu_usage():
    load1, load5, load15 = psutil.getloadavg() 
    return (load15/os.cpu_count()) * 100

def test_mem(dot_each = 10, total = 100, verbose=False):
    plot_data = pd.DataFrame(np.random.uniform(size=[200,2]), columns=['a', 'b'])
    res = (
        ggplot(plot_data) + geom_point(aes(x='a', y='b')) + theme(figure_size=[20,20])
    )

    it = 0
    mem_usage = []
    cpu_usage = []
    while True:
        ggsave(res, filename='test.png')
        it += 1
        if it % dot_each == 0:
            if verbose:
                print('.', end='')
            mem_usage.append(get_mem_usage())
            cpu_usage.append(get_cpu_usage())
        if it == total:
            break
    if verbose:
        return [mem_usage, cpu_usage, (max(mem_usage) - min(mem_usage)) / 1e9]
    else:
        return (max(mem_usage) - min(mem_usage)) / 1e9
    
#mem_usage, cpu_usage, gb_increase = test_mem()
for i in range(5):
    print('Memory increase', test_mem(), 'GB')