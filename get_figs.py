import numpy as np
import matplotlib.pyplot as plt

chunk_sizes = [800,8000,80000,800000]

def plot_chunk_data(chunk_size, lines):
    plt.figure()
    plt.title(str(chunk_size))
    x = [line.split()[0] for line in lines if int(line.split()[1]) == chunk_size]
    y1 = [int(line.split()[2]) for line in lines if int(line.split()[1]) == chunk_size]
    y2 = [int(line.split()[3]) for line in lines if int(line.split()[1]) == chunk_size]
    y3 = [int(line.split()[4]) for line in lines if int(line.split()[1]) == chunk_size]
    y4 = [int(line.split()[5]) for line in lines if int(line.split()[1]) == chunk_size]
    plt.plot(x,y1, label='avx')
    plt.plot(x,y2, label='real-imag')
    plt.plot(x,y3, label='rev-filter')
    plt.plot(x,y4, label='vanilla')
    plt.legend()
    plt.figure()

with open("linux_benchmarks.txt") as f:
    lines = f.readlines()
    for c in chunk_sizes:
        plot_chunk_data(c, lines)
    plt.show()

