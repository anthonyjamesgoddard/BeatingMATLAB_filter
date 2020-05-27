import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

chunk_sizes = [800,8000,80000,800000]

# the data is organised as follows
# filter_size | size | avx | real_imag | rev_filter | vanilla

d = {2:'avx', 3:'real_imag', 4:'rev_filter', 5:'vanilla'}

def plot_chunk_data(chunk_size, lines):
    plt.figure()
    plt.title("SIZE :" + str(chunk_size))
    x = [line.split()[0] for line in lines if int(line.split()[1]) == chunk_size]
    y1 = [int(line.split()[2]) for line in lines if int(line.split()[1]) == chunk_size]
    y2 = [int(line.split()[3]) for line in lines if int(line.split()[1]) == chunk_size]
    y3 = [int(line.split()[4]) for line in lines if int(line.split()[1]) == chunk_size]
    y4 = [int(line.split()[5]) for line in lines if int(line.split()[1]) == chunk_size]
    plt.xlabel("FILTER_SIZE")
    plt.ylabel("ms")
    plt.loglog(x,y1, label='avx')
    plt.loglog(x,y2, label='real-imag')
    plt.loglog(x,y3, label='rev-filter')
    plt.loglog(x,y4, label='vanilla')
    ax = plt.gca()
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    plt.xticks([8,16,32,64,128,256, 512,1024])
    plt.xlim(7,1025)
    plt.legend(loc=2)
    plt.savefig("plots/" + str(chunk_size) + ".pdf", dpi=100)

def plot_col(col, lines):
    plt.figure()
    plt.title(d[col])
    x = [int(line.split()[0]) for line in lines]
    all_data = [int(line.split()[col]) for line in lines]
    x1 = x[0:7]
    y1 = all_data[0:7]
    x2 = x[8:15]
    y2 = all_data[8:15]
    x3 = x[16:24]
    y3 = all_data[16:24]
    x4 = x[25:32]
    y4 = all_data[25:32]
    plt.xlabel("FILTER_SIZE")
    plt.ylabel("ms")
    plt.loglog(x1,y1, label='SIZE = 800')
    plt.loglog(x2,y2, label='SIZE = 8000')
    plt.loglog(x3,y3, label='SIZE = 80000')
    plt.loglog(x4,y4, label='SIZE = 800000')
    ax = plt.gca()
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    plt.xticks([8,16,32,64,128,256, 512,1024])
    plt.xlim(7,1025)
    plt.legend(loc=2)
    plt.savefig("plots/" + d[col] + ".png", dpi=100)
    

with open("linux_benchmarks.txt") as f:
    lines = f.readlines()
    for c in chunk_sizes:
        plot_chunk_data(c, lines)
    for i in range(2,6):
        plot_col(i, lines)

