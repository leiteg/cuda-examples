#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def do_plot(df, xlabel, label, ax=None):
    if not ax:
        fig, ax = plt.subplots()
    ax.plot(df[0], df[2], 'o-', label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Effective Bandwith (GB/s)')
    ax.legend()
    ax.set_ylim((0, 160))
    plt.tight_layout()
    plt.show()
    return ax


def main():
    df_s = pd.read_csv('stride.txt', sep='\t', header=None)
    df_o = pd.read_csv('offset.txt', sep='\t', header=None)
    do_plot(df_s, 'Stride', 'GTX 1060 6GB')
    do_plot(df_o, 'Offset', 'GTX 1060 6GB')


if __name__ == '__main__':
    main()
