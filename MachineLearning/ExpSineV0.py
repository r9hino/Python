#!/usr/bin/python3

import numpy as np
import sklearn
from sklearn.datasets.samples_generator import make_regression
import matplotlib.pyplot as plt
from scipy import stats


if __name__ == '__main__':
    t = np.arange(0,8,0.01)
    y = 4*np.exp(-t/2)*np.sin(2*np.pi*2*t)

    # Random noise.
    noise = np.random.randn(len(t))
    y_n = y + noise



    # Plot.
    plt.plot(t, y, 'b-', label='Original signal')
    plt.scatter(t, y_n, s=4, c='r', marker='o', label='Noisy signal')
    plt.title('Oscilation with exponential decay')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    print("Done!")
