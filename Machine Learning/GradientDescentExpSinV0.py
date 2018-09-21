#!/usr/bin/python3

import numpy as np
import sklearn
from sklearn.datasets.samples_generator import make_regression
import matplotlib.pyplot as plt
from scipy import stats

def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0
    m = x.shape[0] # number of samples

    # initial theta
    a_est = np.random.random(x.shape[1])
    b_est = np.random.random(x.shape[1])
    f_est = np.random.random(x.shape[1])
    J = []

    # Cost function J.
    J = np.append(J,sum([(a_est*np.exp(-x*b_est)*np.sin(2*np.pi*f_est*x) - y[i])**2 for i in range(m)])/(2*m))

    # Iterate Loop
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0/m * sum([(t0 + t1*x[i] - y[i]) for i in range(m)])
        grad1 = 1.0/m * sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)])

        # update the theta_temp
        t0 = t0 - alpha * grad0
        t1 = t1 - alpha * grad1

        # mean squared error
        MSE = sum( [ (t0 + t1*x[i] - y[i])**2 for i in range(m)] )/(2*m)

        iter += 1  # update iter
        J = np.append(J,MSE)

        if abs(J[iter]-J[iter-1]) <= ep:
            print('Converged, iterations: ', iter, '!!!')
            converged = True

        if iter == max_iter:
            print('Max interactions exceeded!')
            converged = True

    return t0,t1,J

if __name__ == '__main__':
    x = np.arange(0,8,0.01)
    y = 4*np.exp(-x/2)*np.sin(2*np.pi*2*x)

    # Random noise.
    noise = np.random.randn(len(x))
    y_n = y + noise

    # Descent gradient parameters.
    alpha = 0.01 # learning rate
    ep = 0.01 # convergence criteria

    # Call gradient descent, get: a, tau, f
    a, tau, f, J = gradient_descent(alpha, x, y_n, ep, max_iter=1000)

    # Plot.
    plt.plot(t, y, 'b-', label='Original signal')
    plt.scatter(t, y_n, s=4, c='r', marker='o', label='Noisy signal')
    plt.title('Oscilation with exponential decay')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    print("Done!")
