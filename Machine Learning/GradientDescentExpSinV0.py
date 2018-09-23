#!/usr/bin/python3
# Simple Gradient descent algorithm. It get stuck on local minimas.

import numpy as np
import sklearn
from sklearn.datasets.samples_generator import make_regression
import matplotlib.pyplot as plt
from scipy import stats

def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0
    m = len(x) # number of samples

    # initial theta
    a_est = 5
    b_est = 0.7
    f_est = 1
    J = []

    # Cost function J.
    J = np.append(J,sum([(a_est*np.exp(-x[i]*b_est)*np.cos(2*np.pi*f_est*x[i]) - y[i])**2 for i in range(m)])/(2*m))

    # Iterate Loop.
    while not converged:
        # For each training sample, compute the gradient.
        grad0 = 1.0/m * sum([((a_est*np.exp(-b_est*x[i])*np.cos(2*np.pi*f_est*x[i])-y[i])*np.exp(-b_est*x[i])*np.cos(2*np.pi*f_est*x[i])) for i in range(m)])
        grad1 = 1.0/m * sum([(a_est*x[i]*np.exp(-b_est*x[i])*np.cos(2*np.pi*f_est*x[i])*(y[i]-a_est*np.exp(-b_est*x[i])*np.cos(2*np.pi*f_est*x[i]))) for i in range(m)])
        grad2 = 1.0/m * sum([(2*np.pi*a_est*x[i]*np.exp(-b_est*x[i])*np.sin(2*np.pi*f_est*x[i])*(y[i]-a_est*np.exp(-b_est*x[i])*np.cos(2*np.pi*f_est*x[i]))) for i in range(m)])

        # Update estimated parameters
        a_est_temp = a_est - 0.05*grad0
        b_est_temp = b_est - 0.005*grad1
        f_est_temp = f_est - 0.02*grad2

        a_est = a_est_temp
        b_est = b_est_temp
        f_est = f_est_temp

        # Mean squared error
        MSE = sum([(a_est*np.exp(-x[i]*b_est)*np.cos(2*np.pi*f_est*x[i]) - y[i])**2 for i in range(m)])/(2*m)

        iter += 1  # update iter
        J = np.append(J,MSE)

        if abs(J[iter]-J[iter-1]) <= ep:
            print('Converged, iterations: ', iter, '!!!')
            converged = True

        if iter == max_iter:
            print('Max interations exceeded!')
            converged = True

    return a_est, b_est, f_est, J

if __name__ == '__main__':
    x = np.arange(0,6,0.01)
    a = 7
    tau = 2
    f = 2

    y = a*np.exp(-x/tau)*np.cos(2*np.pi*f*x)

    # Random noise.
    noise = 0.8*np.random.randn(len(x))
    y_n = y + noise

    # Descent gradient parameters.
    alpha = 0.01 # learning rate
    ep = 0.0005 # convergence criteria

    # Call gradient descent, get: a, tau, f
    a_est, b_est, f_est, J = gradient_descent(alpha, x, y_n, ep, max_iter=1000)
    tau_est = 1/b_est
    print("a = %s tau = %s f = %s" % (a, tau, f))
    print("a_est = %s tau_est = %s f_est = %s" % (a_est, tau_est, f_est))

    iterations = np.arange(len(J))

    y_est = a_est*np.exp(-x/tau_est)*np.cos(2*np.pi*f_est*x)

    # Plot.
    plt.subplot(2,1,1)
    plt.plot(x, y, 'b-', label='Original signal')
    plt.scatter(x, y_n, s=4, c='r', marker='o', label='Noisy signal')
    plt.plot(x, y_est, 'k-', label='Estimated signal')
    plt.title('Oscilation with exponential decay')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(iterations,J)
    plt.title('Error vs Iterations')
    plt.show()
    print("Done!")
