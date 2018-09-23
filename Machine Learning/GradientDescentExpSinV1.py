#!/usr/bin/python3
# Simple Gradient descent algorithm. It get stuck on local minimas.

import time
import numpy as np
from numpy import exp, cos, sin, pi
import sklearn
from sklearn.datasets.samples_generator import make_regression
import matplotlib.pyplot as plt
from scipy import stats

def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):
    converged = False
    k = 0 # Iterations counter.
    m = len(x) # number of samples

    # initial theta
    a_est = 6.5
    b_est = 1.7
    f_est = 1.5
    iterations = []
    J = []

    # Cost function J.
    iterations = np.append(iterations, k)
    J = np.append(J,sum([(a_est*exp(-x[i]/b_est)*cos(2*pi*f_est*x[i]) - y[i])**2 for i in range(m)])/(2*m))

    # Plot parameters convergence.
    plt.figure()
    plt.scatter(b_est, f_est, s=1, c='r')
    plt.xlabel('Decay value: b (s)')
    plt.ylabel('Frequency value: f (Hz)')
    plt.title('Parameters convergence')
    plt.xlim(0,3)
    plt.ylim(0,3)
    #plt.show()

    # Iterate Loop.
    while not converged:
        # For each training sample, compute the gradient.
        grad0 = 1.0/m * sum([(exp(-x[i]/b_est)*cos(2*pi*f_est*x[i])*(a_est*exp(-x[i]/b_est)*cos(2*pi*f_est*x[i])-y[i])) for i in range(m)])
        grad1 = 1.0/m * sum([(a_est*x[i]/(b_est**2)*exp(-x[i]/b_est)*cos(2*pi*f_est*x[i])*(a_est*exp(-x[i]/b_est)*cos(2*pi*f_est*x[i])-y[i])) for i in range(m)])
        grad2 = 1.0/m * sum([(2*pi*a_est*x[i]*exp(-x[i]/b_est)*sin(2*pi*f_est*x[i])*(y[i]-a_est*exp(-x[i]/b_est)*cos(2*pi*f_est*x[i]))) for i in range(m)])

        # Update estimated parameters
        a_est_temp = a_est - alpha*grad0
        b_est_temp = b_est - alpha*grad1
        f_est_temp = f_est - alpha*grad2

        a_est = a_est_temp
        b_est = b_est_temp
        f_est = f_est_temp

        #plt.plot(b_est, f_est)

        plt.scatter(b_est, f_est, s=1, c='r')
        #plt.show()
        #plt.pause(0.0001)
        # Mean squared error
        MSE = sum([(a_est*exp(-x[i]/b_est)*cos(2*pi*f_est*x[i]) - y[i])**2 for i in range(m)])/(2*m)

        k += 1
        if k%10 == 0:
            print("Iteration number: %s" % k)
        iterations = np.append(iterations, k)  # update iter
        J = np.append(J,MSE)

        if abs(J[k]-J[k-1]) <= ep:
            print('Converged, iterations: ', k, '!!!')
            converged = True

        if k == max_iter:
            print('Max interations exceeded!')
            converged = True

    return a_est, b_est, f_est, iterations, J

if __name__ == '__main__':
    x = np.arange(0,6,0.01)
    a = 7
    b = 2
    f = 2

    y = a*exp(-x/b)*cos(2*pi*f*x)

    # Random noise.
    noise = 0.5*np.random.randn(len(x))
    y_n = y + noise

    # Descent gradient parameters.
    alpha = 0.005 # learning rate
    ep = 0.001 # convergence criteria

    # Call gradient descent, get: a, tau, f
    start_time = time.time()
    a_est, b_est, f_est, iterations, J = gradient_descent(alpha, x, y_n, ep, max_iter=1000)
    end_time = time.time()
    print("Algorithm execution time: %s s" % (end_time - start_time))
    print("Average time per iteration: %s ms" % (1000*(end_time - start_time)/iterations[-1]))
    print("a = %s b = %s f = %s" % (a, b, f))
    print("a_est = %s b_est = %s f_est = %s" % (a_est, b_est, f_est))

    y_est = a_est*exp(-x/b_est)*cos(2*pi*f_est*x)

    # Plot.
    plt.figure()
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
