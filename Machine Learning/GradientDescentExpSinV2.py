#!/usr/bin/python3
# Mini-batch gradient descent algorithm.

import time
import numpy as np
from numpy import exp, cos, sin, pi
import sklearn
from sklearn.datasets.samples_generator import make_regression
import matplotlib.pyplot as plt
from scipy import stats

def mini_batch_gradient_descent(alpha, beta, x, y, ep=0.0001, batch_size=30, max_iter=10000):
    converged = False
    k = 0 # Iterations counter.
    iterations = []
    iterations = np.append(iterations, k)
    m = len(x) # number of samples

    # initial theta
    a_est = 6.5
    b_est = 1.6
    f_est = 1.4
    v_a = 0
    v_b = 0
    v_f = 0
    J = []

    # Cost function J.
    error = a_est*exp(-x/b_est)*cos(2*pi*f_est*x) - y;
    MSE = np.dot(error,error)/(2*m)
    J = np.append(J,MSE)

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
        grad0 = np.dot(exp(-x/b_est)*cos(2*pi*f_est*x), error)/m
        grad1 = np.dot(a_est*x/(b_est**2)*exp(-x/b_est)*cos(2*pi*f_est*x),error)/m
        grad2 = np.dot(-2*pi*a_est*x*exp(-x/b_est)*sin(2*pi*f_est*x),error)/m
        # Update estimated parameters.
        v_a = beta*v_a + alpha*grad0
        v_b = beta*v_b + alpha*grad1
        v_f = beta*v_f + alpha*grad2

        a_est = a_est - v_a
        b_est = b_est - v_b
        f_est = f_est - v_f

        #plt.plot(b_est, f_est)

        plt.scatter(b_est, f_est, s=1, c='r')
        #plt.show()
        #plt.pause(0.0001)
        # Mean squared error
        error = a_est*exp(-x/b_est)*cos(2*pi*f_est*x) - y;
        MSE = np.dot(error,error)/(2*m)

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
    batch_size = 60
    alpha = 0.004 # learning rate
    beta = 0.93
    ep = 0.0002 # convergence criteria

    # Call gradient descent, get: a, tau, f
    start_time = time.time()
    a_est, b_est, f_est, iterations, J = mini_batch_gradient_descent(alpha, beta, x, y_n, ep, batch_size, max_iter=1000)
    end_time = time.time()
    print("Algorithm execution time: %s s" % (end_time - start_time))
    print("Average time per iteration: %s ms" % (1000*(end_time - start_time)/iterations[-1]))
    print("a = %s b = %s f = %s" % (a, b, f))
    print("a_est = %s b_est = %s f_est = %s" % (np.round(a_est,3), np.round(b_est,3), np.round(f_est,3)))
    print("Cost function J: %s" % (np.round(J[-1],3)))

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
