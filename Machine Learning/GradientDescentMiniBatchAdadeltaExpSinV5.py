#!/usr/bin/python3
# http://ruder.io/optimizing-gradient-descent/index.html#momentum
# Mini-batch gradient descent Adadelta algorithm.

import time
import numpy as np
from numpy import exp, cos, sin, pi
import matplotlib.pyplot as plt

def mini_batch_gradient_descent_adagrad(alpha, x, y, ep=0.0001, batch_size=50, max_iter=10000):
    epsilon = 0.000001
    beta = 0.9
    converged = False
    k = 0 # Iterations counter.
    iterations = []
    iterations = np.append(iterations, k)
    m = len(x) # number of samples
    idx = np.arange(0,m)
    idx_random = np.random.permutation(idx)#np.random.choice(idx,m,replace=False)
    batch_counter = 0

    # Initialize batch of data sets.
    x_batch = x[idx_random[batch_size*batch_counter:batch_size*(batch_counter+1)]]
    y_batch = y[idx_random[batch_size*batch_counter:batch_size*(batch_counter+1)]]

    # Initial paramters.
    a_est = [3]
    b_est = [1]
    f_est = [0.6]
    g_a = 0
    g_b = 0
    g_f = 0
    v_a = 0
    v_b = 0
    v_f = 0
    J = []

    # Cost function J.
    error = a_est[k]*exp(-x_batch/b_est[k])*cos(2*pi*f_est[k]*x_batch) - y_batch;
    MSE = np.dot(error,error)/(2*batch_size)
    J = np.append(J,MSE)

    # Iterate Loop.
    while not converged:
        # For each training sample, compute the gradient.
        grad_a = np.dot(exp(-x_batch/b_est[k])*cos(2*pi*f_est[k]*x_batch), error)/batch_size
        grad_b = np.dot(a_est[k]*x_batch/(b_est[k]**2)*exp(-x_batch/b_est[k])*cos(2*pi*f_est[k]*x_batch),error)/batch_size
        grad_f = np.dot(-2*pi*a_est[k]*x_batch*exp(-x_batch/b_est[k])*sin(2*pi*f_est[k]*x_batch),error)/batch_size

        # Update estimated parameters.
        g_a = beta*g_a + (1-beta)*grad_a**2
        g_b = beta*g_b + (1-beta)*grad_b**2
        g_f = beta*g_f + (1-beta)*grad_f**2

        delta_a = -np.sqrt(v_a + epsilon)/np.sqrt(g_a + epsilon)*grad_a
        delta_b = -np.sqrt(v_b + epsilon)/np.sqrt(g_b + epsilon)*grad_b
        delta_f = -np.sqrt(v_f + epsilon)/np.sqrt(g_f + epsilon)*grad_f

        v_a = beta*v_a + (1-beta)*delta_a**2
        v_b = beta*v_b + (1-beta)*delta_b**2
        v_f = beta*v_f + (1-beta)*delta_f**2

        a_est = np.append(a_est, a_est[k] + delta_a)
        b_est = np.append(b_est, b_est[k] + delta_b)
        f_est = np.append(f_est, f_est[k] + delta_f)

        error = a_est[k+1]*exp(-x_batch/b_est[k+1])*cos(2*pi*f_est[k+1]*x_batch) - y_batch;
        MSE = np.dot(error,error)/(2*batch_size)

        k += 1
        if k%10 == 0:
            print("Iteration number: %s" % k)
        iterations = np.append(iterations, k)  # update iter
        J = np.append(J,MSE)

        if abs(J[k]-J[k-1]) <= ep:
            print('Converged, iterations: ', k, '!!!')
            converged = True

        if k == max_iter:
            print('Max iterations exceeded!')
            converged = True

        # Update batch sets.
        batch_counter +=1
        if batch_size*(batch_counter+1)>m:
            batch_counter = 0
            idx_random = np.random.permutation(idx)#np.random.choice(idx,m,replace=False)
        x_batch = x[idx_random[batch_size*batch_counter:batch_size*(batch_counter+1)]]
        y_batch = y[idx_random[batch_size*batch_counter:batch_size*(batch_counter+1)]]
        error = a_est[k]*exp(-x_batch/b_est[k])*cos(2*pi*f_est[k]*x_batch) - y_batch;

    return a_est, b_est, f_est, iterations, J

if __name__ == '__main__':
    x = np.arange(0,6,0.01)
    a = 7
    b = 2
    f = 2
    y = a*exp(-x/b)*cos(2*pi*f*x)

    # Random noise.
    noise = 0.8*np.random.randn(len(x))
    y_n = y + noise

    # Descent gradient parameters.
    batch_size = 25
    alpha = 0.1 # learning rate
    ep = 0.00005 # convergence criteria
    max_iter = 8000

    # Call gradient descent, get: a, tau, f
    start_time = time.time()
    a_est, b_est, f_est, iterations, J = mini_batch_gradient_descent_adagrad(alpha, x, y_n, ep, batch_size, max_iter)
    end_time = time.time()
    print("Algorithm execution time: %s s" % (np.round(end_time - start_time,2)))
    print("Average time per iteration: %s ms" % (np.round(1000*(end_time - start_time)/iterations[-1],2)))
    print("a = %s b = %s f = %s" % (a, b, f))
    print("a_est = %s b_est = %s f_est = %s" % (np.round(a_est[-1],2), np.round(b_est[-1],2), np.round(f_est[-1],2)))
    print("Cost function J: %s" % (np.round(J[-1],6)))

    y_est = a_est[-1]*exp(-x/b_est[-1])*cos(2*pi*f_est[-1]*x)

    # Plot.
    plt.figure()
    plt.subplot(311)
    plt.plot(x, y, 'b-', label='Original signal')
    plt.scatter(x, y_n, s=4, c='r', marker='o', label='Noisy signal')
    plt.plot(x, y_est, 'k-', label='Estimated signal')
    plt.title('Oscilation with exponential decay')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(323)
    plt.plot(iterations,J)
    plt.title('Error vs Iterations')

    # Plot parameters convergence.
    #plt.figure()
    plt.subplot(324)
    plt.scatter(b_est, f_est, s=1, c='r')
    plt.xlabel('Decay value: b (s)')
    plt.ylabel('Frequency value: f (Hz)')
    plt.title('Parameters convergence')
    plt.xlim(0,3)
    plt.ylim(0,3)

    plt.subplot(325)
    plt.scatter(a_est, b_est, s=1, c='r')
    plt.xlabel('Amplitude value: a (m)')
    plt.ylabel('Decay value: b (s)')
    plt.xlim(2,8)
    plt.ylim(0,3)

    plt.subplot(326)
    plt.scatter(a_est, f_est, s=1, c='r')
    plt.xlabel('Amplitude value: a (m)')
    plt.ylabel('Frequency value: f (Hz)')
    plt.xlim(2,8)
    plt.ylim(0,3)
    plt.show()
    print("Done!")


#import random
#a = ['a', 'b', 'c']
#b = [1, 2, 3]
#c = list(zip(a, b))
#random.shuffle(c)
#a, b = zip(*c)
