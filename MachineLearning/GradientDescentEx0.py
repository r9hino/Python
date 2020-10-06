#!/usr/bin/python3
#https://www.bogotobogo.com/python/python_numpy_batch_gradient_descent_algorithm.php

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
    t0 = np.random.random(x.shape[1])
    t1 = np.random.random(x.shape[1])
    J = []

    # total error, J(theta)
    J = np.append(J,sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])/(2*m))

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
    x, y = make_regression(n_samples=100, n_features=1,
        n_informative=1, random_state=0, noise=35)
    print('x.shape = %s y.shape = %s' % (x.shape, y.shape))

    alpha = 0.01 # learning rate
    ep = 0.01 # convergence criteria

    # call gredient decent, and get intercept(=theta0) and slope(=theta1)
    theta0, theta1, J = gradient_descent(alpha, x, y, ep, max_iter=1000)
    iterations = np.arange(len(J))
    print("theta0 = %s theta1 = %s" % (theta0, theta1))

    # check with scipy linear regression
    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x[:,0], y)
    print("intercept = %s slope = %s" % (intercept, slope))

    # plot
    for i in range(x.shape[0]):
        y_predict = theta0 + theta1*x
    plt.subplot(2,1,1)
    plt.plot(x,y,'o')
    plt.plot(x,y_predict,'k-')
    plt.title('Regression Result')

    plt.subplot(2,1,2)
    plt.plot(iterations,J)
    plt.title('Error vs Iterations')
    plt.show()
    print("Done!")
