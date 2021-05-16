import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
#-*- coding: utf-8 -*-


def score_function(X, y, theta):
    '''
    INPUT: Feature vector (X) , class (y), and weight vector (theta)
    Dimension:
    X: N*(d-1)
    theta: d
    y: N
    OUTPUT: The score.
    '''
    score = np.zeros_like(y)

    score = y * (np.dot(X, theta[1:]) + theta[0])

    return score


def prediction_function(X, theta):
    '''
    INPUT: Feature vector (X), and weight vector (theta)
    Dimension:
    X: N*(d-1)
    theta: d
    OUTPUT: The prediction.
    '''
    prediction = np.zeros(X.shape[0])
    
    theta_t_X = np.dot(X, theta[1:]) + theta[0]
    for i in range(len(prediction)):
        prediction[i] = 1 if theta_t_X[i] >= 0 else -1

    return prediction


def hinge_loss(X, y, theta):
    '''
    INPUT: Feature vector (X) , class (y), and weight vector (theta)
    Dimension:
    X: N*(d-1)
    theta: d
    y: N
    OUTPUT: Hinge loss vector.
    '''
    loss = np.zeros(X.shape[0])
    score = score_function(X, y, theta)
    for i in range(len(score)):
        loss[i] = 1 - score[i] if 1 - score[i] >= 0 else 0


    return loss


def objective_function(theta, X, y, C):
    '''
    Objective function. 

    INPUT: Feature vector (X) , class (y), weight vector (theta) and constant (C)
    Dimension:
    X: N*(d-1)
    theta: d
    y: N
    OUTPUT: Objective function value.
    '''
    obj = 0
    
    theta_l2_norm = np.sum(theta[1] ** 2)
    obj = (1 / 2) * theta_l2_norm + C * np.sum(hinge_loss(X, y, theta))
    
    return obj


def update_svm(theta, X, y, C, num_iters=1000, alpha=0.00001, log_term = 20,print_log=True):
    ''' 
    INPUT: Feature vector (X) , class (y), weight vector (theta) and constant (C)
           number of updates (num_iters), learning rate(alpha)
    Dimension:
    X: N*(d-1)
    theta: d
    y: N
    OUTPUT: Updated theta
    '''
    cost = 0
    updated_theta = np.copy(theta)

    for i in range(num_iters):
        theta1_grad = 0
        theta0_grad = 0
        loss = hinge_loss(X, y, updated_theta)

        for j in range(len(loss)):
            theta1_grad += (C * y[j] * X[j]) if loss[j] > 0 else 0
            theta0_grad += (C * y[j]) if loss[j] > 0 else 0

        updated_theta[1:] = updated_theta[1:] + alpha * theta1_grad
        updated_theta[0] = updated_theta[0] + alpha * theta0_grad
        if print_log:
            if (i+1) % log_term ==0:
                cost = objective_function(updated_theta, X, y, C)            
                print('Iter [{}] - Cost : {:.4f}'.format(i+1, cost))
    print('Done')
    
    return updated_theta

def polynomial_kernel(x, z,degree,bias):
    K = 0. # You should return this kernel function correctly
    K = (bias + np.dot(x, z)) ** degree
    return K
    
def gaussian_kernel(x, z,sigma):   
    K = 0. # You should return this kernel function correctly
    K = np.exp(-np.linalg.norm(x-z)**2 / (2 * (sigma ** 2)))
    return K

def gridsearch(parameters, X, y):
    clf = None 
    clf = GridSearchCV(estimator=SVC(kernel='rbf'), param_grid=parameters, cv=10)
    clf.fit(X=X, y=y)
    # =================================================================  
    return clf
