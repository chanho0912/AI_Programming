import autograd.numpy as np
from autograd import grad
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
    # ====================== YOUR CODE HERE ===========================
    # Instructions: Implement the score function
    #               Please consider X and theta having arbitary dimension 
    #               (not speicific fixed dimension)
    #               when you implement.

    
    # =================================================================  

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
    # ====================== YOUR CODE HERE ===========================
    # Instructions : Implement the prediction function
    #               Please consider X and theta having arbitary dimension 
    #               (not speicific fixed dimension)
    #               when you implement.
    
    
    
 
    
    # =================================================================  

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
    # ====================== YOUR CODE HERE ===========================
    # Instructions : compute the hinge loss
    #                Using the score function you implemented,
    #               Please consider X and theta having arbitary dimension 
    #               (not speicific fixed dimension)
    #               when you implement.
    
    
    

    # =================================================================  

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
    # ====================== YOUR CODE HERE ===========================
    # Instructions : Compute the objective function value 
    #                Using the hinge loss you implemented would be helpful
    #                
    

    # =================================================================  
    
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
    # ====================== YOUR CODE HERE ===========================
    # Instructions : Compute the graident of linear SVM and 
    #                update theta by batch gradient descent
    #               Please consider X and theta having arbitary dimension 
    #               (not speicific fixed dimension)
    #               when you implement.
    #
    # Hint) Do not forget hinge loss gives gradient 0 when loss < 0
    #   

        
    # =================================================================  
        if print_log:
            if (i+1) % log_term ==0:
                cost = objective_function(updated_theta, X, y, C)            
                print('Iter [{}] - Cost : {:.4f}'.format(i+1, cost))
  
    print('Done')
    
    return updated_theta

def polynomial_kernel(x, z,degree,bias):
    K = 0. # You should return this kernel function correctly
    # ==================== YOUR CODE HERE ====================

    # ========================================================
    return K
    
def gaussian_kernel(x, z,sigma):   
    K = 0. # You should return this kernel function correctly
    # ==================== YOUR CODE HERE ====================

    # ========================================================
    return K

def gridsearch(parameters, X, y):
    clf = None 
    # ====================== YOUR CODE HERE ===========================
    # Instructions: Use GridSearchCV Function(Only use SVC(kernel='rbf') estimator) 
    #               in scikit-learn package to maximize accuracy!!.
    #               Set the number of folds to 10
    #               You should return the clf(classifier) correctly after fitting the data
    #
    # Hint) https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    
    # =================================================================  
    return clf
