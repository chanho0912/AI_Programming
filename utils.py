import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from YourAnswer import gaussian_kernel, polynomial_kernel 

# visualize data 

def plot_svc_with(pred_ob, X, y, h=0.02, pad=0.25, plot_support_vector=False, plot_mis=False, sklearn=True, score_func=None, w=None):
    plt.figure(figsize=(12,8))
#     x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
#     y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    x_min = -4
    x_max = 4
    y_min = -4
    y_max = 4
    xx, yy = np.meshgrid(np.arange(x_min-pad, x_max+pad, h), np.arange(y_min-pad, y_max+pad, h))
    
    if sklearn:
        Z = pred_ob.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = pred_ob(np.c_[xx.ravel(), yy.ravel()], w)
    Z = Z.reshape(xx.shape)    
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)
    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    
 
    x_plot = np.array([x_max, x_min])
    y_plot = (-w[0] - w[1]*x_plot) / (w[2])
    plt.plot(x_plot, y_plot, 'k-', label='Linear SVM separating hyperplane')


    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()
                    

def plot_svc(pred_ob, X, y, h=0.02, pad=0.25, plot_support_vector=False, plot_mis=False, sklearn=True, score_func=None, w=None):
    plt.figure(figsize=(12,8))
#     x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
#     y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    x_min = -4
    x_max = 4
    y_min = -4
    y_max = 4
    xx, yy = np.meshgrid(np.arange(x_min-pad, x_max+pad, h), np.arange(y_min-pad, y_max+pad, h))
    
    if sklearn:
        Z = pred_ob.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = pred_ob(np.c_[xx.ravel(), yy.ravel()], w)
    Z = Z.reshape(xx.shape)    
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2) 
    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    
    # Support vectors indicated in plot by vertical lines
    if plot_support_vector:
        if sklearn:
            sv = pred_ob.support_vectors_
        else:
            mask = (1 - score_func(X, y, w)) >= 0.
            sv = X[mask]
            
            x_plot = np.array([x_max, x_min])
            y_plot = (1-w[0] - w[1]*x_plot) / (w[2])
            
            plt.plot(x_plot, y_plot, 'k--', label='marginal hyperplane for 1')
            
            x_plot = np.array([x_max, x_min])
            y_plot = (-1-w[0] - w[1]*x_plot) / (w[2])
            plt.plot(x_plot, y_plot, 'k-.', label='marginal hyperplane for -1')
            
        plt.scatter(sv[:,0], sv[:,1], c='k', marker='|', s=100, linewidths=1)
        num_sv = pred_ob.support_.size if sklearn else len(sv)
        print('Number of support vectors: ', num_sv)

        
    if plot_mis:
        pred = pred_ob.predict(X) if sklearn else pred_ob(X, w)
        label = y != pred
        print('Misclassified data: ', sum(label))
        plt.plot(X[label,0], X[label,1], 'rx')
        
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
                    
                    
def plot_all(svm_weight, logistic, X, y, h=0.02, pad=0.25):
    plt.figure(figsize=(12,8))
#     x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
#     y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    x_min = -4.
    x_max = 4.
    y_min = -4.
    y_max = 4.
    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    
    #svm autograd
    x = np.array([x_max, x_min])
    y = (-svm_weight[0]-svm_weight[1]*x) / svm_weight[2]
    plt.plot(x, y, 'k:', label='SVM')
    
    #logistic
    coef = logistic.coef_
    bias = logistic.intercept_
    x_logi = np.array([x_max, x_min])
    y_logi = (-bias-coef[:,0]*x) / coef[:,1]
    plt.plot(x_logi, y_logi, 'k', label='Logistic regression')
        
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()
    
    
def vis_data(x, y = None, c='r', title=None):
    plt.figure(figsize=(8, 5))
    plt.title(title)
    x_min = -4.
    x_max = 4.
    y_min = -4.
    y_max = 4.
    if y is None: 
        y = [None] * len(x)
    for x_, y_ in zip(x, y):
        if y_ is None:
            plt.plot(x_[0], x_[1], 'o', markerfacecolor='none', markeredgecolor=c)
        elif(y_==-1):
            plt.plot(x_[0], x_[1], 'o', markerfacecolor='none', markeredgecolor='b')
        elif(y_==1):
            plt.plot(x_[0], x_[1], '+', markerfacecolor='none', markeredgecolor=c)
        else:
            plt.plot(x_[0], x_[1], 'v', markerfacecolor='none', markeredgecolor='g')
            
    plt.grid(True)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    return


def gen_multimodal():
    x = np.random.multivariate_normal([-1,-1], [[1,0],[0,1]], 100)
    x = np.concatenate((x, np.random.multivariate_normal([1,1], [[1,0],[0,1]], 100)))
    x = np.concatenate((x, np.random.multivariate_normal([3,3], [[1,0],[0,1]], 50)))
    y = np.ones(250)
    y[100:200] = -1
    return x, y

def poly_proxy_kernel(degree, bias ):
    def proxy_kernel(X, Y, K=polynomial_kernel):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                gram_matrix[i, j] = K(x, y, degree,bias)
        return gram_matrix    
    return proxy_kernel

def gaussian_proxy_kernel(sigma):
    def proxy_kernel(X, Y, K=gaussian_kernel):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                gram_matrix[i, j] = K(x, y,sigma)
        return gram_matrix        
    return proxy_kernel    

