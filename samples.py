#!/usr/bin/env python3
#https://github.com/akcarsten/Independent_Component_Analysis
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import pareto
from scipy.stats import rv_continuous
from scipy.stats.sampling import SimpleRatioUniforms

from sympy import *
from sympy.stats import ContinuousRV, P


class my_pdf(rv_continuous):
    def _pdf(self, x:float)->float:
        return 3/4 * (np.abs(x)+1.5)**(-2.1)

class ht_pdf:
    def pdf(self, x:float)->float:
        return 3/4 * (np.abs(x)+1.5)**(-2.1)        

def generate_matrix(n):
    # Generate a matrix with entries sampled from Gaussian distribution
    mat = np.random.normal(loc=0, scale=1, size=(n, n))
    
    # Normalize each column to a unit vector
    norm = np.linalg.norm(mat, axis=0)
    mat_normalized = mat / norm[np.newaxis, :]
    
    return mat_normalized

def center(x):
    mean = np.mean(x, axis=1, keepdims=True)
    centered =  x - mean
    return centered, mean

def covariance(x):
    mean = np.mean(x, axis=1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x - mean

    return (m.dot(m.T))/n    

def whiten(x):
    # Calculate the covariance matrix
    coVarM = covariance(X)

    # Single value decoposition
    U, S, V = np.linalg.svd(coVarM)

    # Calculate diagonal matrix of eigenvalues
    d = np.diag(1.0 / np.sqrt(S))

    # Calculate whitening matrix
    whiteM = np.dot(U, np.dot(d, U.T))

    # Project onto whitening matrix
    Xw = np.dot(whiteM, X)

    return Xw, whiteM
    
#https://github.com/akcarsten/Independent_Component_Analysis
def fastIca(signals,  alpha = 1, thresh=1e-8, iterations=5000):
    m, n = signals.shape

    # Initialize random weights
    W = np.random.rand(m, m)

    for c in range(m):
            w = W[c, :].copy().reshape(m, 1)
            w = w / np.sqrt((w ** 2).sum())

            i = 0
            lim = 100
            while ((lim > thresh) & (i < iterations)):

                # Dot product of weight and signal
                ws = np.dot(w.T, signals)

                # Pass w*s into contrast function g
                wg = np.tanh(ws * alpha).T

                # Pass w*s into g prime
                wg_ = (1 - np.square(np.tanh(ws))) * alpha

                # Update weights
                wNew = (signals * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()

                # Decorrelate weights              
                wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
                wNew = wNew / np.sqrt((wNew ** 2).sum())

                # Calculate limit condition
                lim = np.abs(np.abs((wNew * w).sum()) - 1)

                # Update weights
                w = wNew

                # Update counter
                i += 1

            W[c, :] = w.T
    return W 

if __name__ == "__main__":


    #my_cv=my_pdf(name='my_pdf')
    urng = np.random.default_rng()

    dist = ht_pdf()

    rng = SimpleRatioUniforms(dist, mode=0,random_state=urng)

    rvs = rng.rvs(300)

    #rvs_mat=rvs.reshape((3,1000))

    #print(rvs_mat.shape)

    x = np.linspace(rvs.min()-0.1, rvs.max()+0.1, num=100)

    fx = dist.pdf(x)

    plt.plot(x, fx, 'r-', lw=2, label='true distribution')

    plt.hist(rvs, bins=20, density=True, alpha=0.8, label='random variates')

    plt.xlabel('x')
    #plt.ylim(0, 1)

    plt.ylabel('PDF(x)')

    plt.title('Simple Ratio Uniforms Samples')

    plt.legend()

    plt.show()
  
  #FastICA
    a,m=3., 2. #shape and mode
    N=100   #number of samples
    k=3  #number of signals
    S = ((np.random.default_rng().pareto(a, size=(k,N))+1)*m).T

    print(S.shape)  

    A = generate_matrix(k)
    #print(mat) 
    X=S.dot(A.T).T
    #X=np.matmul(A,S)
    #print(X)

    # Number of samples
    ns = np.linspace(0, N, num=N)
    # Plot sources & signals
    fig, ax = plt.subplots(1, 1, figsize=[18, 5])
    ax.plot(ns, S, lw=5)
    ax.set_xticks([])
    ax.set_yticks([0, 10])
    ax.set_xlim(0, N)
    ax.tick_params(labelsize=12)
    ax.set_title('Independent sources', fontsize=25)

    fig, ax = plt.subplots(3, 1, figsize=[18, 5], sharex=True)
    ax[0].plot(ns, X[0], lw=5)
    ax[0].set_title('Mixed signals', fontsize=25)
    ax[0].tick_params(labelsize=12)

    ax[1].plot(ns, X[1], lw=5)
    ax[1].tick_params(labelsize=12)
    ax[1].set_xlim(0, N)

    ax[2].plot(ns, X[2], lw=5)
    ax[2].tick_params(labelsize=12)
    ax[2].set_xlim(0, N)
    ax[2].set_xlabel('Sample number', fontsize=20)

    plt.show()
    #norm=np.linalg.norm(mat[:,1])
    #print(norm)


    # Center signals
    Xc, meanX = center(X)

    # Whiten mixed signals
    Xw, whiteM = whiten(Xc)

    # Check if covariance of whitened matrix equals identity matrix
    print(np.round(covariance(Xw)))

    W = fastIca(Xw,  alpha=1)

    #Un-mix signals using
    unMixed = Xw.T.dot(W.T)

    # Subtract mean
    unMixed = (unMixed.T - meanX).T

    # Plot input signals (not mixed)
    fig, ax = plt.subplots(1, 1, figsize=[18, 5])
    ax.plot(S, lw=5)
    ax.tick_params(labelsize=12)
    ax.set_xticks([])
    ax.set_yticks([0, 5])
    ax.set_title('Source signals', fontsize=25)
    ax.set_xlim(0, N)

    fig, ax = plt.subplots(1, 1, figsize=[18, 5])
    ax.plot(unMixed, '--', label='Recovered signals', lw=5)
    ax.set_xlabel('Sample number', fontsize=20)
    ax.set_title('Recovered signals', fontsize=25)
    ax.set_xlim(0, N)

    plt.show() 

    x = Symbol('x')
    X = ContinuousRV(x, 3/4 * (np.abs(x)+1.5)**(-2.1), Interval(-10, 10))
    print(P(X>0.5))
    from sklearn.decomposition import FastICA

    # Compute ICA
    ica = FastICA(n_components=3, whiten="unit-variance", max_iter=50000, tol=0.0001)
    S_ = ica.fit_transform(X)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix
    
    print(A_)
    # We can `prove` that the ICA model applies by reverting the unmixing.
    assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)
