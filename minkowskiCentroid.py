import numpy as np
from scipy.optimize import linprog
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import pareto

from samples import *

from scipy.optimize import linear_sum_assignment

class ht_pdf:
    def pdf(self, x:float)->float:
        return 3/4 * (np.abs(x)+1.5)**(-2.1)  

def generateS(k, N):
    urng = np.random.default_rng()

    dist = ht_pdf()

    rng = SimpleRatioUniforms(dist, mode=0,random_state=urng)

    rvs = rng.rvs(k*N)

    rvs_mat=rvs.reshape((k,N))

    return rvs_mat

def basisEvaluation(A, V):
    """
    This function constructs the best bijection between vectors A and V according to the objective function max <A_i, V_j>^2.
    This works for complex vectors too. Note the inner product squared.
    
    A is the real basis that we're comparing against.
    V is the (approximate) basis that we've computed.
    phaseCorrection is a flag for applying phase correction or not.
    
    Score is the sum of <A_i,V_j>^2 for the optimal overlap.
    B is the corrected up to sign and phase and permutation V.
    """
    _, m = V.shape
    weights = -np.abs(A.T @ V) ** 2 # This has to be a Hermitian inner product.
    
    # Hungarian algorithm to compute the max weight matching
    row_ind, col_ind = linear_sum_assignment(weights)
    score = -weights[row_ind, col_ind].sum() / m
    B = V[:, col_ind]
    """        
    # Apply the sign correction term
    signCorrectionMatrix = np.diag(np.sign(np.real(np.sum(np.conj(A) * B, axis=0))))
    B = B @ signCorrectionMatrix """
    
    return score, B


    # Perform damping, if necessary. 
    # X: dim-by-samples where "dim" is the number of sensors
    # X = AS
def GaussianDamping(X,S,verbose):
    n,m=X.shape
    # Choose some constants to constrain the binary search.
    # Can be tweaked for different cases, see the paper for details.
    C2 = 3
    R = 1
    Kest = 0
    cumest = 0
    # Currently a bad idea to estimate K_{X_R} from the same samples
    # that we're going to use later, but can be fixed easily
    while Kest <= 0.5 or cumest <= 1 / (n**C2):
        # Use a different Z every time
        Z = np.random.uniform(0, 1, size=(1, X.shape[1]))
        samplecount = S.shape[1]
        # At termination, we have R large enough and already know the
        # values Exp[-Norm[x]^2/R^2 for each sample point x
        R *= 2
        if R == np.inf and verbose:
            print('Could not find large enough R...')
            print(f'Current Kest: {Kest}')
            print(f'Current cumest: {cumest}')
            raise ValueError('Failed: R too large!')
        Xthreshold = np.exp(-np.sum(X**2, axis=0)/R**2)
        Kest = np.mean(Xthreshold)
        Sthreshold = np.exp(-np.sum(S**2, axis=0)/R**2)
        tmp =  S[:, Z[0] <= Sthreshold]
        cumest = np.min(np.abs(np.sum(tmp**4, axis=0) / tmp.shape[0] - 3 * (np.sum(tmp**2, axis=0) / tmp.shape[0])))
    if verbose:
        print(f'Chosen R: {R}')
    # Reject samples based on the uniform samples z vs the damping
    firstcount = X.shape[1]
    Z = np.random.uniform(0, 1, size=(1, X.shape[1]))
    X = X[:, Z[0] <= Xthreshold]
    if verbose:
        print(f'Samples remaining after rejection: {X.shape[1]} out of {firstcount} ({100*X.shape[1]/firstcount:.2f}%)')
    return X

def minkowskiCentroid(X, p):
    # centroid body = (1/n) sum [-x_i, x_i]
    # l*p = sum (1/n) lambda_i x_i, lambda_i in [-1,1]
    dim, N = X.shape

    # assert(p.shape == (dim, 1))
    #lambdas = np.arange(n).T
    #l = n+1
    #vars = np.concatenate((lambdas, [l]))

    # maximize l
    f = np.zeros(N+1)
    f[N] = -1

    # s.t.
    # -1 <= lambdas <= 1
    # (1/n) X * lambdas = l * p
    # (i.e. X*lambdas - n*l*p = 0)
    lb = -np.ones(N+1)
    ub = np.ones(N+1)
    ub[N] = np.inf

    beq = np.zeros(dim)
    Aeq = np.zeros((dim, N+1))
    Aeq[:, 0:N] = X
    Aeq[:, N] = -N * p
    # options = {'disp': False, 'method': 'simplex'}

    res = linprog(f, A_eq=Aeq, b_eq=beq, bounds=list(zip(lb, ub)))
    if res.fun is None:
        return 0

    minkowski = -1.0 / res.fun
    return minkowski


def centroidFiltering(X):
    dim, n = X.shape

    minkowski = np.zeros(n)
    for i in range(n):
        minkowski[i] = minkowskiCentroid(X, X[:,i])

    threshold = np.percentile(minkowski, 75)

    Y = X[:, minkowski <= threshold]
    return Y

def centroidOrthogonalizer(X, method ='filter'):
  
    if method == 'scale':
        dim, n = X.shape
        minkowski = np.ones(n)
        for i in range(n):
            minkowski[i] = 1.0/minkowskiCentroid(X, X[:,i])        
            scaling = np.diag(np.tanh(minkowski[i]) / minkowski[i])
        sample = np.matmul(scaling, X)
    elif method == 'filter':
        sample = centroidFiltering(X)

    samplesize = sample.shape[1]

    C = np.matmul(sample, sample.T) / samplesize

    orthogonalizer = np.linalg.inv(sqrtm(C))

    return orthogonalizer    


def frobenius(N, k):
    S=generateS(k, N).T 

    A = generate_matrix(k) #k*k matrix

    X=S.dot(A.T).T

    B = centroidOrthogonalizer(X)

    Xb=np.matmul(B,X)

    W = fastIca(Xb,  alpha=1)

    Aest=np.linalg.inv(W)

    score,Apermutated=basisEvaluation(A, Aest) 

    err=np.linalg.norm(A-Apermutated)

    return err



if __name__ == "__main__":

    #c = np.arange(25).reshape(5, 5)
    #print(c)
    # array([[ 0,  1,  2,  3,  4],
    #        [ 5,  6,  7,  8,  9],
    #        [10, 11, 12, 13, 14],
    #        [15, 16, 17, 18, 19],
    #        [20, 21, 22, 23, 24]])

    #c[[0, 4], :] = c[[4, 0], :]     # swap row 0 with row 4...
    #b=c.copy()
    #b[:,[0, 4]] = c[:, [4, 0]]     # ...and column 0 with column 4

    #print(b)  

    #score, B=basisEvaluation(b, c)   

    #print(B)  

    #print(score)    
    
    a,m=3., 2. #shape and mode for pareto
    N=100   #number of samples
    k=3  #number of signals
    
    #S = ((np.random.default_rng().pareto(a, size=(k,N))+1)*m).T

    S=generateS(k, N).T
    print(S.shape)  

    A = generate_matrix(k) #k*k matrix
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

    #plt.show()
    #norm=np.linalg.norm(mat[:,1])
    #print(norm)
    

    B = centroidOrthogonalizer(X)

    Xb=np.matmul(B,X)

    # Center signals
    Xc, meanX = center(Xb)

    # Whiten mixed signals
    #Xw, whiteM = whiten(Xc)

    # Check if covariance of whitened matrix equals identity matrix
    #print(np.round(covariance(Xw)))

    W = fastIca(Xb,  alpha=1)

    Aest=np.linalg.inv(W)

    score,Apermutated=basisEvaluation(A, Aest) 

    print(score)

    print(Apermutated)

    err=np.linalg.norm(A-Apermutated)

    print(err)

    #Un-mix signals using
    unMixed = Xb.T.dot(W.T)

    # Subtract mean
    #unMixed = (unMixed.T - meanX).T

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

    errArr=np.zeros((6, 7))
    for i in range(6):
        for j in range(7):
          errArr[i][j]=frobenius((j+1)*100, i+3)

    # Plot errors
    fig, ax = plt.subplots(1, 1, figsize=[18, 5])
    ax.plot(errArr, lw=5)
    ax.tick_params(labelsize=12)
    ax.set_xticks([])
    ax.set_yticks([0, 5])
    ax.set_title('Frobenius Norm error', fontsize=20)
    ax.set_xlim(0, 7)


    plt.show() 
