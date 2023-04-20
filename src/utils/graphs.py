import os
import numpy as np
import scipy.sparse as ss
import scipy

def degree_matrix(A, sparse=True):
    """Returns the absolute degree matrix of a signed graph
    Args: A (csc or dense matrix): signed adjacency matrix
          sparse (boolean): sparse or dense matrix input and output"""

    if not sparse:
        return np.diag(abs(A).sum(axis=0), 0)

    else:
        return ss.diags(np.array(abs(A).sum(axis=0)).squeeze(), offsets=0).tocsc()

def identity(n, sparse=True):
    """
    Returns identity matrix of size n in sparse (ss.csr) or non-sparse (np.ndarray) format

    :param n (int):
    :param sparse (bool):
    :return:
    """

    if not sparse:
        return np.eye(n)
    else:
        return ss.eye(n)


def laplacian(A, sparse=True):
    """Returns the Laplacian matrix of a graph
    Args: A (csc or dense matrix): adjacency matrix
          sparse (boolean): sparse or dense matrix output"""

    return degree_matrix(A, sparse) - A


def norm_laplacian(A, sparse=True):
    """Returns the symmetric normalized Laplacian matrix
    Args: A (csc or dense matrix): adjacency matrix
          sparse (boolean): sparse or dense matrix output"""

    D = degree_matrix(np.abs(A), sparse)

    if not sparse:
        diag = np.diag(D)
        if diag.any() == 0.:
            print("Regularising graph with isolated nodes")
            diag = diag + 0.01 * np.ones(A.shape[0])
        Dinv = np.diag( 1.0 / np.sqrt(diag))
        #Dinv = np.linalg.inv(np.sqrt(D))
        return identity(A.shape[0]) - (Dinv.dot( A ).dot( Dinv ))

    else:
        diag = D.diagonal()
        if diag.any() == 0.:
            print("Regularising graph with isolated nodes")
            diag = diag + 0.01 * np.ones(A.shape[0])
        Dinv = ss.diags(1.0 / np.sqrt(diag))
        #Dinv = D.power(-0.5)
        return identity(A.shape[0]) - (Dinv @ A @ Dinv)




def laplacian_embeddings(A, k=None, sparse=True, normalize=True):
    """
    Returns the (sym) Laplacian embeddings matrix of dimension k. If the graph is signed, uses the signed Laplacian
    Args: A (csc or dense matrix): adjacency matrix
          sparse (boolean): sparse or dense matrix output
    output: N x k matrix
    """

    if normalize:
        L = norm_laplacian(A, sparse)
    else:
        L = laplacian(A, sparse)
    if k is None:
        k = A.shape[0] // 2
    if sparse:
        _, V = ss.linalg.eigsh(L, k=k, which='SA')
    else:
        _, V = scipy.linalg.eigh(L, subset_by_index=[0,k-1])

    return V

def random_walk_embeddings(A, k=None, sparse=True):
    """
    Returns the (sym) Laplacian embeddings matrix of dimension k. If the graph is signed, uses the signed Laplacian
    Args: A (csc or dense matrix): adjacency matrix
          sparse (boolean): sparse or dense matrix output
    output: N x k matrix
    """

    if k is None:
        k = 1
    P = np.zeros((A.shape[0], k))
    R = np.dot(degree_matrix(A, sparse), A)
    for i in range(1, k+1):
        Q = np.linalg.matrix_power(a=R.todense(), n=i)
        P[:,i-1] = np.diag(Q, k=0)

    return P