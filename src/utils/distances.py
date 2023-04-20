import numpy as np
import scipy.sparse as ss
import grakel
from grakel.kernels import WeisfeilerLehman, VertexHistogram, ShortestPath, WeisfeilerLehmanOptimalAssignment, GraphletSampling, MultiscaleLaplacian
from src.utils.graphs import laplacian_embeddings
from netrd.distance import DeltaCon
import networkx as nx

# Graph distances

def distance_procrustes_LE(A, B, k = 10, normalize=True):
    """
    Compute Procrustes distance between the principal subspaces of 2 adjacency matrices
    """
    sparse = False
    if isinstance(A, (ss.csc_matrix, ss.coo_matrix, ss.csr_matrix)):
        sparse = True
    LE1 = laplacian_embeddings(A, k=k, sparse=sparse, normalize=normalize)

    sparse = False
    if isinstance(B, (ss.csc_matrix, ss.coo_matrix, ss.csr_matrix)):
        sparse = True
    LE2 = laplacian_embeddings(B, k=k, sparse=sparse, normalize=normalize)

    dist = distance_procrustes(LE1, LE2)

    return dist


def DeltaConDistance(A, B):
    """
    Compute Delta Connectivity distance between 2 adjancency matrices
    """

    assert type(A) == np.ndarray
    assert type(A) == type(B)

    G1, G2 = nx.from_numpy_array(A), nx.from_numpy_array(B)
    metric = DeltaCon()
    d1 = metric.dist(G1=G1, G2=G2)
    return (d1)


def distance_frobenius(A, B):
    """
    Compute "Frobenius" distance between 2 matrices
    """

    assert type(A) == type(B)

    if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        dist = np.linalg.norm(A - B) ** 2 / (np.linalg.norm(A) * np.linalg.norm(B))

    elif isinstance(A, (ss.csc_matrix, ss.coo_matrix, ss.csr_matrix)) \
            and isinstance(B, (ss.csc_matrix, ss.coo_matrix, ss.csr_matrix)):
        dist = ss.linalg.norm(A - B) ** 2 / (ss.linalg.norm(A) * ss.linalg.norm(B))

    else:
        print(type(A), type(B))
        raise ValueError('Type not recognised')

    return dist


def distance_procrustes(A, B):
    """
    Compute "Procrustes" distance between 2 matrices
    """

    SD = SubspaceDistance(metric='Procrustes')
    SD.fit(A, B)
    dist = SD.distance_

    return dist


def WL_distance(A, B, n_iter=3):
    """
    Computes inverse of WL kernel
    """

    nodes_attributes = {}
    for n in range(A.shape[0]):
        nodes_attributes[n] = 1  # sbm.features[i][n,:].tolist()
    G1 = grakel.Graph(A, node_labels=nodes_attributes)
    G2 = grakel.Graph(B, node_labels=nodes_attributes)
    wl_kernel = WeisfeilerLehman(n_iter=n_iter, normalize=True, base_graph_kernel=VertexHistogram)
    K = wl_kernel.fit_transform([G1, G2])

    return (1.0 / (1 + K[0,1]))


# Subspace distances

def projection(theta):
    """
    Compute Projection distance between two equi-dimensional subspaces.

    Parameters:
    - - - - -
    theta: float, array
    """

    return np.sqrt((np.sin(theta) ** 2).sum())

def binetcauchy(theta):
    """
    Compute Binet-Cauchy distance between two equi-dimensional subspaces.

    Parameters:
    - - - - -
    theta: float, array
    """

    return np.sqrt((1 - np.prod(np.cos(theta) ** 2)))


def procrustes(theta):
    """
    Compute Procrustes distance between two equi-dimensional subspaces.

    Parameters:
    - - - - -
    theta: float, array
        arccos(singular values) of inner product of two equi-dimensional subspaces
    """

    return 2 * np.sqrt((np.sin(theta / 2) ** 2).sum())


class SubspaceDistance(object):
    """
    Class to compute a variety of distance between subspaces of equal dimension.
    """

    def __init__(self, metric='Grassmann'):
        """
        Parameters:
        - - - - -
        metric: string
            metric / distance to use
        """

        assert metric in ['Asimov', 'BinetCauchy', 'Chordal', 'FubiniStudy',
                          'Grassmann', 'Martin', 'Procrustes', 'Projection',
                          'Spectral']

        self.metric = metric
        self.distance_map = {'BinetCauchy': binetcauchy,
                             #'Asimov': asimov,
                             #'Chordal': chordal,
                             #'FubiniStudy': fubinistudy,
                             #'Grassmann': grassmann,
                             #'Martin': martin,
                             #'Spectral': spectral,
                             'Procrustes': procrustes,
                             'Projection': projection,
                             }

    def fit(self, X, Y):
        """
        Fit subspace distance between two equi-dimensional subspaces.

        Parameters:
        - - - - -
        X, Y: float, array
            two equi-dimensional subspaces
        """

        p = np.min([X.shape[1], Y.shape[1]])
        [q1, r1] = np.linalg.qr(X)
        [q2, r2] = np.linalg.qr(Y)

        S = q1.T.dot(q2)
        [u, s, v] = np.linalg.svd(S, full_matrices=False)

        theta = np.arccos(s)

        self.x_ = q1.dot(u)
        self.y_ = q2.dot(v)

        # print('Computing %s distance.' % (self.metric))
        self.distance_ = self.distance_map[self.metric](theta)