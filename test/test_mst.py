import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    # assert that the MST consists of exactly V-1 edges
    num_edges = np.sum(mst > 0) // 2 
    assert num_edges == adj_mat.shape[0] - 1, f'Proposed MST has incorrect number of edges (has {num_edges}, but should have {adj_mat.shape[0] - 1} edges)'

    # assert symmetry of the MST across the diagonal
    assert np.all(mst - mst.T < allowed_error)

    # build the MST as a Graph object and assert that the MST is connected (use built in connected
    # method added to the Graph object)
    g_mst = Graph(mst)
    assert g_mst.connected() == True, f'Proposed MST is not connected'


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_type_student():
    """
    TODO: Write at least one unit test for MST construction.
    
    Unit test for the attempted construction of a graph with incorrect data type
    """
    
    # assert that a TypeError is thrown when attempting to build a graph object with the
    # wrong data type as input
    with pytest.raises(TypeError, match='Input must be a valid path or an adjacency matrix'):
        _ = Graph(123)


def test_mst_empty_student():
    """
    TODO: Write at least one unit test for MST construction.
    
    Unit test for the attempted construction of an MST on an empty adjacency matrix
    """
    
    # assert that a ValueError is thrown when one attempts to construct an MST for an empty graph
    empty_mat = np.array([])
    g_empty = Graph(empty_mat)
    with pytest.raises(ValueError, match="This graph is empty and does not contain any nodes."):
        g_empty.construct_mst()


def test_mst_disconnected_student():
    """
    TODO: Write at least one unit test for MST construction.

    Unit test for the attempted construction of an MST on a disconnected graph
    """
    
    # assert that a ValueError is thrown when one attempts to construct an MST for a graph where one node has no edges
    file_path = './data/small_disconnected.csv'
    g = Graph(file_path)
    with pytest.raises(ValueError, match="This graph is disconnected. No minimum spanning tree exists."):
        g.construct_mst()
