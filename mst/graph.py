import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """

        # check if the tree is fully connected; if not, throw an error
        # this will happen if any node has no connections row/col sums to 0
        if np.any(self.adj_mat.sum(axis=0) == 0):
            raise ValueError("This graph is not fully connected. No minimum spanning tree exists.")

        # init arbitrary starting node s, a set of explored nodes, and the MST
        s = 0                                       # start exploring from node 0
        explored = {}                               # set of visited explored nodes (just s for now)
        mst = np.zeros_like(self.adj_mat)           # the current mst

        # init list min_cost to track the cheapest known edge between node v and explored nodes.
        # (e.g. min_cost[2] is the cheapest known edge from node 2 to any of the explored nodes)
        min_cost = [float('inf')] * self.adj_mat.shape[0]
        min_cost[s] = 0

        # init list pred, which keeps track of each node to the current MST
        # (e.g. pred[2] is the predecessor of node 2 in the MST)
        pred = [None] * self.adj_mat.shape[0]

        # init an empty min priority queue pq and push all values into pq ordered by their
        # cheapest known edge to the explored nodes
        pq = [(min_cost[v], v) for v in range(self.adj_mat.shape[0])]
        heapq.heapify(pq)

        # while the priority queue is not empty, continue to grow the tree
        while len(pq) > 0:
            
            # pop the cheapest node
            cost, u = heapq.heappop(pq)     

            # if node u has already been explored, continue (no need to even consider)
            if u in explored:
                continue

            # add node u to the explored list
            explored.append(u)

            # as long as this is not the first iteration, update the edges in the mst
            if pred != None:
                mst[u, pred[u]] = cost          # connect u to its predecessor already in the mst
                mst[pred[u], u] = cost          # also update symmetric edge in adj matrix

            # look at each node connected to u
            u_neighbors = np.nonzero(self.adj_mat[u, :])[0]
            for v in u_neighbors:

                # if node v has already been explored, continue (no need to even consider)
                if v in explored:
                    continue
                
                # get the edge cost from u to v
                edge_cost = self.adj_mat[u, v]

                # if the cost from u to v is less than the current cost to add v to the tree,
                # update min_cost and pred accordingly
                if edge_cost < min_cost[v]:
                    min_cost[v] = edge_cost     # updates 
                    pred[v] = u
                    heapq.heappush(pq, (edge_cost, v))

        self.mst = mst