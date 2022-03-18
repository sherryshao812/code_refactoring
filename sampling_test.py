import networkx as nx
import dgl
import unittest
from sampling_refactor import *

nx_G1 = nx.Graph()
nx_G1.add_nodes_from([0,1,2,3])
nx_G1.add_edges_from([(0,1), (0,2), (0,3), (1,2)])

nx_G2 = nx.Graph()
nx_G2.add_nodes_from([0,1])
nx_G2.add_edge(0,1)

nx_G3 = nx

cora = dgl.data.CoraGraphDataset()
cora_graph = cora[0]
nx_cora_graph = dgl.to_networkx(cora_graph)

class Test(unittest.TestCase):

    # test importance_base_sampling(population, weight_seq, size)
    def test_importance_base_sampling_1(self):
        for i in range(5):
            train, test = importance_base_sampling([0,1,2,3], [1000,1,1,1], 3)
            self.assertEqual(train[0], 0)

    def test_importance_base_sampling_2(self):
        train, test = importance_base_sampling([0,1,2,3], [1,1,1,1], 0)
        self.assertEqual(train, [])
        self.assertEqual(set(test), {0,1,2,3})

    def test_importance_base_sampling_3(self):
        train, test = importance_base_sampling([], [], 0)
        self.assertEqual(train, [])
        self.assertEqual(test, [])

    # test uniform_sampling(population, size)
    def test_uniform_sampling_1(self):
        train, test = uniform_sampling([], 0)
        self.assertEqual(train, [])
        self.assertEqual(test, [])

    def test_uniform_sampling_2(self):
        for i in range(10):
            train, test = uniform_sampling([0,1,2], 1)
            self.assertTrue(train in [[0],[1],[2]])

    def test_uniform_sampling_3(self):
        zeros = 0
        for i in range(1000):
            train, test = uniform_sampling([0,1],1)
            if train == [0]:
                zeros += 1
        self.assertTrue(zeros/1000 > 0.4 and zeros/1000 < 0.6)

    # test weighted_vertex_cover(graph, weighted_vertex_cover)
    def test_weighted_vertex_cover_1(self):
        result = weighted_vertex_cover(nx_G1)
        self.assertEqual(result, {0,1})

    def test_weighted_vertex_cover_2(self):
        empty_graph = nx.Graph()
        result = weighted_vertex_cover(empty_graph)
        self.assertEqual(result, set())

    def test_weighted_vertex_cover_3(self):
        g = nx.Graph()
        g.add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8)])
        result = weighted_vertex_cover(g)
        self.assertTrue(len(result) <= 2)

    # test k_center(graph, size, h):
    def test_k_center_1(self):
        train, test = k_center(nx_cora_graph, 500, 1)
        self.assertEqual(len(train), 500)
        for n in train:
            for m in train:
                self.assertTrue(n not in list(nx_cora_graph[m]))

    # test max_cover(graph, size, h)
    def test_max_cover_1(self):
        train, test = max_cover(nx_G1, 1, 1)
        self.assertEqual(train, [0])
        self.assertEqual(test, [1,2,3])

    def test_max_cover_2(self):
        train, test = max_cover(nx_G1, 0, 1)
        self.assertEqual(train, [])
        self.assertEqual(test, [0,1,2,3])

    def test_max_cover_3(self):
        train, test = max_cover(nx_cora_graph, 500, 1)
        self.assertEqual(len(train), 500)
        for n in train:
            for m in train:
                self.assertTrue(n not in list(nx_cora_graph[m]))
        node_degree = [(n, len(nx_cora_graph[n])) for n in nx_cora_graph.nodes]
        maxDegree = max([d for (n,d) in node_degree])
        maxDegree_node = [n for (n,d) in node_degree if d == maxDegree]
        self.assertTrue(train[0] in maxDegree_node)

    # test set_cover_formulation(graph, h)
    def test_set_cover_formulation_1(self):
        result = set_cover_formulation(nx_G1, 1)
        self.assertEqual(result, 2)

    def test_set_cover_formulation_2(self):
        result = set_cover_formulation(nx_G2, 1)
        self.assertEqual(2, 2)

    # test centroid(dglgraph, node_list, r, k)
    def test_centroid_1(self):
        result = centroid(cora_graph, list(range(100)), 1, 10)
        self.assertEqual(len(result), 10)

    # test closest_vector(target_vector, vectors_list):
    def test_closest_vector_1(self):
        result = closest_vector((0,0), [(1,1),(1,3),(0,1),(-1,-1)])
        self.assertEqual(result, 2)

    def test_closest_vector_2(self):
        result = closest_vector((0,0,0,0,0), [(0,1,0,1,0), (10,0,0,0,0),(0,-1,0,0,0)])
        self.assertEqual(result, 2)









if __name__ == '__main__':
    unittest.main()
