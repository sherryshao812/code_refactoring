import networkx as nx
import dgl
import unittest
from utility_refactor import *

nx_G1 = nx.Graph()
nx_G1.add_nodes_from([0,1,2,3])
nx_G1.add_edges_from([(0,1), (0,2), (0,3), (1,2)])

nx_G2 = nx.Graph()
nx_G2.add_nodes_from([0,1])
nx_G2.add_edge(0,1)

cora = dgl.data.CoraGraphDataset()
cora_graph = cora[0]
nx_cora_graph = dgl.to_networkx(cora_graph)


class Test(unittest.TestCase):

    # test create_binary_label(node_id_list, population_size):
    def test_create_binary_label_1(self):
        result = create_binary_label([0,2], 4)
        self.assertEqual(result, [1,0,1,0])

    def test_create_binary_label_2(self):
        result = create_binary_label([], 5)
        self.assertEqual(result, [0,0,0,0,0])

    def test_create_binary_label_3(self):
        result = create_binary_label([], nx_cora_graph.number_of_nodes())
        self.assertEqual(len(result), 2708)

    # test extract_node_id_for_label(node_label_list, target_label):
    def test_extract_node_id_for_label_1(self):
        result = extract_node_id_for_label([0,1,2,3,0,1,2,3], 1)
        self.assertEqual(result, [1,5])

    def test_extract_node_id_for_label_2(self):
        result = extract_node_id_for_label([], 0)
        self.assertEqual(result, [])

    def test_extract_node_id_for_label_3(self):
        result = extract_node_id_for_label(cora_graph.ndata['label'], 0)
        result_2 = [i for i in range(len(cora_graph.ndata['label'])) if cora_graph.ndata['label'][i] == 0]
        self.assertEqual(result, result_2)

    # test extract_node_id_from_mask(mask):
    def test_extract_node_id_from_mask_1(self):
        result = extract_node_id_from_mask([1,0,0,1,0,0,1])
        self.assertEqual(result, [0,3,6])

    def test_extract_node_id_from_mask_2(self):
        result = extract_node_id_from_mask([])
        self.assertEqual(result, [])

    # test extract_index_from_mask(mask):
    def test_extract_index_from_mask_1(self):
        result = extract_index_from_mask([0,1,0,1,0,1])
        self.assertEqual(result, [1,3,5])

    def test_extract_index_from_mask_2(self):
        result = extract_index_from_mask([])
        self.assertEqual(result, [])

    # test node_id_list_to_mask(node_id_list, num_node):
    def test_node_id_list_to_mask_1(self):
        result = node_id_list_to_mask([1,3,5], 7)
        self.assertEqual(result, [0,1,0,1,0,1,0])

    def test_node_id_list_to_mask_2(self):
        result = node_id_list_to_mask([], 5)
        self.assertEqual(result, [0,0,0,0,0])

    def test_node_id_list_to_mask_3(self):
        result = node_id_list_to_mask([1,3,5], 0)
        self.assertEqual(result, [])

    # test id_list_to_mask(node_id_list, graph_size):
    def test_id_list_to_mask_1(self):
        result = id_list_to_mask([1,3,5], 7)
        self.assertEqual(result, [0,1,0,1,0,1,0])

    def test_id_list_to_mask_2(self):
        result = id_list_to_mask([], 5)
        self.assertEqual(result, [0,0,0,0,0])

    def test_id_list_to_mask_3(self):
        result = id_list_to_mask([1,3,5], 0)
        self.assertEqual(result, [])

    # test node_id_in_degree_sorted(node_id_list, graph)
    def test_node_id_in_degree_sorted_1(self):
        result = node_id_in_degree_sorted([0,1,2,3], nx_G1)
        self.assertEqual(result, [0,1,2,3])

    def test_node_id_in_degree_sorted_2(self):
        result = node_id_in_degree_sorted([], nx_G1)
        self.assertEqual(result, [])

    def test_node_id_in_degree_sorted_3(self):
        result = node_id_in_degree_sorted(list(nx_cora_graph.nodes), nx_cora_graph)
        #self.assertEqual(cora_graph.ndata['label'][result[0]], 6)
        self.assertEqual(nx_cora_graph.degree(result[0]), max([nx_cora_graph.degree(n) for n in nx_cora_graph.nodes]))
        self.assertEqual(nx_cora_graph.degree(result[-1]), min([nx_cora_graph.degree(n) for n in nx_cora_graph.nodes]))

    # test degree_distribution(graph, node_id_list)
    def test_degree_distribution_1(self):
        result = degree_distribution(nx_G1, list(nx_G1.nodes))
        self.assertEqual(result[1], [3])
        self.assertEqual(result[2], [1,2])
        self.assertEqual(result[3], [0])

    def test_degree_distribution_2(self):
        result = degree_distribution(nx_G1, [])
        self.assertEqual(result, {})

    # test degree_dict(graph)
    def test_degree_dict_1(self):
        result = degree_dict(nx_G1)
        self.assertEqual(result[0], 3)
        self.assertEqual(result[1], 2)
        self.assertEqual(result[2], 2)
        self.assertEqual(result[3], 1)

    def test_degree_dict_2(self):
        result = degree_dict(nx_cora_graph)
        self.assertEqual(result[0], nx_cora_graph.degree(0))
        self.assertEqual(result[100], nx_cora_graph.degree(100))
        self.assertEqual(result[1000], nx_cora_graph.degree(1000))

    # test extract_k_neighbor(G, target_vertex, k)
    def test_extract_k_neighbor_1(self):
        result = [extract_k_neighbor(nx_G2, 0, k) for k in range(5)]
        result_2 = [extract_k_neighbor(nx_G2, 1, j) for j in range(5)]

        self.assertEqual(result, [[0],[1],[0],[1],[0]])
        self.assertEqual(result_2, [[1],[0],[1],[0],[1]])

    def test_extract_k_neighbor_2(self):
        result = extract_k_neighbor(nx_cora_graph, 0, 1)
        self.assertEqual(result, list(nx_cora_graph[0]))

    # test extract_k_hop_neighbor(G, vertex, k)
    def test_extract_k_hop_neighbor_1(self):
        result = extract_k_hop_neighbor(nx_G1, 0, 2)
        result_2 = extract_k_hop_neighbor(nx_G1, 1, 2)
        result_3 = extract_k_hop_neighbor(nx_G1, 1, 0)
        self.assertEqual(result, [{0}, {1,2,3}, set()])
        self.assertEqual(result_2, [{1}, {0,2}, {3}])
        self.assertEqual(result_3, [{1}])

    def test_extract_k_hop_neighbor_2(self):
        result = extract_k_hop_neighbor(nx_cora_graph, 1, 5)
        self.assertEqual(result[1], set(nx_cora_graph[1]))

    # test extract_k_hop_neighbor_label(G, vertex, k, node_label_list)
    def test_extract_k_hop_neighbor_label_1(self):
        result = extract_k_hop_neighbor_label(nx_cora_graph, 0, 3, list(cora_graph.ndata['label']))
        self.assertEqual(result[0], [cora_graph.ndata['label'][0]])

        neighbor_1 = [int(n) for n in nx_cora_graph[0] if int(n)!= 0]
        neighbor_1_label = [cora_graph.ndata['label'][n] for n in neighbor_1]
        self.assertEqual(result[1], neighbor_1_label)

        neighbor_2 = [int(n2) for n1 in neighbor_1 for n2 in nx_cora_graph[n1] if int(n2) not in neighbor_1 and int(n2) != 0]
        neighbor_2_removeDuplicate = []
        for n in neighbor_2:
            if n not in neighbor_2_removeDuplicate:
                neighbor_2_removeDuplicate.append(n)
        neighbor_2_label = [cora_graph.ndata['label'][n] for n in neighbor_2_removeDuplicate]
        self.assertEqual(result[2], neighbor_2_label)






if __name__ == '__main__':
    unittest.main()
