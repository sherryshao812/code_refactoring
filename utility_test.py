import networkx as nx
import dgl
import unittest
from utility_refactor import *

nx_G1 = nx.Graph()
nx_G1.add_nodes_from([0,1,2,3])
nx_G1.add_edges_from([(0,1), (0,2), (0,3), (1,2)])

cora = dgl.data.CoraGraphDataset()
cora_graph = cora[0]
nx_cora_graph = dgl.to_networkx(cora_graph)


class Test(unittest.TestCase):

    # test the create_binary_label function
    def test_create_binary_label_1(self):
        result = create_binary_label([0,2], 4)
        self.assertEqual(result, [1,0,1,0])

    def test_create_binary_label_2(self):
        result = create_binary_label([], 5)
        self.assertEqual(result, [0,0,0,0,0])

    def test_create_binary_label_3(self):
        result = create_binary_label([], nx_cora_graph.number_of_nodes())
        self.assertEqual(len(result), 2708)

    # test the extract_node_id_for_label function
    def test_extract_node_id_for_label_1(self):
        result = extract_node_id_for_label([0,1,2,3,0,1,2,3], 1)
        self.assertEqual(result, [1,5])

    def test_extract_node_id_for_label_2(self):
        result = 



if __name__ == '__main__':
    unittest.main()
