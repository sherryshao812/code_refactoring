
import networkx as nx

################################################################################
#  create binary labels for a sequence of nodes id
################################################################################
def create_binary_label(node_id_list, population_size):
    # input:
    #     node_id_list - the target node id list
    #     population_size - the length of the node id sequence
    # return:
    #     binary_label_list - a list of binary labels of length population_size,
    #                         with 1 saying the node id is in the node_id_list

    binary_label_list = []
    for id in range(population_size):
        if id in node_id_list:
            binary_label_list.append(1)
        else:
            binary_label_list.append(0)
    return binary_label_list

################################################################################
#  given a node label list, return a list of node id of a specific lable
################################################################################
def extract_node_id_for_label(node_label_list, target_label):
    # input:
    #     node_label_list - the list of node label, ordered by node id
    #     target_label - the target label
    # return:
    #     node_id_list - the list of node id whose label is the target_label

    # list to store the node id
    node_id_list = []

    # iterate through the node population and check for matched label
    for id in range(len(node_label_list)):
        if node_label_list[id] == target_label:
            #stored node id of the matched node
            node_id_list.append(id)
    return node_id_list

################################################################################
#  convert a boolean mask to a list of node id being masked
################################################################################
def extract_node_id_from_mask(mask):
    # input:
    #     mask - a boolean list, ordered by node id
    # return:
    #     node_id_list - a list of node id whose value is True in the mask

    node_id_list = []
    for id in range(len(mask)):
        if mask[id] == True:
            node_id_list.append(id)

    return node_id_list

################################################################################
#  extract a node id list from a boolean mask
################################################################################
def extract_index_from_mask(mask):
    # input:
    #     mask - a boolean list, ordered by node id
    # return:
    #     a list of node id whose value is True in the mask

    return [idx for idx, v in enumerate(mask) if v]

################################################################################
#  convert a node id list to a boolean mask
################################################################################
def node_id_list_to_mask(node_id_list, num_node):
    # input:
    #     node_id_list - the node id list
    #     num_node - total number of nodes
    # return:
    #     bool_list - a boolean list of length num_node, with each value
    #                 indicating whether each node id is in the node_id_list

    # construct a all-False boolean list of length num_node
    bool_list = num_node * [False]
    # change value to True if the corresponding node id is in the node_id_list
    for i in range(num_node):
        if i in node_id_list:
            bool_list[i] = True

    return bool_list

################################################################################
#  convert a node id list to a boolean mask
################################################################################
def id_list_to_mask(node_id_list, graph_size):
    # input:
    #     node_id_list - the node id list
    #     graph_size - total number of nodes in the graph
    # return:
    #     mask - a boolean list, indicating whether each node id is in the node_id_list

    mask = []
    for i in range(graph_size):
        if i in node_id_list:
            mask.append(True)
        else:
            mask.append(False)

    return mask

################################################################################
#  return the node id list sorted by nodes' degree
################################################################################
def node_id_in_degree_sorted(node_id_list, graph):
    # input:
    #     node_id_list - the node id list to be sorted
    #     graph - a networkx graph
    # return:
    #     node_list_sort - the sorted node id list sorting by nodes' degree

    degree_list  = list(graph.degree(node_id_list))
    degree_list_sort = sorted(degree_list,reverse = True, key=lambda element: element[1])
    node_list_sort = [ele[0] for ele in degree_list_sort]

    return node_list_sort

################################################################################
#  given a graph and node_id_list, return a dictionary
#  ['degree'] = {list of vertex in node_id_list with this degree}
################################################################################
def degree_distribution(graph, node_id_list):
    # input:
    #     graph - a networkx graph
    #     node_id_list - a list of node id being considerated
    # return:
    #     degree_vertex_dict - a dictionary with degrees as keys and lists of
    #                          node id in node_id_list with this degree as values

    degree_vertex_dict = {}
    vertex_degree_dict = dict(graph.degree(node_id_list))
    for k, v in vertex_degree_dict.items():
        degree_vertex_dict[v] = degree_vertex_dict.get(v, []) + [k]

    return degree_vertex_dict

################################################################################
#  given a networkx graph, return the dictionary ['node_id'] = degree of this node
################################################################################
def degree_dict(graph):
    # input:
    #     graph - a networkx graph
    # return:
    #     degree_dict - a dictionary storing the degree of each node in the graph

    vertex_list = list(graph.nodes)
    degree_dict = dict(graph.degree(vertex_list))
    return degree_dict

################################################################################
#  extract the node id from the k-th neighbor of a given vertex in graph G
################################################################################
def extract_k_neighbor(G, target_vertex, k):
    # input:
    #     G - a networkx graph
    #     target_vertex - the target vertex whose neighbor to be extracted
    #     k - the radius of the neighbor to extract
    # return:
    #     a list of node id from the vertexes of the corresponding neighbor

    # set to store node id
    neighbor_set = set([target_vertex])

    # iterative extract node id from the k-hop neighbor
    for l in range(k):
        neighbor_set = set((node_id for vertex in neighbor_set for node_id in G[vertex]))

    return list(neighbor_set)

################################################################################
#  extract the node id for (0~k)-th neighbors of a given vertex in graph G
################################################################################
def extract_k_hop_neighbor(G, vertex, k):
    # input:
    #     G - a networkx graph
    #     vertex - the target vertex whose neighbor to be extracted
    #     k - the radius of the neighbors to be extracted
    # return:
    #     neighbor_list - a list containing k+1 sets of the vertexes of the
    #                     corresponding 0~k -th neighbor, no repeating

    # list to store the neighbors
    neighbor_list = []
    # set to store the visited vertexes
    visited       = set()

    neighbor_list.append({vertex})
    visited.add(vertex)

    for i in range(k):
        # set to store the i-th neighbor, possibly repeatted
        neighbor_candidate = set()
        for v in neighbor_list[i]:
            to_vist = set([n for n in G.neighbors(v)])
            neighbor_candidate.update(to_vist)

        # set to store the i-th neighbor, with no repeat
        next_neighbor = set()
        for u in neighbor_candidate:
            if u not in visited:
                next_neighbor.add(u)
                visited.add(u)

        neighbor_list.append(next_neighbor)

    return neighbor_list

################################################################################
#  extract the node label for the (0~k)-th neighbor of a given vertex in graph G
################################################################################
def extract_k_hop_neighbor_label(G, vertex, k, node_label_list):
    # input:
    #     G - a networkx graph
    #     vertex - the target vertex whose neighbor to be extracted
    #     k - the radius of the neighbors to be extracted
    #     node_label_list -
    # return:
    #     neighbor_list_label - a list containing (k+1) sub-lists of the lables
    #                           of 0~k -th neighbors

    # list to store the neighbors
    neighbor_list = []
    # list to store the neighbors' label
    neighbor_list_label = []
    # list to store the visited vertexes
    visited       = set()

    neighbor_list.append({vertex})
    neighbor_list_label.append([node_label_list[vertex]])
    visited.add(vertex)

    for i in range(k):
        # set to store the i-th neighbor, possibly repeatted
        neighbor_candidate = set()
        for v in neighbor_list[i]:
            to_vist = set([n for n in G.neighbors(v)])
            neighbor_candidate.update(to_vist)

        # set to store the i-th neighbor, with no repeat
        next_neighbor       = set()
        # list to store the i-th neighbors' label
        next_neighbor_label = []
        for u in neighbor_candidate:
            if u not in visited:
                next_neighbor.add(u)
                next_neighbor_label.append(node_label_list[u])
                visited.add(u)

        neighbor_list.append(next_neighbor)
        neighbor_list_label.append(next_neighbor_label)

    return neighbor_list_label
