import utility
import random

import networkx as nx

from sklearn.cluster import KMeans
import numpy as np


################################################################################
#  select the training set based on nodes' degree sequence
################################################################################
def importance_base_sampling(population, weight_seq, size):
    # input:
    #     population - a list of target nodes id
    #     weight_seq - a list of weight of the node in pooulation
    #     size - the train set size
    # return:
    #     train_set, test_set - a list of node id - the sampling result of
    #                           train set, with remaining nodes as the test_set

    train_set = []
    new_population = population
    new_deg_seq    = weight_seq
    for i in range(size):
        # randomly pick a vertex based on degree
        data = random.choices(new_population, new_deg_seq, k=1)[0]
        train_set.append(data)

        # remove the selected vertex to avoid repetitive sampling
        index = new_population.index(data)
        new_population.pop(index)
        new_deg_seq.pop(index)

    test_set  = [i for i in population if not i in train_set]

    return train_set, test_set

################################################################################
#  randomly select the training set of specific size
################################################################################
def uniform_sampling(population, size):
    # input:
    #     population - a list of target nodes id
    #     size - the train set size
    # return:
    #     train_set, test_set - a list of node id - the sampling result of
    #                           train set, with remaining nodes as the test_set

    train_set = random.sample(population, k=size)
    test_set  = [i for i in population if not i in train_set]

    return train_set, test_set

################################################################################
#  returns an approximate minimum weighted vertex cover.
################################################################################
def weighted_vertex_cover(graph, weighted_vertex_cover=None):
    # input:
    #     graph - a networkx graph
    #     weighted_vertex_cover - a string of node attribute to be used as node weight
    #                             if None,  every node has weight 1.
    # return:
    #     cover - a set of nodes whose weight sum is no more than twice the
    #             weight sum of the minimum weight vertex cover.

    cover = nx.algorithms.approximation.min_weighted_vertex_cover(graph, weighted_vertex_cover)
    return cover

################################################################################
#  select the training set with each vertex being a k-centre
################################################################################
def k_center(graph, size, h):
    # input:
    #     graph - a networkx graph
    #     size - the size of the train set
    #     h - the effective cover of each vertex
    # return:
    #     train_set, test_set - a list of node id - the sampling result of
    #                           train set, with remaining nodes as the test_set

    train_set = []
    original_population = list(graph.nodes)
    current_graph = nx.Graph(graph)

    for i in range(size):
        population = list(current_graph.nodes)
        degree_dict = list(current_graph.degree)
        degree_list = [ele[1] for ele in degree_dict]

        # pick the next node based on degree
        data = random.choices(population, degree_list, k=1)[0]
        train_set.append(data)

        # remove the h-th neighbor of the selected vertex from the graph
        to_remove = utility.extract_k_neighbor(current_graph, data, h)
        current_graph.remove_nodes_from(to_remove)

    test_set  = [i for i in original_population if not i in train_set]
    return train_set, test_set

################################################################################
#  select the training set by covering strategy
################################################################################
def max_cover(graph, size, h):
    # input:
    #     graph - a networkx graph
    #     size - the size of the train set
    #     h - the effective cover of each vertex
    # return:
    #     train_set, test_set - a list of node id - the sampling result of
    #                           train set, with remaining nodes as the test_set

    # list to store train set
    train_set = []
    original_population = list(graph.nodes)
    current_graph = nx.Graph(graph)

    for i in range(size):
        population = list(current_graph.nodes)

        # pick a node with the highest degree
        degree_dict = list(current_graph.degree)
        degree_list = [ele[1] for ele in degree_dict]
        max_deg = max(degree_list)
        max_node_index = [i for i, e in enumerate(degree_list) if e == max_deg]
        chosen_node_index = random.choices(max_node_index, k=1)[0]

        # append the chosen node to train set
        data = population[chosen_node_index]
        train_set.append(data)

        # remove the chose node and its neighbor
        to_remove = utility.extract_k_neighbor(current_graph, data, h)
        current_graph.remove_nodes_from(to_remove)

    test_set  = [i for i in original_population if not i in train_set]
    return train_set, test_set

################################################################################
#  find the minimum number of centers needed
################################################################################
def set_cover_formulation(graph, h):
    # input:
    #     graph - a networkx graph
    #     h - the effective cover of each vertex
    # return:
    #     num_set - minimum number of centres need to fully cover the
    #               vertexes by greedy approach

    # create a k-hop neighbor set for each vertex
    cover_list = []

    node_set = set(graph.nodes)
    for v in node_set:
        h_neighbor_set = set(utility.extract_k_neighbor(graph, v, h))
        cover_list.append(h_neighbor_set)

    # apply the greedy approach
    uncovered_elem_num_list = [len(i) for i in cover_list]
    max_uncover_num = max(uncovered_elem_num_list)
    covered_element = set()
    num_set = 0

    while max_uncover_num > 0:
        num_set = num_set + 1
        max_uncover_set_index = uncovered_elem_num_list.index(max_uncover_num)
        max_uncover_set       = cover_list[max_uncover_set_index]
        covered_element = covered_element.union(max_uncover_set)

        # update cover number
        for i in range(len(cover_list)):
            cover_list[i] = cover_list[i].difference(covered_element)

        uncovered_elem_num_list = [len(i) for i in cover_list]
        max_uncover_num = max(uncovered_elem_num_list)

    return num_set


################################################################################
#  find the nodes id closest to the k centres by performing k-mean clustering
################################################################################
def centroid(dglgraph, node_list, radius, k):
    # input:
    #     dglgraph - a dgl graph object
    #     node_list - a list of node id
    #     radius - the radius of the neighbor
    #     k - the number of centres to be extracted
    # return:
    #     node_id_list - a list of k node id, which are closest to the
    #                    calculated k centres by performing k-mean clustering
    #                    on nodes feature vector

    # transform to to undirected networkx graph
    nx_graph = dglgraph.to_networkx().to_undirected()
    population = list(nx_graph.nodes)

    # a list to store the mean surounding feature information for each node
    mean_feat_list = []
    for v in node_list:
        neighbor_set = utility.extract_k_neighbor(nx_graph, v, radius)
        feature_set = dglgraph.ndata['feat'][list(neighbor_set)]

        # the mean feature vector
        mean_feat = feature_set.mean(0)
        mean_feat_list.append(mean_feat.numpy())

    kmeans = KMeans(n_clusters=k, random_state=0).fit(mean_feat_list)
    # k vectors of cluster centers.
    centroids = kmeans.cluster_centers_

    # a list to store the closest vertex to the calculated center
    node_id_list = []
    for center in centroids:
        # find the vertex closest to the centres
        closest_vector_index = closest_vector(center, mean_feat_list)
        node_id_list.append(node_list[closest_vector_index])

    return node_id_list

################################################################################
#  find the vector in a vector list which is closest to the target vector
################################################################################
def closest_vector(target_vector, vectors_list):
    # input:
    #     target_vector - the target vector
    #     vectors_list - a list of vectors
    # return:
    #     the index of vector in the vector_list closest to the target vector

    vectors_list = np.asarray(vectors_list)
    dist_2 = np.sum((vectors_list - target_vector)**2, axis=1)
    return np.argmin(dist_2)
