# Python program for Dijkstra's single
# source shortest path algorithm. The program is
# for adjacency matrix representation of the graph

# Library for INT_MAX
from random import randint
import matplotlib.pyplot as plt
import sys


class Graph():

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[-1 for column in range(vertices)]
                      for row in range(vertices)]

    def printSolution(self, dist):
        print("Vertex tDistance from Source")
        for node in range(self.V):
            print(node, "t", dist[node])

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):

        # Initilaize minimum distance for next node
        min = sys.maxsize

        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v

        return min_index

    # Funtion that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, src):

        dist = [sys.maxsize] * self.V
        dist[src] = 0
        sptSet = [False] * self.V

        for cout in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            try:
                u = self.minDistance(dist, sptSet)
            except Exception as e:
                u = 0

            # Put the minimum distance vertex in the
            # shotest path tree
            sptSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shotest path tree
            for v in range(self.V):
                if self.graph[u][v] > -1 and \
                        sptSet[v] == False and \
                        dist[v] > dist[u] + self.graph[u][v]:
                    dist[v] = dist[u] + self.graph[u][v]

        self.printSolution(dist)
        return dist

    # Driver program

    def extract_edges(self):
        """
        this method returns a set of all of the edges. List of threesomes
        :return:
        """
        edges = set()
        for i in range(0, len(self.graph)):
            for j in range(i, len(self.graph[i])):
                if self.graph[i][j] > -1:
                    edges.add((i, j, self.graph[i][j]))
        return edges


def generate_random_graph(number_of_nodes, weight=1):
    """
    This function generates random graph from number of nodes.
    :param number_of_nodes:
    :return:
    """
    graph_to_return = Graph(number_of_nodes)
    graph_to_return.graph = [[weight if randint(0, 1) == 0 else -1 for column in range(number_of_nodes)]
                             for row in range(number_of_nodes)]

    return graph_to_return


def spanners_algorithm(graph_object, r):
    edges = graph_object.extract_edges()
    yael = sorted(edges, key=lambda x: x[2])
    g_tag = Graph(graph_object.V)
    for u, v, w in yael:
        shortest_distance_uv = g_tag.dijkstra(u)[v]
        if w > -1 and r * w < shortest_distance_uv:
            g_tag.graph[u][v] = w
            g_tag.graph[v][u] = w
    return g_tag


g = Graph(9)
g.graph = [[-1, 4, -1, -1, -1, -1, -1, 8, -1],
           [4, -1, 8, -1, -1, -1, -1, 11, -1],
           [-1, 8, -1, 7, -1, 4, -1, -1, 2],
           [-1, -1, 7, -1, 9, 14, -1, -1, -1],
           [-1, -1, -1, 9, -1, 1 - 1, -1, -1, -1],
           [-1, -1, 4, 14, 1 - 1, -1, 2, -1, -1],
           [-1, -1, -1, -1, -1, 2, -1, 1, 6],
           [8, 11, -1, -1, -1, -1, 1, -1, 7],
           [-1, -1, 2, -1, -1, -1, 6, 7, -1]
           ]


# def basic_random_graphs(ziv_):


def create_spanners_alg(graph_object, spanners_values_array):
    """
    This function returns spanner graphs for a specific graph, getting as input all of the r values in array.
    :param graph_object: Graph Object.
    :param spanners_values_array: array of int
    :return: array of graphs.
    """
    return [spanners_algorithm(graph_object, i) for i in spanners_values_array]


def graph_list_appender(list_of_random_graphs, list_of_graph_lists):
    list_dimension = len(list_of_graph_lists[0])  # if we have n spanners then for each graph then n
    r_spanners_array = [[] for i in range(0, list_dimension)]

    for graph_list in list_of_graph_lists:
        for j in range(0, len(graph_list)):
            r_spanners_array[j].append(len(graph_list[j].extract_edges()))

    return r_spanners_array
def experiment_1(number_of_vertices):
    random_graphs = [generate_random_graph(number_of_vertices) for i in range(0, 10)]
    random_graphs_array = [create_spanners_alg(generate_random_graph(5), [2, 4, 7]) for random_graphs in
                           range(0, 10)]  # List<List<Graph>>
    spanners_array = graph_list_appender(random_graphs_array)
    plt.plot(spanners_array[0], 'r')
    # plt.plot(spanners_array[1],'b')
    plt.show()

experiment_1()
