import numpy as np
import multinetx as mx
import networkx as nx
from myStruct import Stack, Queue
import time


def generate_erdos(n, p=0.2):
    g = nx.fast_gnp_random_graph(n, p)
    name = 'graph_' + str(n) + '.graphml'
    return name, g

def read_graph(graph):
    if type(graph) == list:
        return [nx.read_graphml(i[0]) for i in graph]
    else:
        return nx.read_graphml(graph)

#   BETWEENNESS CENTRALITY FOR SIMPLE GRAPH
def btc_s(g):
    Sh = [0 for i in g.nodes()]
    for s in g.nodes():
        S = Stack()
        P = [[] for i in g.nodes()]
        delta = [0 for i in g.nodes()]
        d = [-1 for i in g.nodes()]
        delta[s] = 1
        d[s] = 0
        Q = Queue()
        Q.enqueue(s)
        while not Q.isEmpty():
            v = Q.dequeue()
            S.push(v)
            for w in nx.all_neighbors(g, v):
                if d[w] < 0:
                    Q.enqueue(w)
                    d[w] = d[v] + 1
                if d[w] == (d[v] + 1):
                    delta[w] += delta[v]
                    P[w].append(v)
        dv = [0 for i in g.nodes()]
        while not S.isEmpty():
            w = S.pop()
            for v in P[w]:
                dv[v] += (delta[v]/delta[w]) * (1 + dv[w])
            if w != s:
                Sh[w] += dv[w]
    res = dict()
    for i in range(0,len(Sh)):
        res[i] = Sh[i]
    return res

#   BETWEENNESS CENTRALITY FOR MULTIPLEX GRAPH
def btc_m(g1, N, rang):
    adj_block = mx.lil_matrix(np.zeros((N * 3, N * 3)))
    adj_block[0:  N, 2 * N:3 * N] = np.identity(N)
    adj_block[N:2 * N, 2 * N:3 * N] = np.identity(N)
    adj_block += adj_block.T
    arr = [g1 for i in range(0, rang)]
    mg = mx.MultilayerGraph(list_of_layers=arr, inter_adjacency_matrix=adj_block)
    return btc_s(mg)

if __name__ == '__main__':
    graph_lst = []
    TT = [10, 100, 1000]
    for i in TT:
        graph_lst.append(
            generate_erdos(i)
        )
    g1 = graph_lst[1][1]
    N = TT[1]
    rang = 50
    print(nx.info(g1))
    t0 = time.clock()
    btc_s(g1)
    print("time for simple graph %f"%(time.clock()-t0))
    t0 = time.clock()
    btc_m(g1, N, rang)
    print("time for multiplex %f"%(time.clock() - t0))
    # print(nx.betweenness_centrality(g1))
    # print(mx.betweenness_centrality(mg))