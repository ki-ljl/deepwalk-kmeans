# -*- coding: utf-8 -*-
"""
@Time ： 2021/12/18 20:34
@Author ：KI 
@File ：deepwalk.py
@Motto：Hungry And Humble

"""
import networkx as nx
from gensim.models import Word2Vec
from numpy import random
import numpy as np
import matplotlib.pyplot as plt


class deep_walk:
    def __init__(self, G, d, r, l, k):
        self.G = G
        self.d = d  # dimension
        self.r = r  # walks per node
        self.l = l  # walk length
        self.k = k  # window size

    def random_walk(self, u):
        g = self.G
        walk = [u]
        while len(walk) < self.l:
            curr = walk[-1]
            v_curr = list(g.neighbors(curr))
            if len(v_curr) > 0:
                walk.append(random.choice(v_curr))
            else:
                break

        return walk

    def learning_features(self):
        g = self.G
        walks = []
        nodes = list(g.nodes())
        for t in range(self.r):
            random.shuffle(nodes)
            for node in nodes:
                walk = self.random_walk(node)
                walks.append(walk)
        # embedding
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(sentences=walks, vector_size=self.d, window=self.k, min_count=0, sg=1, workers=3)
        f = model.wv
        print(f['MmeBurgon'])
        return f

    def plot(self, m, K):
        """
        :param m: node embedding
        :param K: number of clusters
        :return: none
        """
        g = self.G
        pos = nx.spring_layout(G)

        color_map = []
        ns = list(G.nodes.data())
        nodes = list(g.nodes)

        res = self.k_means(m, K, 50)

        colors = ['#DCBB8A', '#98BBEF', 'navy', 'indigo', 'orange', 'blue']
        color_map.clear()
        for node in nodes:
            for i in range(len(res)):
                if node in res[i]:
                    color_map.append(colors[i])
                    break
        # draw
        # plt.subplot(2, 1, 2)
        nx.draw(G, node_color=color_map, pos=pos, with_labels=False, node_size=2000)
        plt.show()

        res = self.k_means(m, K, 100)

        colors = ['#DCBB8A', '#98BBEF', 'navy', 'indigo', 'orange', 'blue']
        color_map.clear()
        for node in nodes:
            for i in range(len(res)):
                if node in res[i]:
                    color_map.append(colors[i])
                    break
        # draw
        # plt.subplot(2, 1, 2)
        nx.draw(G, node_color=color_map, pos=pos, with_labels=False, node_size=2000)

        plt.show()

    def get_dis(self, x, y):
        s = 0
        for i in range(len(x)):
            s += (x[i] - y[i]) ** 2

        return np.sqrt(s)

    def k_means(self, m, K, t):
        """
        :param m: node embedding
        :param K: number of clusters
        :return: result
        """
        d = self.d
        nodes = list(G.nodes)
        centers = []
        temp = []
        for i in range(K):
            t = np.random.randint(0, len(nodes) - 1)
            if nodes[t] not in temp:
                temp.append(nodes[t])
                centers.append(m[nodes[t]])  #

        #
        res = {}
        for i in range(K):
            res[i] = []

        for time in range(t):
            # clear
            for i in range(K):
                res[i].clear()
            # Calculate the distance from the vector of each point to the cluster center
            nodes_distance = {}
            for node in nodes:
                # The distance from the node to the central node
                node_distance = []
                for center in centers:
                    node_distance.append(self.get_dis(m[node], center))
                nodes_distance[node] = node_distance  #
            # Reclassify each node, select a nearest node for classification, the class is 0-5
            for node in nodes:
                temp = nodes_distance[node]  #
                cls = temp.index(min(temp))
                res[cls].append(node)

            # Update cluster centers
            centers.clear()
            for i in range(K):
                center = []
                for j in range(d):
                    t = [m[node][j] for node in res[i]]  #
                    center.append(np.mean(t))
                centers.append(center)

        return res


if __name__ == '__main__':
    d, r, l, k = 128, 10, 80, 10
    G = nx.les_miserables_graph()
    deep_walk = deep_walk(G, d, r, l, k)
    model = deep_walk.learning_features()
    deep_walk.plot(model, 6)
