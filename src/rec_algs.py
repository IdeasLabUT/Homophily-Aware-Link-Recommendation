import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as pl
from pprint import pprint
import collections
import itertools
import random
import pickle
from networkx.algorithms.community.quality import modularity


def op_random_rec(g, calc_bool = True):

    if (calc_bool == True):

        edges_to_add = []

        nodes = g.nodes(data=True)
        edges = g.edges()
        nodes = {k: v for k, v in nodes}
        edges = tuple((k, v) for k, v in edges)

        i=0
        for n in list(nodes.keys()):
            if i % int(len(nodes) / 10) == 0:
                print(f'Progress: {i/len(nodes)}')
            i = i+1

            local_net = set(g.neighbors(n))
            local_net.add(n)

            targets = list(set(nodes.keys()) - local_net)

            same_alignment = True
            while same_alignment:
                t = random.choice(targets)
                if nodes[t]['cluster'] != nodes[n]['cluster']:
                    same_alignment = False


            edges_to_add.append((n, t))

        with open('rand_recs.pickle', 'wb') as fileref:
            pickle.dump(edges_to_add, fileref)

    else:
        with open('rand_recs.pickle', 'rb') as fileref:
            edges_to_add = pickle.load(fileref)

    return edges_to_add


def random_rec(g, calc_bool = True):

    if (calc_bool == True):

        edges_to_add = []

        nodes = g.nodes(data=True)
        edges = g.edges()
        nodes = {k: v for k, v in nodes}
        edges = tuple((k, v) for k, v in edges)

        i=0
        for n in list(nodes.keys()):
            if i % int(len(nodes) / 10) == 0:
                print(f'Progress: {i/len(nodes)}')
            i = i+1

            local_net = set(g.neighbors(n))
            local_net.add(n)

            targets = list(set(nodes.keys()) - local_net)

            t = random.choice(targets)

            edges_to_add.append((n, t))

        with open('rand_recs.pickle', 'wb') as fileref:
            pickle.dump(edges_to_add, fileref)

    else:
        with open('rand_recs.pickle', 'rb') as fileref:
            edges_to_add = pickle.load(fileref)

    return edges_to_add





def cn_rec(g, calc_bool, full_net = False):
    # iterate through all nodes
    # add edge to node with most common neighbors
    nodes = g.nodes(data=True)
    edges = g.edges()
    nodes = {k: v for k, v in nodes}
    edges = tuple((k, v) for k, v in edges)


    if (calc_bool == True):

        i = 0

        edges_to_add = []
        for n in list(nodes.keys()):
            if i % int(len(nodes) / 10) == 0:
                print(f'Progress: {i/len(nodes)}')
            i = i+1

            local_net = set(g.neighbors(n))
            local_net.add(n)

            max_neighbors = 0
            max_t = 'error'
            for t in list(set(nodes.keys()) - local_net):

                num_neighbors = len(list(nx.common_neighbors(g, n, t)))

                if num_neighbors > max_neighbors and (n, t) not in edges_to_add and (t, n) not in edges_to_add:
                    max_t = t
                    max_neighbors = num_neighbors

            if max_t == 'error':
                print(f'no suggestion available for node: {n}')
            else:
                edges_to_add.append((n, max_t))

        with open('cn_recs.pickle', 'wb') as fileref:
            pickle.dump(edges_to_add, fileref)

    else:
        if full_net == True:
            with open('full_net_pkl/cn_recs.pickle', 'rb') as fileref:
                edges_to_add = pickle.load(fileref)
        else:
            with open('cn_recs.pickle', 'rb') as fileref:
                edges_to_add = pickle.load(fileref)

    return edges_to_add


def cn_rec2(g, calc_bool, full_net = False):
    # iterate through all nodes
    # add edge to node with most common neighbors
    nodes = g.nodes(data=True)
    edges = g.edges()
    nodes = {k: v for k, v in nodes}
    edges = tuple((k, v) for k, v in edges)


    if (calc_bool == True):

        i = 0

        edges_to_add = []
        for n in list(nodes.keys()):
            if i % int(len(nodes) / 10) == 0:
                print(f'Progress: {i/len(nodes)}')
            i = i+1

            local_net = set(g.neighbors(n))
            local_net.add(n)

            tmp_g = nx.ego_graph(g, n, radius = 2)
            tmp_ego = set(tmp_g.nodes())

            max_neighbors = 0
            max_t = 'error'
            for t in list(tmp_ego - local_net):

                num_neighbors = len(list(nx.common_neighbors(g, n, t)))

                if num_neighbors > max_neighbors and (n, t) not in edges_to_add and (t, n) not in edges_to_add:
                    max_t = t
                    max_neighbors = num_neighbors

            if max_t == 'error':
                print(f'no suggestion available for node: {n}')
            else:
                edges_to_add.append((n, max_t))

        with open('cn_recs.pickle', 'wb') as fileref:
            pickle.dump(edges_to_add, fileref)

    else:
        if full_net == True:
            with open('full_net_pkl/cn_recs.pickle', 'rb') as fileref:
                edges_to_add = pickle.load(fileref)
        else:
            with open('cn_recs.pickle', 'rb') as fileref:
                edges_to_add = pickle.load(fileref)

    return edges_to_add


def op_cn_rec(g, calc_bool, full_net = False):
    # iterate through all nodes
    # add edge to node with most common neighbors
    nodes = g.nodes(data=True)
    edges = g.edges()
    nodes = {k: v for k, v in nodes}
    edges = tuple((k, v) for k, v in edges)


    if (calc_bool == True):

        i = 0

        edges_to_add = []
        for n in list(nodes.keys()):
            if i % int(len(nodes) / 10) == 0:
                print(f'Progress: {i/len(nodes)}')
            i = i+1

            local_net = set(g.neighbors(n))
            local_net.add(n)

            max_neighbors = 0
            max_t = 'error'
            for t in list(set(nodes.keys()) - local_net):

                num_neighbors = len(list(nx.common_neighbors(g, n, t)))

                if num_neighbors > max_neighbors and (n, t) not in edges_to_add and (t, n) not in edges_to_add and nodes[t]['cluster'] != nodes[n]['cluster']:
                    max_t = t
                    max_neighbors = num_neighbors

            if max_t == 'error':
                print(f'no suggestion available for node: {n}')
            else:
                edges_to_add.append((n, max_t))

        with open('op_cn_recs.pickle', 'wb') as fileref:
            pickle.dump(edges_to_add, fileref)

    else:
        if full_net == True:
            with open('full_net_pkl/op_cn_recs.pickle', 'rb') as fileref:
                edges_to_add = pickle.load(fileref)
        else:
            with open('op_cn_recs.pickle', 'rb') as fileref:
                edges_to_add = pickle.load(fileref)

    return edges_to_add

def op_cn_scores(g, calc_bool, full_net = False):
    # iterate through all nodes
    # add edge to node with most common neighbors
    nodes = g.nodes(data=True)
    edges = g.edges()
    nodes = {k: v for k, v in nodes}
    edges = tuple((k, v) for k, v in edges)


    if (calc_bool == True):

        i = 0

        scores = pd.DataFrame(columns = list(nodes.keys()), index = list(nodes.keys()))

        edges_to_add = []
        for n in list(nodes.keys()):
            if i % int(len(nodes) / 10) == 0:
                print(f'Progress: {i/len(nodes)}')
            i = i+1

            local_net = set(g.neighbors(n))
            local_net.add(n)

            max_neighbors = 0
            max_t = 'error'
            
            nearby = [ n for n, l in nx.single_source_shortest_path_length(g, n, cutoff=2).items() if l > 1 ]
            
            for t in nearby:

                num_neighbors = len(list(nx.common_neighbors(g, n, t)))
                
                if nodes[t]['cluster'] != nodes[n]['cluster']:
                    scores.loc[n, t] = num_neighbors
                    
                if num_neighbors > max_neighbors and (n, t) not in edges_to_add and (t, n) not in edges_to_add and nodes[t]['cluster'] != nodes[n]['cluster']:
                    max_t = t
                    max_neighbors = num_neighbors

            if max_t == 'error':
                print(f'no suggestion available for node: {n}')
            else:
                edges_to_add.append((n, max_t))

        with open('op_cn_recs.pickle', 'wb') as fileref:
            pickle.dump(edges_to_add, fileref)
        with open('op_cn_scores.pickle', 'wb') as fileref:
            pickle.dump(scores, fileref)

    else:
        if full_net == True:
            with open('full_net_pkl/op_cn_recs.pickle', 'rb') as fileref:
                edges_to_add = pickle.load(fileref)
        else:
            with open('op_cn_recs.pickle', 'rb') as fileref:
                edges_to_add = pickle.load(fileref)

    return scores