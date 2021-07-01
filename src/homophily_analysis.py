# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 04:00:51 2021

@author: CmdrRubz
"""
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import collections
import itertools
import random
import pickle
from networkx.algorithms.community.quality import modularity

# local imports
from score_functions import *
from rec_algs import *


def display_graph_info(g):

    nodes = g.nodes(data=True)
    edges = g.edges()
    info = nx.info(g)
    print(info)
    nodes = {k: v for k, v in nodes}
    edges = tuple((k, v) for k, v in edges)
    fname = 'cluster'
    fcon = 'right'
    flib = 'left'
    fcen = '-'
    con = [k for k, v in nodes.items() if v[fname] == fcon]
    lib = [k for k, v in nodes.items() if v[fname] == flib]
    cen = [k for k, v in nodes.items() if v[fname] == fcen]
    pprint(dict(con=len(con), lib=len(lib), cen=len(cen), total=(len(con) + len(lib) + len(cen))))


def calc_ei_homophily(g):

    nodes = g.nodes(data=True)
    edges = g.edges()
    nodes = {k: v for k, v in nodes}
    edges = tuple((k, v) for k, v in edges)

    scores = {}

    i=0
    for n in list(nodes.keys()):
        if i % int(len(nodes) / 10) == 0:
            print(f'Progress: {i/len(nodes)}')
        i=i+1


        local_net = set(g.neighbors(n))

        internal = 0
        external = 0
        for t in local_net:
            if nodes[n]['cluster'] == nodes[t]['cluster']:
                internal = internal + 1
            else:
                external = external + 1

        scores[n] = (external - internal) / (external + internal)

    with open('ei_scores.pickle', 'wb') as fileref:
        pickle.dump(scores, fileref)

    return scores

def read_polarization():
    full_pickle_file = 'full_net_pkl/ei_scores.pickle'
    with open(full_pickle_file, 'rb') as fileref:
        scores = pickle.load(fileref)

    return scores

def calc_polarization(g):
    nodes = g.nodes(data=True)
    edges = g.edges()
    nodes = {k: v for k, v in nodes}
    edges = tuple((k, v) for k, v in edges)

    scores = {}

    i=0
    for n in list(nodes.keys()):
        if i % int(len(nodes) / 10) == 0:
            print(f'Progress: {i/len(nodes)}')
        i=i+1


        local_net = set(g.neighbors(n))

        internal = 0
        external = 0
        neutral = 0

        if nodes[n]['cluster'] == '-':
            polarization = 0
        else:
            for t in local_net:
                if nodes[t] == '-':
                    neutral = neutral + 1
                if nodes[n]['cluster'] == nodes[t]['cluster']:
                    internal = internal + 1
                else:
                    external = external + 1

            polarization = (external - internal) / (external + internal)

        scores[n] = polarization

    with open('ei_scores.pickle', 'wb') as fileref:
        pickle.dump(scores, fileref)

    return scores

def dcg(result):
    dcg = []
    for idx, val in enumerate(result): 
        numerator = 2**val - 1
        # add 2 because python 0-index
        denominator =  np.log2(idx + 2) 
        score = numerator/denominator
        dcg.append(score)
    return sum(dcg)

def get_ndcg(g, scores):
    nodes = g.nodes(data=True)
    nodes = {k: v for k, v in nodes}
    fname = 'cluster'
    fcon = 'right'
    flib = 'left'
    fcen = '-'
    degrees = [(node,val) for (node, val) in g.degree()]
    
    not_con = [(k, v) for k, v in degrees if nodes[k][fname] != fcon]
    not_con = sorted(con, key = lambda x: x[1], reverse = True)
    not_lib = [(k, v) for k, v in degrees if nodes[k][fname] != flib]
    not_lib = sorted(lib, key = lambda x: x[1], reverse = True)
    not_cen = [(k, v) for k, v in degrees if nodes[k][fname] != fcen]
    not_cen = sorted(cen, key = lambda x: x[1], reverse = True)

    deg_sort = sorted(degrees, key=lambda x: x[1], reverse=True)
    
    node_ndcg = []

    
    for n in nodes.keys():
        n_scores = pd.to_numeric(scores[n])
        n_scores.sort_values(ascending=False)
        n_scores = n_scores[n_scores > 0]
        n_ranks = list(n_scores.index)
        
        res = []
        
        if np.sum(n_scores) == 0:
            # use degree rank instead
            # rank all nodes outside local net by degree
            local_net = set(g.neighbors(n))
            local_net.add(n)
            
            if nodes[node]['cluster'] == fcon:
                #notcon
                res = not_con
                found = False
                i = 0
                while not found:
                    if not_con[i][0] in local_net:
                        res.remove(not_con[i])
                    
                    if not_con[i][1] == 0:
                        found = True
            elif nodes[node]['cluster'] == flib:
                #notlib
                res = not_lib
                found = False
                i = 0
                while not found:
                    if not_lib[i][0] in local_net:
                        res.remove(not_con[i])
                    
                    if not_lib[i][1] == 0:
                        found = True
            else:
                #notcen
                res = not_cen
                found = False
                i = 0
                while not found:
                    if not_cen[i][0] in local_net:
                        res.remove(not_con[i])
                    
                    if not_lib[i][1] == 0:
                        found = True
            
            #deg_sort is ranking based on degree
            n_ranks = [elem[0] for elem in res]
            
        
        # relevance is num common neighbors
        rel_scores = []
        for node in n_ranks:
            rel_scores.append(len(list(nx.common_neighbors(g, n, node))))
        
        rel_scores = [ x / 1 for x in rel_scores]
        if(sum(rel_scores)) > 0:
            sort_rel_scores = list(np.sort(rel_scores))
            sort_rel_scores = sort_rel_scores[::-1]
            node_ndcg.append(dcg(rel_scores) / dcg(sort_rel_scores))
        else:
            node_ndcg.append(0)
        
    return np.average(node_ndcg)

def get_score_recs(g, rec_scores):

    scores = rec_scores.copy()

    rec_pairs = []

    nodes = list(g.nodes())
    t = ''
    i=0
    for n in nodes:
        if i % int(len(nodes) / 10) == 0:
            print(f'Progress: {i/len(nodes)}')
        i = i+1

        t='err'
        retry = True
        while retry:
            t = pd.to_numeric(scores[n]).idxmax()
            # Make sure there isn't an edge between n and t already (scores already shouldn't be generated though)
            if scores.loc[n, t] == 0:
                retry = False
            elif g.has_edge(n, t) or (t, n) in rec_pairs:
                scores.loc[n, t] = 0
            else:
                scores.loc[n, t] = 0
                scores.loc[t, n] = 0
                retry = False


        if t == 'err':
            print(f'unable to find paring for {n}. t = {t}.')

        if (t, n) in rec_pairs:
            print(f'{n} - {t} already in network...')

        rec_pairs.append((n, t))


    return rec_pairs

def display_modularity_change(g, scores, groups):
    g = g.copy()

    nodes = g.nodes(data=True)
    edges = g.edges()
    nodes = {k: v for k, v in nodes}
    edges = tuple((k, v) for k, v in edges)

    fcon = 'right'
    flib = 'left'
    fcen = '-'
    con = [k for k, v in nodes.items() if v['cluster'] == fcon]
    lib = [k for k, v in nodes.items() if v['cluster'] == flib]
    cen = [k for k, v in nodes.items() if v['cluster'] == fcen]



    # calc modularity of network
    q =  modularity(g, [con, lib, cen])
    pprint(dict(modularity=q))


    links_to_add = get_score_recs(g, scores)


    for e in links_to_add:
        if g.has_edge(e[0], e[1]):
            print(f"EDGE ALREADY IN NETWORK! {e[0]}, {e[1]}")
        g.add_edge(e[0], e[1])

    q_res = modularity(g, [con, lib, cen])
    pprint(dict(modularity=q_res))

    return links_to_add

# Threshold the network by degree and remove any nodes with no edges
def thresh_network(g, deg_thresh):
    nodes_to_remove = []
    for n in list(g.nodes()):
        if g.degree[n] < deg_thresh:
            nodes_to_remove.append(n)

    for n in nodes_to_remove:
        g.remove_node(n)

    for n in list(g.nodes()):
        if g.degree[n] == 0:
            g.remove_node(n)

    return g


def homophily_graph(scores, filename):
    data = list(scores.values())
    bins = np.linspace(-1, 1, 10)
    inds = np.digitize(data, bins)
    # bin_means = (np.histogram(data, bins, weights=data)[0])


    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(1, 1, figsize=(10,7))
    plt.tight_layout(pad = 3.2)
    plt.xlabel('Polarization', fontsize = 20)
    plt.ylabel('Num Nodes', fontsize = 20)
    # print(len(bins))
    # print(len(bin_means))
    plt.hist(data, bins='auto')

    plt.savefig(filename)



def get_extra_supp_rec(g, n):
    nodes = g.nodes(data=True)
    nodes = {k: v for k, v in nodes}

    n_nbrs = g.neighbors(n)
    n_sec_nbrs = set()
    for v in n_nbrs:
        n_sec_nbrs.update(g.neighbors(v))

    local_net = set(g.neighbors(n))
    local_net.add(n)

    max_cn = 0
    max_t = 'error'

    for t in list(set(nodes.keys()) - local_net):
        t_sec_nbrs = set()
        for u in g.neighbors(t):
            t_sec_nbrs.update(g.neighbors(u))


        num_cn = len(n_sec_nbrs.intersection(t_sec_nbrs))
        if num_cn > max_cn:     # Need to limit to not repeat???
            max_t = t
            max_cn = num_cn

    if max_t == 'error':
        print(f'WOW! No second supp rec available for {n}.')
    return (n, max_t)




def supp_recs(g, recs):
    full_recs = recs

    nodes = g.nodes(data=True)
    nodes = {k: v for k, v in nodes}

    # for each node in g
    for n in list(nodes.keys()):
        #   if g not in first elems of recs
        if not n in [p[0] for p in recs]:

            local_net = set(g.neighbors(n))
            local_net.add(n)

            max_neighbors = 0
            max_t = 'error'
            for t in list(set(nodes.keys()) - local_net):

                num_neighbors = len(list(nx.common_neighbors(g, n, t)))

                if num_neighbors > max_neighbors and (n, t) not in full_recs and (t, n) not in full_recs:
                    max_t = t
                    max_neighbors = num_neighbors

            if max_t == 'error':
                print(f'no supplementary suggestion available for node: {n} trying again')
                new_rec = get_extra_supp_rec(g, n)
                full_recs.append((n, new_rec))
            else:
                full_recs.append((n, max_t))


    return full_recs

def supp_cn_op(g, recs):
    full_recs = recs

    nodes = g.nodes(data=True)
    nodes = {k: v for k, v in nodes}

    # for each node in g
    for n in list(nodes.keys()):
        #   if g not in first elems of recs
        if not n in [p[0] for p in recs]:

            local_net = set(g.neighbors(n))
            local_net.add(n)

            max_deg = 0
            max_t = 'error'
            
            # t through every node other than local neighborhood
            for t in list(set(nodes.keys()) - local_net):

                # get degree of t
                t_deg = g.degree[t]

                # if highest degree so far and op aligned and not previously recommended
                if t_deg > max_deg and nodes[n]['cluster'] != nodes[t]['cluster'] and (n, t) not in full_recs and (t, n) not in full_recs:
                    max_t = t
                    max_deg = t_deg

            # add max degree oposite aligned node as rec for n (a node without a rec)
            if max_t == 'error':
                print(f'no supplementary suggestion available for node: {n} trying again')
                new_rec = get_extra_supp_rec(g, n)
                full_recs.append((n, new_rec))
            else:
                full_recs.append((n, max_t))


    return full_recs

# If no CNs, rank set to number of CN targets (max rank)
def get_cn_ranks(g, recs):

    nodes = g.nodes(data=True)
    edges = g.edges()
    nodes = {k: v for k, v in nodes}
    edges = tuple((k, v) for k, v in edges)

    rankings = []


    # for each rec
    for j, r in enumerate(recs):
        if j % 1000 == 0:
            print(f'Progress: {j/len(recs)}')

        n = r[0]

        local_net = set(g.neighbors(n))
        local_net.add(n)

        max_neighbors = 0
        max_t = 'error'

        cn_targets = []
        targets = []
        
        
        nearby = [ n for n, l in nx.single_source_shortest_path_length(g, n, cutoff=2).items() if l > 1 ]

        # Get num common neighbors with every node outside of local network
        for t in nearby:

            num_neighbors = len(list(nx.common_neighbors(g, n, t)))

            # save each num neighbors
            cn_targets.append((t, num_neighbors))
            targets.append(t)

        # sort to get rankings
        cn_targets.sort(reverse = True, key = lambda x: x[1])


        if r[1] in targets and cn_targets[0][1] > 0:

            for i, tar in enumerate(cn_targets):
                if r[1] == tar[0]:
                    rankings.append((i, tar[1] / cn_targets[0][1]))

        else:
            # set rank to max if no candidate (no common neighbors)
            rankings.append((len(cn_targets), 0))

    return rankings




if __name__ == "__main__":

    thresh_net = False
    re_calc = False


    g = nx.read_graphml('all.graphml')
    # convert from digraph to undirected
    if g.is_directed():
        g = g.to_undirected()
    # convert from multigraph to standard graph
    g = nx.Graph(g)

    print(f'num nodes b4: {len(g.nodes())}')

    # original network: 22405 nodes
    # 10 gives 2351
    if thresh_net:
        g = thresh_network(g, 10)

    print(f'num nodes after: {len(g.nodes())}')

    nodes = g.nodes(data=True)
    edges = g.edges()
    nodes = {k: v for k, v in nodes}
    edges = tuple((k, v) for k, v in edges)

    fcon = 'right'
    flib = 'left'
    fcen = '-'
    con = [k for k, v in nodes.items() if v['cluster'] == fcon]
    lib = [k for k, v in nodes.items() if v['cluster'] == flib]
    cen = [k for k, v in nodes.items() if v['cluster'] == fcen]

    groups = [con, lib, cen]

    if thresh_net:
        polar_scores = calc_polarization(g)
    else:
        polar_scores = read_polarization() # On full network
        
    # TODO: graph of polarizations

    homophily_graph(polar_scores, 'polarization_graph.pdf')

    # print('getting cn and cn_op recs...')
    # # cn_recs = cn_rec(g, re_calc, full_net = not thresh_net)
    # cn_op_recs = op_cn_rec(g, re_calc, full_net = not thresh_net)
    # print('done')
    
    # # TODO: cn_op_supplemented
    # full_cn_op = supp_cn_op(g, cn_op_recs)
    

    # # print(f'num_edges: {len(g.edges())}')
    # # s_edges = len(g.edges())


    # # print(len(cn_op_recs))
    # # new_cn_op_recs = supp_recs(g, cn_op_recs)
    # # print(len(new_cn_op_recs))

    # # get supp recs for op_cn_recs


    # # print('apply recommendations...')
    # # adj_recs = display_modularity_change(g, scores, groups)

    # # print(f'num_edges: {len(g.edges())}')
    # # e_edges = len(g.edges())

    # # added_edges = e_edges - s_edges
    # # print(f'edges added: {added_edges}')


    # # print('getting scores...')
    # # pickle_file = 'full_net_pkl/adj_h_score.pickle'
    # # with open(pickle_file, 'rb') as fileref:
    # #     scores = pickle.load(fileref)

    # scores = op_cn_scores(g, True, False)
    
    # print("getting ndcg...")
    # print("ndcg: ", get_ndcg(g, scores))

    # # print(f'num_edges: {len(g.edges())}')
    # # print('ego recs mod change...')
    # # ego_recs = display_modularity_change(g, scores, groups)
    # # print(f'num_edges: {len(g.edges())}')

    # # # get CN rankings on recs
    # # print('num recs: ', len(ego_recs))
    # rankings = get_cn_ranks(g, full_cn_op)

    # print('Mean rank: ', np.mean([rec_comp[0] for rec_comp in rankings]))
    # print('Median rank: ', np.median([rec_comp[0] for rec_comp in rankings]))
    # print('Mean prop of max: ', np.mean([rec_comp[1] for rec_comp in rankings]))

    # # calc modularity of network
    # q =  modularity(g, [con, lib, cen])
    # pprint(dict(modularity=q))


    # # links_to_add = get_score_recs(g, scores)
    # links_to_add = full_cn_op

    # print(len(links_to_add))

    # edges_before=len(g.edges())

    # for e in links_to_add:
    #     if g.has_edge(e[0], e[1]):
    #         print(f"EDGE ALREADY IN NETWORK! {e[0]}, {e[1]}")
    #     g.add_edge(e[0], e[1])

    # q_res = modularity(g, [con, lib, cen])
    # pprint(dict(modularity=q_res))

    # print('Edges added: ', len(g.edges()) - edges_before)

    # # new_polar_scores = calc_polarization(g)

    # # homophily_graph(new_polar_scores, 'after_ego_homophily_graph.png')


    # # same_rec = 0
    # # for r in ego_recs:
    # #     if r in cn_recs:
    # #         same_rec = same_rec + 1

    # # print('Num recs: ',len(ego_recs))
    # # print('Num orecs: ',len(cn_recs))
    # # print(f'Num same: {same_rec}')
    # # print(f'proportion: {same_rec / len(ego_recs)}')

    # # print('adj recs...')

    # # adj_recs = display_modularity_change(g, scores_adj, groups)


    # # same_rec = 0
    # # for r in adj_recs:
    # #     if r in cn_recs:
    # #         same_rec = same_rec + 1

    # # print('Num recs: ',len(adj_recs))
    # # print('Num orecs: ',len(cn_recs))
    # # print(f'Num same: {same_rec}')
    # # print(f'proportion: {same_rec / len(adj_recs)}')




    # # links_to_add = op_random_rec(g, True)

    # # q = modularity(g, [con, lib, cen])
    # # pprint(dict(modularity=q))

    # # s_edges = len(g.edges())

    # # for e in links_to_add:
    # #     g.add_edge(e[0], e[1])

    # # e_edges = len(g.edges())

    # # print(f'edges added: {e_edges - s_edges}')

    # # # calc modularity of network
    # # q_res =  modularity(g, [con, lib, cen])
    # # pprint(dict(modularity=q_res))

    # # print(f'change in modularity: {q_res - q}')
