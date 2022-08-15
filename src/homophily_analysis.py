# -*- coding: utf-8 -*-
"""
Base code for homophily analysis on a politically polarized network.

@author: CmdrRubz
"""

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import pickle
from networkx.algorithms.community.quality import modularity

# local imports
from score_functions import *
from rec_algs import *

# Print basic graph info including political alignment cluster totals
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

# get pickled polarization scores
def read_polarization():
    full_pickle_file = 'full_net_pkl/ei_scores.pickle'
    with open(full_pickle_file, 'rb') as fileref:
        scores = pickle.load(fileref)

    return scores

# return polarization scores for whole network and pickle the results
# polarization is ei-homophily where neutrals are neither internal nor external
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


# Get recommendations for each node from a set of scores.
# Returns recommendations as node pairs
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
            # print(f'{n} - {t} already in network...')
            pass

        rec_pairs.append((n, t))


    return rec_pairs

# Print the modularity change from a set of scores on the network.
# Uses get_score_recs(). Returns added links.
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

# Save a histogram of the polarization scores on the network
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





# Supplement a set of recommendations with simple Common Neighbors recommendations
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


# Further supplement recommendations with path length 2 neighbors.
# Unused for analysis.
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
        if num_cn > max_cn:
            max_t = t
            max_cn = num_cn

    if max_t == 'error':
        print(f'WOW! No second supp rec available for {n}.')
    return (n, max_t)


# Supplement the opposite common neighbors recommendations with an opposite
# aligned high degree node when no recommendation exists.
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


# Get the rankings of recommendations based on the common neighbors recs.
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


    # original network: 22405 nodes
    # 10 gives 2351
    if thresh_net:
        g = thresh_network(g, 10)

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

    # polar_scores = calc_polarization(g)
    if thresh_net:
        polar_scores = calc_polarization(g)
    else:
        polar_scores = read_polarization() # On full network
        

    # homophily_graph(polar_scores, 'polarization_graph.pdf')

    print('getting cn and cn_op recs...')
    # cn_recs = cn_rec(g, re_calc, full_net = not thresh_net)
    cn_op_recs = op_cn_rec(g, re_calc, full_net = not thresh_net)
    print('done')
    
    print('getting scores...')
    # calc scores
    # egonet_homophily_scores(g)
    
    #read scores from pickle
    pickle_file = 'full_net_pkl/ego_h_score.pickle'
    with open(pickle_file, 'rb') as fileref:
        scores = pickle.load(fileref)
        
    recs = get_score_recs(g, scores)
    
    node_limit = [r[0] for r in cn_op_recs]
    limited_recs = [r for r in recs if r[0] in node_limit]
    
    # use limited recommendations for metrics
    recs = limited_recs
    
    # full_cn_op = supp_cn_op(g, cn_op_recs)
    

    # scores = op_cn_scores(g, True, False)
    
    # print("getting ndcg...")
    # print("ndcg: ", get_ndcg(g, scores))

    

    rankings = get_cn_ranks(g, recs)

    print('Mean rank: ', np.mean([rec_comp[0] for rec_comp in rankings]))
    print('Median rank: ', np.median([rec_comp[0] for rec_comp in rankings]))
    print('Mean prop of max: ', np.mean([rec_comp[1] for rec_comp in rankings]))

    # calc modularity of network
    q =  modularity(g, [con, lib, cen])
    pprint(dict(modularity=q))


    # links_to_add = get_score_recs(g, scores)
    links_to_add = recs

    print(len(links_to_add))

    edges_before=len(g.edges())

    for e in links_to_add:
        if g.has_edge(e[0], e[1]):
            print(f"EDGE ALREADY IN NETWORK! {e[0]}, {e[1]}")
        g.add_edge(e[0], e[1])

    q_res = modularity(g, [con, lib, cen])
    pprint(dict(modularity=q_res))

    print('Edges added: ', len(g.edges()) - edges_before)
