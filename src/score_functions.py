# -*- coding: utf-8 -*-
"""
Egonet and Adjacent scores to calculate a polarization based link prediction
scores for the whole network.

@author: CmdrRubz
"""


import pandas as pd
import networkx as nx
import numpy as np
import pickle



# Calculate scores for whole network. Egonet weights common neighbors by the
# polarization of the common neighbors
def egonet_homophily_scores(g, calc_bool = True):
    # pickle_file = 'eg_score_small.pickle'
    pickle_file = 'ego_h_score.pickle'
    full_pickle_file = 'full_net_pkl/ego_h_score.pickle'
    if calc_bool == False:
        pickle_file = full_pickle_file

    if (calc_bool == True):

        with open('ei_scores.pickle', 'rb') as fileref:
            ei_scores = pickle.load(fileref)


        nodes = g.nodes(data=True)
        edges = g.edges()
        nodes = {k: v for k, v in nodes}
        edges = tuple((k, v) for k, v in edges)

        scores = pd.DataFrame(columns = list(nodes.keys()), index = list(nodes.keys()))

        i=0
        for n in list(nodes.keys()):
            if i % int(len(nodes) / 10) == 0:
                print(f'Progress: {i/len(nodes)}')
            i=i+1

            local_net = set(g.neighbors(n))
            local_net.add(n)

            targets = list(set(nodes.keys()) - local_net)

            for t in targets:
                common_neighbors = list(nx.common_neighbors(g, n, t))

                score_sum = 0
                for u in common_neighbors:
                    score_sum = score_sum + (1 - np.abs(ei_scores[u]))

                scores.loc[n, t] = score_sum
                if g.has_edge(n,t):
                    print("DONT NEED TO CHECK THESE")


        with open(pickle_file, 'wb') as fileref:
            pickle.dump(scores, fileref)



    else:
        with open(pickle_file, 'rb') as fileref:
            scores = pickle.load(fileref)

    return scores


# Calculate scores for whole network. Adj score weighs the final common neighbors
# sum by the polarization of the two nodes which share those CNs.
def adj_homophily_scores(g, calc_bool = True):
    pickle_file = 'adj_h_score.pickle'
    full_pickle_file = 'full_net_pkl/adj_h_score.pickle'

    if (calc_bool == True):

        with open('ei_scores.pickle', 'rb') as fileref:
            ei_scores = pickle.load(fileref)

        f_out = open('prog.out', 'w')

        nodes = g.nodes(data=True)
        edges = g.edges()
        nodes = {k: v for k, v in nodes}
        edges = tuple((k, v) for k, v in edges)

        scores = pd.DataFrame(columns = list(nodes.keys()), index = list(nodes.keys()))

        i=0
        for n in list(nodes.keys()):
            if i % int(len(nodes) / 10) == 0:
                f_out.write(f'Progress: {i/len(nodes)}\n')
                f_out.flush()
                print(f'Progress: {i/len(nodes)}')
            i=i+1

            local_net = set(g.neighbors(n))
            local_net.add(n)

            targets = list(set(nodes.keys()) - local_net)

            for t in targets:
                num_common_neighbors = len(list(nx.common_neighbors(g, n, t)))

                abs_ei_n = np.abs(ei_scores[n])
                abs_ei_t = np.abs(ei_scores[t])


                score = num_common_neighbors * (np.abs(abs_ei_n - abs_ei_t))


                scores.loc[n, t] = score


        with open('adj_h_score.pickle', 'wb') as fileref:
            pickle.dump(scores, fileref)



    else:
        with open(pickle_file, 'rb') as fileref:
            scores = pickle.load(fileref)

    return scores
