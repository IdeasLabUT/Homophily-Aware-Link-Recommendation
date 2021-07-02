# Homophily-Aware-Link-Recommendation

Data can be obtained from: https://zenodo.org/record/4589065#.YN51ZOhKguU

all.graphml is read using NetworkX's read_graphml() function.


The majority of functionality pertaining to analysis, network functions, and metrics is contained in homophily_analysis.py. The various link recommendation algorithms are in rec_algs.py, as well as a function to get scores for the opposite random algorithm.

The two score functions we define based on polarization, adjacent and egonet polarization scores, are in score_functions.py.

In homophily_analysis.py, __main__ gives an example of how these functions can be used for analysis.
