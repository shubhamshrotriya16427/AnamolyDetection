from graph_feat.parse_features import *
from graph_feat.graph_generation import generate_graph

def parse_graph_features(json):
    graph = generate_graph(json)

    parse_pl_centralities = True
    parse_neighbourliness_centralities = True
    
    return \
          parse_centrality_features(graph, parse_pl_centralities, parse_neighbourliness_centralities) \
            | parse_clique_features(graph) | parse_clustering_features(graph) | parse_misc_features(graph)
