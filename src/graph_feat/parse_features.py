"""
Graph Features:

Centrality features
    - betweeness centrality
    - load centrality
    - closenes centrality
    - harmonic centrality

    - degree centrality
    - eigenvector centrality
    - pagerank centrality

Clique
    - Number of cliques
    - size (nodes in a clique)

Clustering Coefficient

Triangles

Square Clustering Coefficient

Average Neighbour Degree

Eccentricity
"""

import networkx as nx


def add_max_and_average(features_dict, feature_name, feature_values, mark_none=False):
    if mark_none:
        features_dict[f"max_{feature_name}"] = None
        features_dict[f"avg_{feature_name}"] = None
    else:
        if len(feature_values) == 0 or feature_values is None:
            feature_values = [0, ]
        features_dict[f"max_{feature_name}"] = max(feature_values)
        features_dict[f"avg_{feature_name}"] = 0

        if len(feature_values) > 0:
            features_dict[f"avg_{feature_name}"] = sum(feature_values) / len(feature_values)


def parse_path_length_centrality_feautres(graph: nx.graph):
    centrality_features = dict()
    try:
        betweeness_centrality_dict = nx.betweenness_centrality(graph)
        betweeness_centralities = [value for _, value in betweeness_centrality_dict.items()]
        add_max_and_average(centrality_features, "betweenness_centrality", betweeness_centralities)
    except:
        add_max_and_average(centrality_features, "betweenness_centrality", [], True)

    try:
        load_centrality_dict = nx.load_centrality(graph)
        load_centralities = [value for _, value in load_centrality_dict.items()]
        add_max_and_average(centrality_features, "load_centrality", load_centralities)
    except:
        add_max_and_average(centrality_features, "load_centrality", [], True)

    try:
        closeness_centrality_dict = nx.closeness_centrality(graph)
        closeness_centralities = [value for _, value in closeness_centrality_dict.items()]
        add_max_and_average(centrality_features, "closeness_centrality", closeness_centralities)
    except:
        add_max_and_average(centrality_features, "closeness_centrality", [], True)

    
    try:    
        harmonic_centrality_dict = nx.harmonic_centrality(graph)
        harmonic_centralities = [value for _, value in harmonic_centrality_dict.items()]
        add_max_and_average(centrality_features, "harmonic_centrality", harmonic_centralities)
    except:
        add_max_and_average(centrality_features, "harmonic_centrality", [], True)

    return centrality_features


def parse_neighbourliness_centrality_features(graph: nx.graph):
    centrality_features = dict()
    try:
        degree_centrality_dict = nx.degree_centrality(graph)
        degree_centralities = [value for _, value in degree_centrality_dict.items()]
        add_max_and_average(centrality_features, "degree_centrality", degree_centralities)
    except:
        add_max_and_average(centrality_features, "degree_centrality", [], True)

    try:
        eigenvector_centrality_dict = nx.eigenvector_centrality(graph)
        eigenvector_centralities = [value for _, value in eigenvector_centrality_dict.items()]
        add_max_and_average(centrality_features, "eigenvector_centrality", eigenvector_centralities)
    except:
        add_max_and_average(centrality_features, "eigenvector_centrality", [], True)


    return centrality_features


def parse_centrality_features(graph: nx.graph, parse_path_length_centralities: bool, parse_neighbourliness: bool):

    pl_centralities = parse_path_length_centrality_feautres(graph) if parse_path_length_centralities else dict()
    neighbourliness_centralities = parse_neighbourliness_centrality_features(graph) if parse_neighbourliness else dict()

    return pl_centralities | neighbourliness_centralities


def parse_clique_features(graph: nx.graph):
    clique_features = dict()
    try:
        number_of_cliques = [value for _, value in nx.number_of_cliques(graph).items()]
        add_max_and_average(clique_features, "number_of_cliques", number_of_cliques)
    except:
        add_max_and_average(clique_features, "number_of_cliques", [], True)

    try:
        size_of_cliques = [value for _, value in nx.node_clique_number(graph).items()]
        add_max_and_average(clique_features, "size_of_cliques", size_of_cliques)
    except:
        add_max_and_average(clique_features, "size_of_cliques", [], True)

    return clique_features


def parse_clustering_features(graph: nx.graph):
    clustering_features = dict()
    try:
        clustering_coefficient_values = [value for _, value in nx.clustering(graph).items()]
        add_max_and_average(clustering_features, "clustering_coefficent", clustering_coefficient_values)
    except:
        add_max_and_average(clustering_features, "clustering_coefficent", [], True)

    try:
        traingle_values = [value for _, value in nx.triangles(graph).items()]
        add_max_and_average(clustering_features, "traingle_coefficient", traingle_values)
    except:
        add_max_and_average(clustering_features, "traingle_coefficient", [], True)

    try:
        square_clustering_coefficient_values = [value for _, value in nx.square_clustering(graph).items()]
        add_max_and_average(clustering_features, "square_clustering_coefficient", square_clustering_coefficient_values)
    except:
        add_max_and_average(clustering_features, "square_clustering_coefficient", [], True)


    return clustering_features


def parse_misc_features(graph: nx.graph):
    misc_features = dict()
    try:
        avg_neighbour_degeees = [value for _, value in nx.average_neighbor_degree(graph).items()]
        add_max_and_average(misc_features, "avg_neighbour_degree", avg_neighbour_degeees)
    except:
        add_max_and_average(misc_features, "avg_neighbour_degree", [], True)

    try:
        eccentricities = [value for _, value in nx.eccentricity(graph).items()]
        add_max_and_average(misc_features, "eccentricity", eccentricities)
    except:
        add_max_and_average(misc_features, "eccentricity", [], True)


    return misc_features
    