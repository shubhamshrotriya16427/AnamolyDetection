import json
import networkx as nx


def parse_as_path_strings(update_data, get_messages=False):
    msgs_with_as_path = []
    as_paths = []

    for entry in update_data:
        bgp_message = entry.get("bgp_message", {})
        if "type" in bgp_message:
            if bgp_message["type"].get("2") == "UPDATE":  

                if "path_attributes" in bgp_message and bgp_message["path_attributes"]:
                    for attribute in bgp_message["path_attributes"]:
                        if attribute["type"].get("2") == "AS_PATH":
                            for as_path in attribute["value"]:
                                if as_path["type"].get("2") == "AS_SEQUENCE":
                                    as_paths.append(as_path["value"])
                                    if get_messages:
                                        msgs_with_as_path.append(bgp_message)

    return as_paths, msgs_with_as_path


def generate_graph(json_data):
    as_paths, _ = parse_as_path_strings(json_data)

    graph = nx.Graph()
    for as_path in as_paths:
        graph.add_nodes_from(as_path)
        for idx in range(len(as_path)-1):
            graph.add_edge(as_path[idx], as_path[idx+1])

    return graph
    

