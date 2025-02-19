def parse_stat_feat(json):
    # Initialize counters
    updates = 0
    a_updates = 0
    w_updates = 0
    a_prefix = 0
    w_prefix = 0
    a_dup = 0
    w_dup = 0
    aw_mix = 0
    as_path_lengths = []
    packet_sizes = []

    announced_prefixes = set()
    withdrawn_prefixes = set()
    prefix_states = {}

    # Function to convert prefix representation to a hashable type
    def prefix_to_str(prefix):
        return f"{prefix['prefix']}/{prefix['length']}"

    for entry in json:
        bgp_message = entry.get("bgp_message", {})
        if "type" in bgp_message:
            if bgp_message["type"].get("2") == "UPDATE":  # Identifying an UPDATE message
                # Counting the update message
                updates += 1
                packet_sizes.append(bgp_message.get("length", 0))  # Collect packet sizes
                # Checking for announced prefixes
                if "nlri" in bgp_message and bgp_message["nlri"]:
                    if "withdrawn_routes" not in bgp_message or (bgp_message.get("withdrawn_routes") == []):
                        a_updates += 1
                    for prefix in bgp_message["nlri"]:
                        prefix_str = prefix_to_str(prefix)
                        a_prefix += 1
                        # Check for duplicate announcements
                        if prefix_str in announced_prefixes:
                            a_dup += 1
                        else:
                            announced_prefixes.add(prefix_str)
                        # Check if previously withdrawn, now announced (mix)
                        if prefix_states.get(prefix_str) == "withdrawn":
                            aw_mix += 1
                        prefix_states[prefix_str] = "announced"
                # Checking for withdrawn prefixes
                if "withdrawn_routes" in bgp_message and bgp_message["withdrawn_routes"]:
                    if(len(bgp_message["withdrawn_routes"]) > 0):
                        w_updates += 1
                    for prefix in bgp_message["withdrawn_routes"]:
                        prefix_str = prefix_to_str(prefix)
                        w_prefix += 1
                        # Check for duplicate withdrawals
                        if prefix_str in withdrawn_prefixes:
                            w_dup += 1
                        else:
                            withdrawn_prefixes.add(prefix_str)
                        # Check if previously announced, now withdrawn (mix)
                        if prefix_states.get(prefix_str) == "announced":
                            aw_mix += 1
                        prefix_states[prefix_str] = "withdrawn"
                # Processing AS_PATH attributes
                if "path_attributes" in bgp_message and bgp_message["path_attributes"]:
                    for attribute in bgp_message["path_attributes"]:
                        if attribute["type"].get("2") == "AS_PATH":
                            for as_path in attribute["value"]:
                                if as_path["type"].get("2") == "AS_SEQUENCE":
                                    as_path_lengths.append(as_path["length"])

    # Calculate additional metrics
    average_as_path_length = sum(as_path_lengths) / len(as_path_lengths) if as_path_lengths else 0
    max_as_path_length = max(as_path_lengths) if as_path_lengths else 0
    average_packet_size = sum(packet_sizes) / len(packet_sizes) if packet_sizes else 0

    # Print the extracted features
    # print(f"Updates: {updates}")
    # print(f"A-Updates: {a_updates}")
    # print(f"W-Updates: {w_updates}")
    # print(f"A-Prefix: {a_prefix}")
    # print(f"W-Prefix: {w_prefix}")
    # print(f"A-Dup: {a_dup}")
    # print(f"W-Dup: {w_dup}")
    # print(f"AW-Mix: {aw_mix}")
    # print(f"Average AS-path length: {average_as_path_length}")
    # print(f"Maximum AS-path length: {max_as_path_length}")
    # print(f"Average packet size: {average_packet_size}")

    # Assign the extracted features to a dictionary
    stats = {
        "Updates": updates,
        "A-Updates": a_updates,
        "W-Updates": w_updates,
        "A-Prefix": a_prefix,
        "W-Prefix": w_prefix,
        "A-Dup": a_dup,
        "W-Dup": w_dup,
        "AW-Mix": aw_mix,
        "Average AS-path length": average_as_path_length,
        "Maximum AS-path length": max_as_path_length,
        "Average packet size": average_packet_size,
    }

    return stats