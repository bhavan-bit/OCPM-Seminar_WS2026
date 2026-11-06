"""Patch for Synchronization Time (P5) in ocpa.

In Adams et al., Synchronization Time (P5) is defined as the elapsed time
between the first predecessor event finishing and the last predecessor event
finishing before a join node.

When an event has <= 1 predecessor (in-degree <= 1), P5 must strictly return 0.0,
rather than conflating with Waiting Time (P10) or returning a non-zero timestamp delta.
"""

def synchronization_time(node, graph, log):
    """Computes synchronization time with strict boundary condition enforcement.

    Args:
        node: Target event node in the execution graph.
        graph: NetworkX DiGraph representing process execution.
        log: Event log abstraction with get_value(node, attribute) access.

    Returns:
        float: Difference in seconds between latest and earliest predecessor completion,
               or 0.0 if in-degree <= 1.
    """
    in_edges = list(graph.in_edges(node))
    if len(in_edges) <= 1:
        # A single predecessor requires zero multi-object synchronization
        return 0.0
    pred_timestamps = [
        log.get_value(pred_node, "event_timestamp")
        for pred_node, _ in in_edges
    ]
    return (max(pred_timestamps) - min(pred_timestamps)).total_seconds()


def apply_p5_patch():
    """Applies the P5 synchronization patch to ocpa in runtime."""
    try:
        import ocpa.algo.predictive_monitoring.event_based_features.extraction_functions as ef
        ef.synchronization_time = synchronization_time
        print("[patch_ocpa_p5] Successfully patched ocpa synchronization_time (P5).")
    except ImportError:
        print("[patch_ocpa_p5] Warning: ocpa not found in current environment. Patch function ready for import.")


if __name__ == "__main__":
    apply_p5_patch()
