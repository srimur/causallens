"""
causallens.graph.visualize
===========================
Render a CausalGraph as a visual DAG.

Two backends:
1. **pyvis** (preferred): Interactive HTML with physics simulation
2. **matplotlib** + networkx: Static PNG/SVG fallback

Both consume the shared CausalGraph type — zero coupling to
discovery or inference components.

Dependencies
------------
- ``causallens.types`` : CausalGraph, EdgeType
- networkx (graph layout)
- pyvis (interactive HTML) — optional
- matplotlib (static images) — optional
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from causallens.types import CausalGraph, EdgeType


# Color scheme
COLORS = {
    "node": "#4A90D9",
    "node_root": "#2ECC71",
    "node_leaf": "#E67E22",
    "edge_directed": "#333333",
    "edge_undirected": "#999999",
    "edge_bidirected": "#E74C3C",
    "background": "#FFFFFF",
}


def to_networkx(graph: CausalGraph) -> nx.DiGraph:
    """
    Convert a CausalGraph to a networkx DiGraph.

    Undirected edges become two directed edges.
    Bidirected edges become two directed edges with a 'bidirected' attribute.
    """
    G = nx.DiGraph()
    G.add_nodes_from(graph.variables)

    for e in graph.edges:
        attrs = {
            "edge_type": e.edge_type.value,
            "weight": e.weight,
            "p_value": e.p_value,
        }
        if e.edge_type == EdgeType.DIRECTED:
            G.add_edge(e.source, e.target, **attrs)
        elif e.edge_type == EdgeType.UNDIRECTED:
            G.add_edge(e.source, e.target, **attrs)
            G.add_edge(e.target, e.source, **attrs)
        elif e.edge_type == EdgeType.BIDIRECTED:
            G.add_edge(e.source, e.target, **attrs)
            G.add_edge(e.target, e.source, **attrs)

    return G


def render_html(
    graph: CausalGraph,
    output_path: str = "causal_graph.html",
    title: str = "CausalLens — Causal Graph",
    height: str = "600px",
    width: str = "100%",
    physics: bool = True,
    highlight_edges: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    Render an interactive HTML visualization using pyvis.

    Parameters
    ----------
    graph : CausalGraph
        The causal graph to render.
    output_path : str
        Where to save the HTML file.
    title : str
        Title shown in the visualization.
    height, width : str
        Dimensions of the visualization.
    physics : bool
        Enable physics simulation for layout.
    highlight_edges : list of (source, target) tuples
        Edges to highlight in red (e.g., the causal path).

    Returns
    -------
    str : Path to the generated HTML file.
    """
    from pyvis.network import Network

    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=False,
        bgcolor=COLORS["background"],
        font_color="#333333",
    )
    net.heading = title

    if physics:
        net.toggle_physics(True)
        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -100,
                    "centralGravity": 0.01,
                    "springLength": 200,
                    "springConstant": 0.08
                },
                "solver": "forceAtlas2Based",
                "stabilization": {"iterations": 150}
            }
        }
        """)
    else:
        net.toggle_physics(False)

    highlight_set = set()
    if highlight_edges:
        highlight_set = {(s, t) for s, t in highlight_edges}

    # Classify nodes
    G = to_networkx(graph)
    roots = {v for v in graph.variables if G.in_degree(v) == 0}
    leaves = {v for v in graph.variables if G.out_degree(v) == 0}

    # Add nodes
    for var in graph.variables:
        if var in roots:
            color = COLORS["node_root"]
            shape = "dot"
        elif var in leaves:
            color = COLORS["node_leaf"]
            shape = "dot"
        else:
            color = COLORS["node"]
            shape = "dot"

        label = var.replace("_", "\n")
        net.add_node(
            var,
            label=label,
            color=color,
            shape=shape,
            size=30,
            font={"size": 14, "face": "Arial"},
        )

    # Add edges
    for e in graph.edges:
        is_highlight = (e.source, e.target) in highlight_set

        if e.edge_type == EdgeType.DIRECTED:
            color = "#E74C3C" if is_highlight else COLORS["edge_directed"]
            width = 3 if is_highlight else 2
            net.add_edge(
                e.source, e.target,
                color=color,
                width=width,
                arrows="to",
                title=_edge_tooltip(e),
            )
        elif e.edge_type == EdgeType.UNDIRECTED:
            color = COLORS["edge_undirected"]
            net.add_edge(
                e.source, e.target,
                color=color,
                width=1.5,
                arrows="",
                dashes=True,
                title=_edge_tooltip(e),
            )
        elif e.edge_type == EdgeType.BIDIRECTED:
            color = COLORS["edge_bidirected"]
            net.add_edge(
                e.source, e.target,
                color=color,
                width=2,
                arrows="to;from",
                title=_edge_tooltip(e),
            )

    net.save_graph(output_path)
    return os.path.abspath(output_path)


def render_static(
    graph: CausalGraph,
    output_path: str = "causal_graph.png",
    figsize: Tuple[int, int] = (10, 8),
    title: str = "CausalLens — Causal Graph",
    highlight_edges: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    Render a static image using matplotlib + networkx.

    Parameters
    ----------
    graph : CausalGraph
        The causal graph to render.
    output_path : str
        Where to save the image (.png or .svg).
    figsize : tuple
        Figure size in inches.
    title : str
        Plot title.
    highlight_edges : list
        Edges to highlight.

    Returns
    -------
    str : Path to the generated image file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G = to_networkx(graph)

    # Layout
    try:
        pos = nx.planar_layout(G)
    except nx.NetworkXException:
        pos = nx.spring_layout(G, seed=42, k=2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    highlight_set = set()
    if highlight_edges:
        highlight_set = {(s, t) for s, t in highlight_edges}

    # Classify nodes
    roots = {v for v in graph.variables if G.in_degree(v) == 0}
    leaves = {v for v in graph.variables if G.out_degree(v) == 0}
    middle = set(graph.variables) - roots - leaves

    # Draw nodes by type
    if roots:
        nx.draw_networkx_nodes(
            G, pos, nodelist=list(roots),
            node_color=COLORS["node_root"],
            node_size=800, ax=ax,
        )
    if leaves:
        nx.draw_networkx_nodes(
            G, pos, nodelist=list(leaves),
            node_color=COLORS["node_leaf"],
            node_size=800, ax=ax,
        )
    if middle:
        nx.draw_networkx_nodes(
            G, pos, nodelist=list(middle),
            node_color=COLORS["node"],
            node_size=800, ax=ax,
        )

    # Draw edges
    directed_edges = []
    undirected_edges = []
    highlighted = []

    for e in graph.edges:
        if (e.source, e.target) in highlight_set:
            highlighted.append((e.source, e.target))
        elif e.edge_type == EdgeType.DIRECTED:
            directed_edges.append((e.source, e.target))
        else:
            undirected_edges.append((e.source, e.target))

    if directed_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=directed_edges,
            edge_color=COLORS["edge_directed"],
            width=2, arrows=True, arrowsize=20,
            connectionstyle="arc3,rad=0.1", ax=ax,
        )
    if undirected_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=undirected_edges,
            edge_color=COLORS["edge_undirected"],
            width=1.5, style="dashed", arrows=False, ax=ax,
        )
    if highlighted:
        nx.draw_networkx_edges(
            G, pos, edgelist=highlighted,
            edge_color="#E74C3C",
            width=3, arrows=True, arrowsize=25,
            connectionstyle="arc3,rad=0.1", ax=ax,
        )

    # Labels
    labels = {v: v.replace("_", "\n") for v in graph.variables}
    nx.draw_networkx_labels(
        G, pos, labels, font_size=10,
        font_weight="bold", ax=ax,
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(output_path)


def _edge_tooltip(edge) -> str:
    """Generate hover tooltip for an edge."""
    parts = [f"{edge.source} → {edge.target}"]
    if edge.weight is not None:
        parts.append(f"weight: {edge.weight:.4f}")
    if edge.p_value is not None:
        parts.append(f"p-value: {edge.p_value:.4f}")
    parts.append(f"type: {edge.edge_type.value}")
    return "\n".join(parts)


def find_causal_path(
    graph: CausalGraph,
    source: str,
    target: str,
) -> Optional[List[str]]:
    """
    Find a directed path from source to target in the causal graph.

    Returns None if no path exists.
    """
    G = to_networkx(graph)
    try:
        path = nx.shortest_path(G, source, target)
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
