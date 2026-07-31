"""Task #344 — KG Phase 1 build (NetworkX graph + matplotlib visualization)

读取 kg/extracted/*.json 手动抽取, 构建有向图, 输出:
- kg/kg_graph.json (NetworkX node-link format)
- kg/kg_statistics.json (节点/边统计)
- kg/kg_visualization.png (matplotlib 可视化)
"""
import json
import sys
import os
from pathlib import Path

REPO = Path('/home/wlia0047/ar57/wenyu/GeneRec')
KG_DIR = REPO / 'kg'
EXTRACTED_DIR = KG_DIR / 'extracted'

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Color mapping for entity types
ENTITY_COLORS = {
    'Verdict': '#e74c3c',       # red
    'Issue': '#3498db',         # blue
    'Task': '#2ecc71',          # green
    'Method': '#9b59b6',        # purple
    'Hyperparameter': '#f39c12', # orange
    'Metric': '#1abc9c',        # teal
    'RootCause': '#e67e22',     # dark orange
    'Decision': '#34495e',      # dark gray
    'Stage': '#16a085',         # dark teal
    'Gate': '#8e44ad',          # dark purple
    'Hypothesis': '#d35400',    # dark orange
    'Paper': '#7f8c8d',         # gray
}
DEFAULT_COLOR = '#95a5a6'  # gray for unknown


def load_extracted_files():
    """Load all manually extracted verdict JSON files."""
    extractions = {}
    for f in EXTRACTED_DIR.glob('verdict_*.json'):
        with open(f) as fp:
            data = json.load(fp)
        extractions[f.stem] = data
        print(f"  Loaded {f.name}: {len(data.get('entities', {}))} entity types, "
              f"{len(data.get('relationships', []))} relationships")
    return extractions


def build_graph(extractions):
    """Build NetworkX DiGraph from extracted entities + relationships."""
    G = nx.DiGraph()

    # Add entities (deduplicate by ID across extractions)
    for ext_id, ext_data in extractions.items():
        for entity_type, entity_list in ext_data.get('entities', {}).items():
            for entity in entity_list:
                eid = entity['id']
                # Add node with type + attributes
                if eid in G:
                    # Merge: update attrs but keep first type seen
                    G.nodes[eid].update({k: v for k, v in entity.items() if k not in G.nodes[eid]})
                else:
                    G.add_node(eid, type=entity_type, **entity)
                    G.nodes[eid]['source_verdicts'] = [ext_id]
                if ext_id not in G.nodes[eid].get('source_verdicts', []):
                    G.nodes[eid].setdefault('source_verdicts', []).append(ext_id)

    # Add edges (relationships)
    for ext_id, ext_data in extractions.items():
        for rel in ext_data.get('relationships', []):
            from_id = rel['from']
            to_id = rel['to']
            rel_type = rel['type']
            if G.has_edge(from_id, to_id):
                # Multi-edge: append rel type
                existing_types = G.edges[from_id, to_id].get('rel_types', [])
                if rel_type not in existing_types:
                    existing_types.append(rel_type)
                G.edges[from_id, to_id]['rel_types'] = existing_types
            else:
                G.add_edge(from_id, to_id, rel_type=rel_type, rel_types=[rel_type])

    return G


def compute_statistics(G):
    """Compute KG statistics by entity type and relationship type."""
    stats = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'nodes_by_type': {},
        'edges_by_type': {},
        'multi_edge_pairs': [],
        'orphan_nodes': [],
    }

    for node_id, attrs in G.nodes(data=True):
        ntype = attrs.get('type', 'Unknown')
        stats['nodes_by_type'][ntype] = stats['nodes_by_type'].get(ntype, 0) + 1

    for u, v, attrs in G.edges(data=True):
        rel_types = attrs.get('rel_types', [attrs.get('rel_type', 'unknown')])
        for rt in rel_types:
            stats['edges_by_type'][rt] = stats['edges_by_type'].get(rt, 0) + 1
        if len(rel_types) > 1:
            stats['multi_edge_pairs'].append({'from': u, 'to': v, 'rel_types': rel_types})

    # Orphan nodes (no edges)
    for node_id in G.nodes():
        if G.degree(node_id) == 0:
            stats['orphan_nodes'].append(node_id)

    return stats


def visualize_graph(G, output_path):
    """Render KG to matplotlib PNG."""
    # Use spring layout with seed for reproducibility
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)

    # Color nodes by entity type
    node_colors = []
    node_labels = {}
    for node_id, attrs in G.nodes(data=True):
        ntype = attrs.get('type', 'Unknown')
        node_colors.append(ENTITY_COLORS.get(ntype, DEFAULT_COLOR))
        # Use short ID (last part after underscore) for readability
        short_id = node_id.split('_')[-1] if '_' in node_id else node_id
        node_labels[node_id] = short_id

    # Edge styles: solid for single relationship, dashed for multi-relationship
    edge_styles = []
    for u, v, attrs in G.edges(data=True):
        if len(attrs.get('rel_types', [])) > 1:
            edge_styles.append('dashed')
        else:
            edge_styles.append('solid')

    plt.figure(figsize=(20, 14))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color='#7f8c8d', style=edge_styles,
                           arrows=True, arrowsize=15, width=1.2, alpha=0.6)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7, font_weight='bold')

    # Build legend
    legend_handles = []
    used_types = set(attrs.get('type', 'Unknown') for _, attrs in G.nodes(data=True))
    for ntype in sorted(used_types):
        color = ENTITY_COLORS.get(ntype, DEFAULT_COLOR)
        count = sum(1 for _, a in G.nodes(data=True) if a.get('type', 'Unknown') == ntype)
        legend_handles.append(mpatches.Patch(color=color, label=f'{ntype} ({count})'))

    plt.legend(handles=legend_handles, loc='upper left', fontsize=10, framealpha=0.9)
    plt.title(f'GeneRec KG (Phase 1 Pilot) — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges',
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {output_path}")


def print_statistics(stats):
    """Print KG statistics in human-readable format."""
    print("\n" + "=" * 70)
    print("KG STATISTICS")
    print("=" * 70)
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"\n  Nodes by type:")
    for ntype, count in sorted(stats['nodes_by_type'].items(), key=lambda x: -x[1]):
        print(f"    {ntype}: {count}")
    print(f"\n  Edges by type (top 15):")
    for rel_type, count in sorted(stats['edges_by_type'].items(), key=lambda x: -x[1])[:15]:
        print(f"    {rel_type}: {count}")
    if stats['multi_edge_pairs']:
        print(f"\n  Multi-edge pairs (multiple relationships):")
        for pair in stats['multi_edge_pairs']:
            print(f"    {pair['from']} -> {pair['to']}: {pair['rel_types']}")
    if stats['orphan_nodes']:
        print(f"\n  Orphan nodes (no edges): {len(stats['orphan_nodes'])}")
        for node in stats['orphan_nodes']:
            print(f"    {node}")


def main():
    print("=" * 70)
    print("Task #344 — KG Phase 1 build (NetworkX + matplotlib)")
    print("=" * 70)

    print("\n[1/4] Loading extracted verdicts...")
    extractions = load_extracted_files()
    if not extractions:
        print("❌ No extracted files found in kg/extracted/")
        return 1

    print("\n[2/4] Building NetworkX DiGraph...")
    G = build_graph(extractions)
    print(f"  Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n[3/4] Computing statistics...")
    stats = compute_statistics(G)

    # Save outputs
    print("\n[4/4] Saving outputs...")
    # Graph node-link JSON (compatible with NetworkX 3.x)
    graph_path = KG_DIR / 'kg_graph.json'
    from networkx.readwrite import json_graph
    graph_data = json_graph.node_link_data(G)
    with open(graph_path, 'w') as fp:
        json.dump(graph_data, fp, indent=2, ensure_ascii=False, default=str)
    print(f"  Graph JSON: {graph_path}")

    # Statistics
    stats_path = KG_DIR / 'kg_statistics.json'
    with open(stats_path, 'w') as fp:
        json.dump(stats, fp, indent=2, ensure_ascii=False, default=str)
    print(f"  Statistics: {stats_path}")

    # Visualization
    viz_path = KG_DIR / 'kg_visualization.png'
    visualize_graph(G, str(viz_path))

    # Print statistics
    print_statistics(stats)

    print("\n✅ KG Phase 1 build complete")
    print(f"   Open {viz_path} to see the visualization")
    return 0


if __name__ == '__main__':
    sys.exit(main())