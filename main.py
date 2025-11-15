import networkx as nx
import itertools
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pysat.solvers import Glucose3
import matplotlib.pyplot as plt

# === CONFIGURATION ===
MAX_VERTICES = 7  # You can change to 7, 6, etc.
#OUTPUT_FILE = "three_colorable_graphs.txt"
SAVE_DIR = "three_colorable_graphs"
SHOW_PLOTS = False  # Set to True to open plots while running

os.makedirs(SAVE_DIR, exist_ok=True)

def is_three_colorable_sat(graph):
    """Checks if the graph is 3-colorable using SAT."""
    g = Glucose3()
    n = graph.number_of_nodes()
    nodes = list(graph.nodes)
    var = lambda v, c: v * 3 + c + 1  # Color c in {0,1,2} for node v

    for v in nodes:
        g.add_clause([var(v, 0), var(v, 1), var(v, 2)])  # At least one color

    for v in nodes:
        for c1 in range(3):
            for c2 in range(c1 + 1, 3):
                g.add_clause([-var(v, c1), -var(v, c2)])  # At most one color

    for u, v in graph.edges():
        for c in range(3):
            g.add_clause([-var(u, c), -var(v, c)])  # Adjacent nodes ≠ same color

    return g.solve()

def has_at_most_one_triangle(graph):
    """Returns True if the graph has 0 or 1 triangle."""
    return sum(nx.triangles(graph).values()) // 3 <= 1

def generate_all_graphs(n):
    """Generates all non-isomorphic graphs with n nodes and ≤1 triangle."""
    nodes = list(range(n))
    all_edges = list(itertools.combinations(nodes, 2))
    graphs_set = []
    seen_canonical_forms = set()

    for edges_included in itertools.product([0, 1], repeat=len(all_edges)):
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from([e for include, e in zip(edges_included, all_edges) if include])

        if not has_at_most_one_triangle(g):
            continue

        canonical_form = nx.weisfeiler_lehman_graph_hash(g)
        if canonical_form not in seen_canonical_forms:
            seen_canonical_forms.add(canonical_form)
            graphs_set.append(g)

    return graphs_set

def check_and_return_graph(graph_data):
    """Return graph if 3-colorable, else None."""
    idx, g = graph_data
    return g if is_three_colorable_sat(g) else None

def visualize_and_save(graph, idx):
    """Visualizes and saves a graph as PNG."""
    plt.figure(figsize=(3, 3))
    pos = nx.spring_layout(graph, seed=idx)
    nx.draw(graph, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=600)
    plt.title(f"Graph {idx}")
    plt.axis('off')
    filename = os.path.join(SAVE_DIR, f"graph_{idx}.png")
    plt.savefig(filename)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

#Find all threee colorable graphs with at most 1 triangle
def find_three_colorable_graphs(max_n):
    result = []
    idx = 0
    for n in range(1, max_n + 1):
        print(f"\nProcessing graphs with {n} vertices...")
        all_graphs = generate_all_graphs(n)
        for g in tqdm(all_graphs):
            if is_three_colorable_sat(g):
                result.append(g)
                visualize_and_save(g, idx)
                idx += 1
    return result

def satisfies_embeddable_property_some_coloring(graph):
    """
    Returns True iff there exists at least one proper 3-coloring
    for which the deletion process ends in an empty graph or triangle.
    """
    nodes = list(graph.nodes)
    for coloring_tuple in itertools.product(range(3), repeat=len(nodes)):
        coloring = dict(zip(nodes, coloring_tuple))

        # Check if coloring is proper (no edge has same color endpoints)
        if any(coloring[u] == coloring[v] for u, v in graph.edges):
            continue  # Not proper, skip

        G = graph.copy()
        while True:
            deletable_nodes = []
            for v in G.nodes:
                neighbors = list(G.neighbors(v))
                if not neighbors:
                    deletable_nodes.append(v)
                    continue
                neighbor_colors = {coloring[u] for u in neighbors}
                if len(neighbor_colors) == 1:  # Monochromatic neighborhood
                    deletable_nodes.append(v)

            if not deletable_nodes:
                break  # No nodes to delete, stop

            G.remove_nodes_from(deletable_nodes)

        if G.number_of_nodes() == 0 or (G.number_of_nodes() == 3 and G.number_of_edges() == 3):
            return True  # Found one coloring that works

    # Print the input graph that doesn't satisfy the embeddable property
    print(f"Nodes: {graph.number_of_nodes()}\n")
    print(f"Edges: {graph.number_of_edges()}\n")
    print(f"Edge List: {list(graph.edges())}\n\n")
    return False  # No coloring satisfies the embeddable property

# === MAIN EXECUTION ===
#if __name__ == "__main__":

colorable_graphs = find_three_colorable_graphs(MAX_VERTICES)

 # === SAVE RESULTS TO TEXT FILE ===
    #with open(OUTPUT_FILE, "w") as f:
        #f.write(f"3-colorable graphs with less than {MAX_VERTICES} vertices and at most 1 triangle\n")
        #f.write(f"Total: {len(colorable_graphs)}\n\n")
        #for i, g in enumerate(colorable_graphs):
            #f.write(f"Graph {i+1}:\n")
            #f.write(f"Nodes: {g.number_of_nodes()}\n")
            #f.write(f"Edges: {g.number_of_edges()}\n")
            #f.write(f"Edge List: {list(g.edges())}\n\n")

    #print(f"\n✅ Done! Saved {len(colorable_graphs)} graphs to '{OUTPUT_FILE}'.")

print(f"\n✅ Found and saved {len(colorable_graphs)} 3-colorable graphs with ≤{MAX_VERTICES} vertices and ≤1 triangle.")
deletion_some_colorable = [
    g for g in colorable_graphs if satisfies_embeddable_property_some_coloring(g)
]
print(f"\n🧠 {len(deletion_some_colorable)} graphs satisfy the deletion property for at least one 3-coloring.")
