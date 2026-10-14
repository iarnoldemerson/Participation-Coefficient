import warnings
warnings.filterwarnings('ignore')

#!pip install -q biopython networkx leidenalg streamlit

import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
import networkx as nx
import matplotlib.pyplot as plt
import leidenalg as la
import igraph as ig
import streamlit as st
from io import StringIO, BytesIO # Added BytesIO for Excel export

def get_alpha_carbon_coords(pdb_file, target_chains=None):
    """
    Parses a PDB file and extracts the alpha-carbon coordinates for each residue.
    Optionally filters by specified chain IDs.

    Args:
        pdb_file (str): Path to the PDB file or file-like object.
        target_chains (list, optional): A list of chain IDs to process (e.g., ['A', 'B']).
                                        If None, all chains are processed.

    Returns:
        list: A list of dictionaries, each containing residue ID, name, and Cα coordinates.
    """
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)

    coords = []
    for model in structure:
        all_chains_in_pdb = sorted(list(set(chain.id for chain in model)))

        # If no target_chains specified, process all chains found in the PDB
        chains_to_process = target_chains if target_chains is not None else all_chains_in_pdb

        for chain in model:
            if chain.id in chains_to_process:
                for residue in chain:
                    # Only consider standard amino acids and residues with a Cα atom
                    if residue.has_id('CA') and residue.get_resname() not in ['HOH', 'WAT', 'UNL']:
                        ca_atom = residue['CA']
                        coords.append({
                            'residue_id': (chain.id, residue.id[1], residue.id[2]),
                            'residue_name': residue.get_resname(),
                            'chain_id': chain.id,
                            'coordinates': ca_atom.get_coord()
                        })
    return coords

def build_contact_network(ca_coords, cutoff=7.5):
    """
    Builds a protein contact network from alpha-carbon coordinates.

    Args:
        ca_coords (list): List of dictionaries containing residue info and Cα coordinates.
        cutoff (float): Distance cutoff in Angstroms for establishing an edge.

    Returns:
        networkx.Graph: A graph representing the protein contact network.
    """
    G = nx.Graph()

    # Add nodes
    for i, res_data in enumerate(ca_coords):
        node_id = f"{res_data['chain_id']}_{res_data['residue_name']}_{res_data['residue_id'][1]}"
        G.add_node(node_id, **res_data)

    # Add edges based on distance cutoff
    for i in range(len(ca_coords)):
        node_i_data = ca_coords[i]
        node_i_id = f"{node_i_data['chain_id']}_{node_i_data['residue_name']}_{node_i_data['residue_id'][1]}"
        coord_i = node_i_data['coordinates']

        for j in range(i + 1, len(ca_coords)):
            node_j_data = ca_coords[j]
            node_j_id = f"{node_j_data['chain_id']}_{node_j_data['residue_name']}_{node_j_data['residue_id'][1]}"
            coord_j = node_j_data['coordinates']

            distance = np.linalg.norm(coord_i - coord_j)
            if distance <= cutoff:
                G.add_edge(node_i_id, node_j_id, distance=distance)

    return G

# Function to convert DataFrame to Excel format
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

# --- Streamlit Application ---
st.set_page_config(layout="wide")
st.title("Protein Contact Network Analysis with Leiden Communities")

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload a PDB file", type=["pdb"])
    cutoff = st.slider("Distance Cutoff (Angstroms)", min_value=1.0, max_value=15.0, value=7.5, step=0.5)


if uploaded_file is not None:
    # Read the content of the uploaded PDB file
    pdb_content = uploaded_file.read().decode("utf-8")

    # To parse with Bio.PDB, we need a file-like object or a path
    # We'll write it to a temporary file, or use StringIO
    pdb_file_like = StringIO(pdb_content)

    # Get all chains present in the uploaded PDB for selection
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file_like)
    all_chains = sorted(list(set(chain.id for model in structure for chain in model)))
    pdb_file_like.seek(0) # Reset pointer after reading for chains

    selected_chains = st.sidebar.multiselect(
        "Select Chain(s) to Analyze (leave empty for all)",
        options=all_chains,
        default=all_chains # Default to all chains initially
    )

    if not selected_chains:
        st.warning("Please select at least one chain to proceed with the analysis, or leave empty to analyze all chains.")
        ca_coordinates = get_alpha_carbon_coords(pdb_file_like, target_chains=None) # Process all chains if none explicitly selected
    else:
        ca_coordinates = get_alpha_carbon_coords(pdb_file_like, target_chains=selected_chains)

    if not ca_coordinates:
        st.warning("No alpha carbon coordinates found for the selected chain(s) or PDB file.")
    else:
        st.subheader("Protein Contact Network Analysis")

        contact_network = build_contact_network(ca_coordinates, cutoff=cutoff)

        st.write(f"Number of nodes in the network: {contact_network.number_of_nodes()}")
        st.write(f"Number of edges in the network: {contact_network.number_of_edges()}")

        if contact_network.number_of_nodes() > 1 and contact_network.number_of_edges() > 0:
            # Convert networkx graph to igraph for Leiden algorithm
            ig_graph = ig.Graph.from_networkx(contact_network)

            # Run Leiden algorithm
            partition = la.find_partition(ig_graph, la.ModularityVertexPartition)

            # Calculate modularity
            modularity_score = partition.modularity
            st.write(f"Modularity score: {modularity_score:.4f}")

            # Map communities back to networkx graph nodes, starting from 1
            node_to_community = {ig_graph.vs[i]['_nx_name']: partition.membership[i] + 1 for i in range(ig_graph.vcount())}
            nx.set_node_attributes(contact_network, node_to_community, 'community')

            # Calculate Participation Coefficient (P_i)
            participation_coefficients = {}
            all_communities = set(nx.get_node_attributes(contact_network, 'community').values())

            for node_id in contact_network.nodes():
                node_degree = contact_network.degree(node_id)
                if node_degree == 0: # Avoid division by zero for isolated nodes
                    participation_coefficients[node_id] = 0.0
                    continue

                community_links = {comm: 0 for comm in all_communities}
                for neighbor in contact_network.neighbors(node_id):
                    neighbor_community = contact_network.nodes[neighbor]['community']
                    if neighbor_community in community_links:
                        community_links[neighbor_community] += 1

                sum_ratios_squared = 0
                for comm in all_communities:
                    k_is = community_links[comm]
                    if k_is > 0:
                        sum_ratios_squared += (k_is / node_degree)**2

                P_i = 1 - sum_ratios_squared
                participation_coefficients[node_id] = P_i

            nx.set_node_attributes(contact_network, participation_coefficients, 'participation_coefficient')

            # Prepare data for DataFrame
            output_data = []
            for node_id, attributes in contact_network.nodes(data=True):
                chain_id = attributes['chain_id']
                residue_name = attributes['residue_name']
                residue_id = attributes['residue_id'][1]
                community = attributes['community']
                p_coefficient = attributes['participation_coefficient']
                output_data.append([chain_id, residue_name, residue_id, community, p_coefficient])

            df_communities_pi = pd.DataFrame(output_data, columns=['Chain ID', 'Residue Name', 'Residue ID', 'Community', 'Participation Coefficient'])
            st.subheader("Community Assignments and Participation Coefficients")
            st.dataframe(df_communities_pi)

            # Add download button for the DataFrame
            st.download_button(
                label="Download table as Excel",
                data=to_excel(df_communities_pi),
                file_name='community_assignments.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

            # Visualize the network
            st.subheader("Protein Contact Network Visualization")
            fig, ax = plt.subplots(figsize=(12, 10))

            communities_for_plot = [contact_network.nodes[node]['community'] for node in contact_network.nodes()]
            unique_communities_for_plot = sorted(list(set(communities_for_plot)))
            #colors = plt.cm.get_cmap('tab20', len(unique_communities_for_plot))
            colors = plt.cm.get_cmap('tab20')(range(len(unique_communities_for_plot)))
            node_colors = [colors[unique_communities_for_plot.index(c)] for c in communities_for_plot]

            pos = nx.spring_layout(contact_network, seed=42) # Consistent layout
            nx.draw_networkx_nodes(contact_network, pos, node_color=node_colors, node_size=300, alpha=0.9, ax=ax)
            nx.draw_networkx_edges(contact_network, pos, edge_color='gray', width=0.5, alpha=0.6, ax=ax)

            node_labels = {node: node.split('_')[1] + str(node.split('_')[2]) for node in contact_network.nodes()}
            nx.draw_networkx_labels(contact_network, pos, labels=node_labels, font_size=6, font_color='black', ax=ax)

            ax.set_title(f"Protein Contact Network with Leiden Communities (Cutoff: {cutoff} Å)")
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("The generated network has too few nodes or no edges to perform community detection and visualization.")

else:
    st.info("Please upload a PDB file to start the analysis.")
