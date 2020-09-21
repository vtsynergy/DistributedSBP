import argparse
import csv
import os
import sys
from typing import Dict, Optional, List, Tuple

from graph_tool import Graph
from graph_tool import load_graph_from_csv
from graph_tool.inference import BlockState
import numpy as np
import pandas as pd


def load_graph(args: argparse.Namespace) -> Tuple[Graph, np.ndarray]:
    """Loads the graph and the truth partition.

    Parameters
    ----------
    args : argparse.Namespace
        the command-line arguments passed to the program

    Returns
    -------
    graph : Graph
        the loaded graph
    assignment : np.ndarray[int]
        the true vertex-to-community membership array
    """
    input_filename = build_filepath(args)
    if args.gtload:
        graph = load_graph_from_csv(input_filename + ".tsv", not args.undirected,
                                    csv_options={'delimiter': args.delimiter})
    else:
        graph = _load_graph(input_filename)
    print(graph)
    true_membership = load_true_membership(input_filename, graph.num_vertices())

    if args.verbose:
        print('Number of vertices: {}'.format(graph.num_vertices()))
        print('Number of edges: {}'.format(graph.num_edges()))
    if args.degrees:
        save_degree_distribution(args, graph)
    return graph, true_membership
# End of load_graph()


def build_filepath(args: argparse.Namespace) -> str:
    """Builds the filename string.

    Parameters
    ---------
    args : argparse.Namespace
        the command-line arguments passed to the program

    Returns
    ------
    filepath : str
        the path to the dataset base directory
    """
    filepath_base = "{0}/{1}/{2}Overlap_{3}BlockSizeVar/{1}_{2}Overlap_{3}BlockSizeVar_{4}_nodes".format(
        args.directory, args.type, args.overlap, args.blockSizeVar, args.numNodes
    )

    if not os.path.isfile(filepath_base + '.tsv') and not os.path.isfile(filepath_base + '_1.tsv'):
        print("File doesn't exist: '{}'!".format(filepath_base))
        sys.exit(1)

    return filepath_base
# End of build_filepath()


def _load_graph(input_filename: str, part_num: Optional[int] = None, graph: Optional[Graph] = None) -> Graph:
    """Load the graph from a TSV file with standard format.

    Parameters
    ----------
    input_filename : str
        input file name not including the .tsv extension
    part_num : int, optional
        specify which stage of the streaming graph to load
    graph : Graph, optional
        existing graph to add to. This is used when loading the streaming graphs one stage at a time. Note that
        the truth partition is loaded all together at once.

    Returns
    -------
    graph : Graph
        the Graph object loaded or updated from file

    Notes
    -----
    The standard tsv file has the form for each row: "from to [weight]" (tab delimited). Nodes are indexed from 0
    to N-1.
    """
    # read the entire graph CSV into rows of edges
    if part_num is None:
        edge_rows = pd.read_csv('{}.tsv'.format(input_filename), delimiter='\t', header=None).values
    else:
        edge_rows = pd.read_csv('{}_{}.tsv'.format(input_filename, part_num), delimiter='\t', header=None).values

    if graph is None:  # no previously loaded streaming pieces
        N = edge_rows[:, 0:2].max()  # number of nodes
        out_neighbors = [[] for i in range(N)]  # type: List[np.ndarray[int]]
        in_neighbors = [[] for i in range(N)]  # type: List[np.ndarray[int]]
    else:  # add to previously loaded streaming pieces
        N = max(edge_rows[:, 0:2].max(), len(
            graph.out_neighbors))  # number of nodes
        out_neighbors = [list(graph.out_neighbors[i])
                         for i in range(len(graph.out_neighbors))]
        out_neighbors.extend([[] for i in range(N - len(out_neighbors))])
        in_neighbors = [list(graph.in_neighbors[i])
                        for i in range(len(graph.in_neighbors))]
        in_neighbors.extend([[] for i in range(N - len(in_neighbors))])
    weights_included = edge_rows.shape[1] == 3

    # load edges to list of lists of out and in neighbors
    for i in range(edge_rows.shape[0]):
        if weights_included:
            edge_weight = edge_rows[i, 2]
        else:
            edge_weight = 1
        # -1 on the node index since Python is 0-indexed and the standard graph TSV is 1-indexed
        out_neighbors[edge_rows[i, 0] - 1].append([edge_rows[i, 1] - 1, edge_weight])
        in_neighbors[edge_rows[i, 1] - 1].append([edge_rows[i, 0] - 1, edge_weight])

    # convert each neighbor list to neighbor numpy arrays for faster access
    for i in range(N):
        if len(out_neighbors[i]) > 0:
            out_neighbors[i] = np.array(out_neighbors[i], dtype=np.int32)
        else:
            out_neighbors[i] = np.array(
                out_neighbors[i], dtype=np.int32).reshape((0, 2))
    for i in range(N):
        if len(in_neighbors[i]) > 0:
            in_neighbors[i] = np.array(in_neighbors[i], dtype=np.int32)
        else:
            in_neighbors[i] = np.array(
                in_neighbors[i], dtype=np.int32).reshape((0, 2))

    # E = sum(len(v) for v in out_neighbors)  # number of edges

    input_graph = Graph()
    input_graph.add_vertex(N)
    input_graph.add_edge_list(
        [(i, j) for i in range(len(out_neighbors)) if len(out_neighbors[i]) > 0 for j in out_neighbors[i][:, 0]]
    )
    return input_graph
# End of _load_graph()


def load_true_membership(input_filename: str, num_vertices: int) -> np.ndarray:
    """Loads the true community membership from the true membership file.

    Parameters
    ----------
    input_filename : str
        the path to the dataset
    num_vertices : int
        the number of vertices in the graph

    Returns
    -------
    true_membership : np.ndarray[int]
        the true block membership for every vertex in the graph

    Notes
    -----
    The true partition is stored in the file `filename_truePartition.tsv`.
    """
    filename = "{}_truePartition.tsv".format(input_filename)
    if not os.path.exists(filename):
        print("true partition at {} does not exist".format(filename))
        true_b = np.asarray([-1] * num_vertices)
        return true_b
    # read the entire true partition CSV into rows of partitions
    true_b_rows = pd.read_csv(filename, delimiter='\t', header=None).values
    # initialize truth assignment to -1 for 'unknown'
    true_b = np.ones(true_b_rows.shape[0], dtype=int) * -1
    for i in range(true_b_rows.shape[0]):
        # -1 since Python is 0-indexed and the TSV is 1-indexed
        true_b[true_b_rows[i, 0] - 1] = int(true_b_rows[i, 1] - 1)
    print("true membership: {} [{},{}]".format(len(true_b), np.min(true_b), np.max(true_b)))
    return true_b
# End of load_true_membership()


def save_degree_distribution(args: argparse.Namespace, graph: Graph):
    """Saves the in and out degrees of all vertices in the graph.

    Parameters
    ----------
    args : argparse.Namespace
        the command-line arguments provided
    graph : Graph
        the graph object
    """
    write_header = False
    if not os.path.isfile(args.csv + ".csv"):
        directory = os.path.dirname(args.csv + ".csv")
        if directory not in [".", ""]:
            os.makedirs(directory, exist_ok=True)
        write_header = True
    num_vertices = [i for i in range(graph.num_vertices())]
    out_degrees = [v.out_degree() for v in graph.vertices()]
    in_degrees = [v.in_degree() for v in graph.vertices()]
    with open(args.csv + ".csv", "a") as details_file:
        writer = csv.writer(details_file)
        if write_header:
            writer.writerow(["Num Vertices", "In/Out", "Degree"])
        for degree in out_degrees:
            writer.writerow([num_vertices, "Out", degree])
        for degree in in_degrees:
            writer.writerow([num_vertices, "In", degree])
    exit()
# End of save_degree_distribution()


def partition_from_sample(sample_partition: BlockState, graph: Graph, mapping: Dict) -> BlockState:
    """Creates a BlockState for the full graph from sample results.

    Parameters
    ----------
    sample_partition : BlockState
        the partitioning results on the sample subgraph
    graph : Graph
        the full graph
    mapping : Dict[int, int]
        the mapping of sample vertices to full graph vertices

    Returns
    -------
    partition : BlockState
        the extended partition on the full graph
    """
    assignment_property_map = graph.new_vertex_property("int", val=-1)
    assignment = assignment_property_map.get_array()
    sample_assignment = sample_partition.get_blocks().get_array()
    for key, value in mapping.items():
        assignment[key] = sample_assignment[value]
    next_block = sample_partition.get_B()
    for i in range(graph.num_vertices()):
        if assignment[i] >= 0:
            continue
        assignment[i] = next_block
        next_block += 1
    for i in range(graph.num_vertices()):
        if assignment[i] < sample_partition.get_B():
            continue
        # TODO: make a helper function that finds the most connected block
        # TODO: make this an iterative process, so vertices with all neighbors unknown wait until at least one neighbor
        # is known
        block_counts = dict()  # type: Dict[int, int]
        neighbors = graph.get_all_neighbors(i)
        for neighbor in neighbors:
            neighbor_block = assignment[neighbor]
            if neighbor_block < sample_partition.get_B():
                if neighbor_block in block_counts:
                    block_counts[neighbor_block] += 1
                else:
                    block_counts[neighbor_block] = 1
        if len(block_counts) > 0:
            assignment[i] = max(block_counts)
        else:
            assignment[i] = np.random.randint(0, sample_partition.get_B())
    assignment_property_map.a = assignment
    partition = BlockState(graph, assignment_property_map, sample_partition.get_B())
    return partition
# End of partition_from_sample()


def partition_from_truth(graph: Graph, true_membership: np.ndarray) -> BlockState:
    """Creates the BlockState for a graph using the true community membership.

    Parameters
    ----------
    graph : Graph
        the graph for which the partiion is being created
    true_membership : np.ndarray[int]
        the true vertex-to-community membership array

    Returns
    -------
    partition : BlockState
        the partition created from the true community membership for the given graph

    Notes
    -----
    Assumes that that the community membership is non-overlapping.
    """
    assignment_property_map = graph.new_vertex_property("int", val=-1)
    assignment_property_map.a = true_membership
    partition = BlockState(graph, assignment_property_map, np.max(true_membership) + 1)
    return partition
# End of partition_from_truth()


def finetune_assignment(partition: BlockState, args: argparse.Namespace) -> BlockState:
    """Finetunes the block assignment of a given partition.

    Parameters
    ----------
    partition : BlockState
        the current state of the graph partition

    Returns
    -------
    finetuned_partition : BlockState
        the finetuned state of the graph partition
    """
    start_entropy = partition.entropy()
    delta_entropy = np.Inf
    total_attempts = 0
    total_moves = 0
    total_iterations = 0
    if partition.get_B() == 1:
        return partition
    for i in range(100):
        entropy = partition.entropy()
        delta_entropy, nattempts, nmoves = partition.mcmc_sweep(d=0, allow_vacate=False)
        if args.verbose:
            print("dE: {} attempts: {} moves: {} E: {}".format(delta_entropy, nattempts, nmoves, entropy))
        total_attempts += nattempts
        total_moves += nmoves
        total_iterations += 1
        if -delta_entropy < 1e-4 * entropy:
            break
    print("Finetuning results: {} iterations | {} attempts | {} moves | delta entropy = {}".format(
        total_iterations, total_attempts, total_moves, partition.entropy() - start_entropy
    ))
    return partition
# End of finetune_assignment()
