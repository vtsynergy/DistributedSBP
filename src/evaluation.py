"""Module containing the Evaluation class, which stores evaluation results and saves them to file.
"""

import os
import csv

from typing import List, Dict
from argparse import Namespace

from graph_tool import Graph
from graph_tool.clustering import global_clustering
from graph_tool.inference import BlockState
from graph_tool.topology import pseudo_diameter, extract_largest_component
import numpy as np

from util import partition_from_truth


class Evaluation(object):
    """Stores the evaluation results, and saves them to file.
    """

    FIELD_NAMES = [
        'block size variation',
        'block overlap',
        'streaming type',
        'num vertices',
        'num edges',
        'blocks retained (%)',
        'within to between edge ratio',
        'difference from ideal sample',
        'expansion quality',
        'sampled graph clustering coefficient',
        'full graph clustering coefficient',
        'sampled graph diameter',
        'full graph diameter',
        'sampled graph largest component',
        'full graph largest component',
        'sampled graph island vertices',
        'sampled graph num vertices',
        'sampled graph num edges',
        'sampled graph within to between edge ratio',
        'sampled graph num blocks in algorithm partition',
        'sampled graph num blocks in truth partition',
        'sampled graph accuracy',
        'sampled graph rand index',
        'sampled graph adjusted rand index',
        'sampled graph pairwise recall',
        'sampled graph pairwise precision',
        'sampled graph entropy of algorithm partition',
        'sampled graph entropy of truth partition',
        'sampled graph entropy of algorithm partition given truth partition',
        'sampled graph entropy of truth partition given algorithm partition',
        'sampled graph mutual information',
        'sampled graph fraction of missed information',
        'sampled graph fraction of erroneous information',
        'num block proposals',
        'beta',
        'sample size (%)',
        'sampling iterations',
        'sampling algorithm',
        'delta entropy threshold',
        'nodal update threshold strategy',
        'nodal update threshold factor',
        'nodal update threshold direction',
        'num blocks in algorithm partition',
        'num blocks in truth partition',
        'accuracy',
        'rand index',
        'adjusted rand index',
        'pairwise recall',
        'pairwise precision',
        'entropy of algorithm partition',
        'entropy of truth partition',
        'entropy of algorithm partition given truth partition',
        'entropy of truth partition given algorithm partition',
        'mutual information',
        'fraction of missed information',
        'fraction of erroneous information',
        'sampled graph description length',
        'max sampled graph description length',
        'full graph description length',
        'max full graph description length',
        'sampled graph modularity',
        'full graph modularity',
        'graph loading time',
        'sampling time',
        'sampled graph partition time',
        'total partition time',
        'merging partitioned sample time',
        'cluster propagation time',
        'finetuning membership time',
        'tag'
    ]

    DETAILS_FIELD_NAMES = [
        'num vertices',
        'community overlap',
        'community size variation',
        'community 1 id',
        'sampled_graph',
        'community 2 id',
        'size',
        'algorithm',
        'sample_size',
        'sample_algorithm',
        'sample_iterations',
        'tag'
    ]

    def __init__(self, args: Namespace, graph: Graph) -> None:
        """Creates a new Evaluation object.

        Parameters
        ----------
        args : Namespace
            the command-line arguments
        graph : Graph
            the loaded graph to be partitioned
        """
        # CSV file into which to write the results
        self.args = args
        self.csv_file = args.csv + ".csv"
        self.csv_details_file = args.csv + "_details.csv"
        # Dataset parameters
        self.block_size_variation = args.blockSizeVar
        self.block_overlap = args.overlap
        self.streaming_type = args.type
        self.num_nodes = graph.num_vertices()
        self.num_edges = graph.num_edges()
        # Sampling evaluation
        self.blocks_retained = 0.0
        self.graph_edge_ratio = 0.0
        self.difference_from_ideal_sample = 0.0
        self.expansion_quality = 0.0
        self.sampled_graph_clustering_coefficient = 0.0
        self.full_graph_clustering_coefficient = 0.0
        self.sampled_graph_diameter = 0
        self.full_graph_diameter = 0
        self.sampled_graph_largest_component = 0
        self.full_graph_largest_component = 0
        self.sampled_graph_island_vertices = 0
        self.sampled_graph_num_vertices = 0
        self.sampled_graph_num_edges = 0
        self.sampled_graph_edge_ratio = 0.0
        self.sampled_graph_num_blocks_algorithm = 0
        self.sampled_graph_num_blocks_truth = 0
        self.sampled_graph_accuracy = 0.0
        self.sampled_graph_rand_index = 0.0
        self.sampled_graph_adjusted_rand_index = 0.0
        self.sampled_graph_pairwise_recall = 0.0
        self.sampled_graph_pairwise_precision = 0.0
        self.sampled_graph_entropy_algorithm = 0.0
        self.sampled_graph_entropy_truth = 0.0
        self.sampled_graph_entropy_algorithm_given_truth = 0.0
        self.sampled_graph_entropy_truth_given_algorithm = 0.0
        self.sampled_graph_mutual_info = 0.0
        self.sampled_graph_missed_info = 0.0
        self.sampled_graph_erroneous_info = 0.0
        # Algorithm parameters
        self.num_block_proposals = args.blockProposals
        self.beta = args.beta
        self.sample_size = args.sample_size
        self.sampling_iterations = args.sample_iterations
        self.sampling_algorithm = args.sample_type
        self.delta_entropy_threshold = args.threshold
        self.nodal_update_threshold_strategy = args.nodal_update_strategy
        self.nodal_update_threshold_factor = args.factor
        self.nodal_update_threshold_direction = args.direction
        # Goodness of partition measures
        self.num_blocks_algorithm = 0
        self.num_blocks_truth = 0
        self.accuracy = 0.0
        self.rand_index = 0.0
        self.adjusted_rand_index = 0.0
        self.pairwise_recall = 0.0
        self.pairwise_precision = 0.0
        self.entropy_algorithm = 0.0
        self.entropy_truth = 0.0
        self.entropy_algorithm_given_truth = 0.0
        self.entropy_truth_given_algorithm = 0.0
        self.mutual_info = 0.0
        self.missed_info = 0.0
        self.erroneous_info = 0.0
        self.sampled_graph_description_length = 0.0
        self.max_sampled_graph_description_length = 0.0
        self.full_graph_description_length = 0.0
        self.max_full_graph_description_length = 0.0
        self.sampled_graph_modularity = 0.0
        self.full_graph_modularity = 0.0
        # Algorithm runtime measures
        self.loading = 0.0
        self.sampling = 0.0
        self.sampled_graph_partition_time = 0.0
        self.total_partition_time = 0.0
        self.merge_sample = 0.0
        self.propagate_membership = 0.0
        self.finetune_membership = 0.0
        self.prepare_next_partitions = list()  # type: List[float]
        # self.finetuning_details = None
        # Community details
        self.real_communities = dict()  # type: Dict[int, int]
        self.algorithm_communities = dict()  # type: Dict[int, int]
        self.sampled_graph_real_communities = dict()  # type: Dict[int, int]
        self.sampled_graph_algorithm_communities = dict()  # type: Dict[int, int]
        self.contingency_table = None  # type: np.ndarray
        self.sampled_graph_contingency_table = None  # type: np.ndarray
    # End of __init__()

    def evaluate_sampling(self, full_graph: Graph, sampled_graph: Graph, full_partition: BlockState,
                          sampled_graph_partition: BlockState, block_mapping: Dict[int, int],
                          vertex_mapping: Dict[int, int], assignment: np.ndarray):
        """Evaluates the goodness of the samples.

        Parameters
        ----------
        full_graph : Graph
            the full, unsampled Graph object
        sampled_graph : Graph
            the sampled graph
        full_partition : Partition
            the partitioning results on the full graph
        sampled_graph_partition : Partition
            the partitioning results on the sampled graph
        block_mapping : Dict[int, int]
            the mapping of blocks from the full graph to the sampled graph
        vertex_mapping : Dict[int, int]
            the mapping of vertices from the full graph to the sampled graph
        assignment : np.ndarray[int]
            the true vertex-to-community mapping
        """
        #####
        # General
        #####
        self.sampled_graph_num_vertices = sampled_graph.num_vertices()
        self.sampled_graph_num_edges = sampled_graph.num_edges()
        self.blocks_retained = sampled_graph_partition.get_B() / full_partition.get_B()
        # pseudo_diameter returns a tuple: (diameter, (start_vertex, end_vertex))
        self.sampled_graph_diameter = pseudo_diameter(sampled_graph)[0]
        self.full_graph_diameter = pseudo_diameter(full_graph)[0]
        for vertex in sampled_graph.vertices():
            if (vertex.in_degree() + vertex.out_degree()) == 0:
                self.sampled_graph_island_vertices += 1
        self.sampled_graph_largest_component = extract_largest_component(sampled_graph, directed=False).num_vertices()
        self.full_graph_largest_component = extract_largest_component(full_graph, directed=False).num_vertices()

        ######
        # Expansion quality (http://portal.acm.org/citation.cfm?doid=1772690.1772762)
        ######
        # Expansion factor = Neighbors of sample / size of sample
        # Maximum expansion factor = (size of graph - size of sample) / size of sample
        # Expansion quality = Neighbors of sample / (size of graph - size of sample)
        # Expansion quality = 1 means sample is at most 1 edge away from entire graph
        sampled_graph_vertices = set(vertex_mapping.keys())
        neighbors = set()
        for vertex in sampled_graph_vertices:
            for neighbor in full_graph.get_out_neighbors(vertex):
                neighbors.add(neighbor)
        neighbors = neighbors - sampled_graph_vertices
        self.expansion_quality = len(neighbors) / (full_graph.num_vertices() - sampled_graph.num_vertices())

        ######
        # Clustering coefficient
        ######
        self.sampled_graph_clustering_coefficient = global_clustering(sampled_graph)[0]
        self.full_graph_clustering_coefficient = global_clustering(full_graph)[0]

        ######
        # Info on communities
        ######
        self.get_community_details(assignment, full_partition.get_blocks().get_array(),
                                   sampled_graph_partition.get_blocks().get_array(), vertex_mapping)

        if np.unique(assignment).size == 1:  # Cannot compute below metrics if no true partition is provided
            return

        #####
        # % difference in ratio of within-block to between-block edges
        #####
        sample_assignment = assignment[np.fromiter(vertex_mapping.keys(), dtype=np.int32)]
        true_sampled_graph_partition = partition_from_truth(sampled_graph, sample_assignment)
        sampled_graph_blockmatrix = true_sampled_graph_partition.get_matrix()
        self.sampled_graph_edge_ratio = sampled_graph_blockmatrix.diagonal().sum() / sampled_graph_blockmatrix.sum()
        true_full_partition = partition_from_truth(full_graph, assignment)
        full_blockmatrix = true_full_partition.get_matrix()
        self.graph_edge_ratio = full_blockmatrix.diagonal().sum() / full_blockmatrix.sum()

        #####
        # Normalized difference from ideal-block membership
        #####
        membership_size = max(np.max(assignment), np.max(sample_assignment)) + 1
        full_graph_membership_nums = np.zeros(membership_size)
        for block_membership in assignment:
            full_graph_membership_nums[block_membership] += 1
        sampled_graph_membership_nums = np.zeros(membership_size)
        for block_membership in sample_assignment:
            sampled_graph_membership_nums[block_membership] += 1
        ideal_block_membership_nums = full_graph_membership_nums * \
            (sampled_graph.num_vertices() / full_graph.num_vertices())
        difference_from_ideal_block_membership_nums = np.abs(
            ideal_block_membership_nums - sampled_graph_membership_nums)
        self.difference_from_ideal_sample = np.sum(
            difference_from_ideal_block_membership_nums / sampled_graph.num_vertices())
    # End of evaluate_sampling()

    def get_community_details(self, assignment: np.ndarray, algorithm_assignment: np.ndarray,
                              sampled_graph_assignment: np.ndarray, vertex_mapping: Dict[int, int]):
        """Saves information on community details.

        Parameters
        ----------
        assignment : np.ndarray[int]
            the true vertex-to-community assignment for the entire graph
        algorithm_assignment : np.ndarray[int]
            the vertex-to-community assignment generated by the algorithm for the entire graph
        sampled_graph_assignment : np.ndarray[int]
            the vertex-to-community assignment generated by the algorithm for the sampled graph
        vertex_mapping : Dict[int, int]
            the mapping of full graph to sampled vertices
        """
        for community in assignment:
            if community in self.real_communities:
                self.real_communities[community] += 1
            else:
                self.real_communities[community] = 1

        for community in algorithm_assignment:
            if community in self.algorithm_communities:
                self.algorithm_communities[community] += 1
            else:
                self.algorithm_communities[community] = 1

        sampled_graph_real_assignment = assignment[np.fromiter(vertex_mapping.keys(), dtype=np.int32)]
        for community in sampled_graph_real_assignment:
            if community in self.sampled_graph_real_communities:
                self.sampled_graph_real_communities[community] += 1
            else:
                self.sampled_graph_real_communities[community] = 1

        for community in sampled_graph_assignment:
            if community in self.sampled_graph_algorithm_communities:
                self.sampled_graph_algorithm_communities[community] += 1
            else:
                self.sampled_graph_algorithm_communities[community] = 1
    # End of get_community_details()

    def total_runtime(self, start_t: float, end_t: float):
        """Finalizes the runtime of the algorithm.

            Parameters
            ---------
            start_t : float
                the start time of the partitioning
            end_t : float
                the end time of the partitioning
        """
        runtime = end_t - start_t
        self.total_partition_time = runtime
    # End of total_runtime()

    def clustering_coefficient(self, graph: Graph) -> float:
        """Calculates the clustering coefficient of a given graph.

        Clustering coefficient = number of closed triangles / total possible number of triangles.

        Current version also counts self-connections as triangles as well.

            Parameters
            ---------
            graph : Graph
                the graph whose clustering coefficient is of interest

            Returns
            -------
            clustering_coefficient : float
                the clustering coefficient of said graph
        """
        n_triangles_sample = 0
        for vertex in range(graph.num_vertices()):
            for neighbor in graph.get_out_neighbors(vertex):
                for neighbor2 in graph.get_out_neighbors(vertex):
                    # TODO: If not counting self-links, add check for that here
                    if neighbor2 in graph.get_out_neighbors(neighbor):
                        n_triangles_sample += 1
        return n_triangles_sample / (graph.num_vertices() * (graph.num_vertices() - 1))
    # End of clustering_coefficient()

    def save(self):
        """Saves the evaluation to a CSV file. Creates a new CSV file one the path of csv_file doesn't exist. Appends
        results to the CSV file if it does.
        """
        write_header = False
        if not os.path.isfile(self.csv_file):
            directory = os.path.dirname(self.csv_file)
            if directory not in [".", ""]:
                os.makedirs(directory, exist_ok=True)
            write_header = True
        with open(self.csv_file, "a") as csv_file:
            writer = csv.writer(csv_file)
            if write_header:
                writer.writerow(Evaluation.FIELD_NAMES)
            writer.writerow([
                self.block_size_variation,
                self.block_overlap,
                self.streaming_type,
                self.num_nodes,
                self.num_edges,
                self.blocks_retained,
                self.graph_edge_ratio,
                self.difference_from_ideal_sample,
                self.expansion_quality,
                self.sampled_graph_clustering_coefficient,
                self.full_graph_clustering_coefficient,
                self.sampled_graph_diameter,
                self.full_graph_diameter,
                self.sampled_graph_largest_component,
                self.full_graph_largest_component,
                self.sampled_graph_island_vertices,
                self.sampled_graph_num_vertices,
                self.sampled_graph_num_edges,
                self.sampled_graph_edge_ratio,
                self.sampled_graph_num_blocks_algorithm,
                self.sampled_graph_num_blocks_truth,
                self.sampled_graph_accuracy,
                self.sampled_graph_rand_index,
                self.sampled_graph_adjusted_rand_index,
                self.sampled_graph_pairwise_recall,
                self.sampled_graph_pairwise_precision,
                self.sampled_graph_entropy_algorithm,
                self.sampled_graph_entropy_truth,
                self.sampled_graph_entropy_algorithm_given_truth,
                self.sampled_graph_entropy_truth_given_algorithm,
                self.sampled_graph_mutual_info,
                self.sampled_graph_missed_info,
                self.sampled_graph_erroneous_info,
                self.num_block_proposals,
                self.beta,
                self.sample_size,
                self.sampling_iterations,
                self.sampling_algorithm,
                self.delta_entropy_threshold,
                self.nodal_update_threshold_strategy,
                self.nodal_update_threshold_factor,
                self.nodal_update_threshold_direction,
                self.num_blocks_algorithm,
                self.num_blocks_truth,
                self.accuracy,
                self.rand_index,
                self.adjusted_rand_index,
                self.pairwise_recall,
                self.pairwise_precision,
                self.entropy_algorithm,
                self.entropy_truth,
                self.entropy_algorithm_given_truth,
                self.entropy_truth_given_algorithm,
                self.mutual_info,
                self.missed_info,
                self.erroneous_info,
                self.sampled_graph_description_length,
                self.max_sampled_graph_description_length,
                self.full_graph_description_length,
                self.max_full_graph_description_length,
                self.sampled_graph_modularity,
                self.full_graph_modularity,
                self.loading,
                self.sampling,
                self.sampled_graph_partition_time,
                self.total_partition_time,
                self.merge_sample,
                self.propagate_membership,
                self.finetune_membership,
                self.args.tag
            ])
        self._save_details()
    # End of save()

    def _save_details(self):
        """Saves the details of the MCMC and Block Merge timings.
        """
        write_header = False
        if not os.path.isfile(self.csv_details_file):
            directory = os.path.dirname(self.csv_details_file)
            if directory not in [".", ""]:
                os.makedirs(directory, exist_ok=True)
            write_header = True
        with open(self.csv_details_file, "a") as details_file:
            writer = csv.writer(details_file)
            if write_header:
                writer.writerow(Evaluation.DETAILS_FIELD_NAMES)
            for community, size in self.real_communities.items():
                writer.writerow([self.args.numNodes, self.args.overlap, self.args.blockSizeVar, community, False,
                                 '-', size, False, self.sample_size, self.sampling_algorithm, self.sampling_iterations,
                                 self.args.tag])
            for community, size in self.algorithm_communities.items():
                writer.writerow([self.args.numNodes, self.args.overlap, self.args.blockSizeVar, community, False,
                                 '-', size, True, self.sample_size, self.sampling_algorithm, self.sampling_iterations,
                                 self.args.tag])
            if self.contingency_table is not None:
                for row in range(np.shape(self.contingency_table)[0]):
                    for col in range(np.shape(self.contingency_table)[1]):
                        writer.writerow([self.args.numNodes, self.args.overlap, self.args.blockSizeVar, row, False,
                                        col, self.contingency_table[row, col], False, self.sample_size,
                                        self.sampling_algorithm, self.sampling_iterations, self.args.tag])
            if (self.sampling_algorithm == "none") or (self.args.sample_size == 0):
                return
            for community, size in self.sampled_graph_real_communities.items():
                writer.writerow([self.args.numNodes, self.args.overlap, self.args.blockSizeVar, community, True,
                                 '-', size, False, self.sample_size, self.sampling_algorithm, self.sampling_iterations,
                                 self.args.tag])
            for community, size in self.sampled_graph_algorithm_communities.items():
                writer.writerow([self.args.numNodes, self.args.overlap, self.args.blockSizeVar, community, True,
                                 '-', size, True, self.sample_size, self.sampling_algorithm, self.sampling_iterations,
                                 self.args.tag])
            if self.sampled_graph_contingency_table is not None:
                for row in range(np.shape(self.sampled_graph_contingency_table)[0]):
                    for col in range(np.shape(self.sampled_graph_contingency_table)[1]):
                        writer.writerow([self.args.numNodes, self.args.overlap, self.args.blockSizeVar, row, False,
                                        col, self.sampled_graph_contingency_table[row, col], False, self.sample_size,
                                        self.sampling_algorithm, self.sampling_iterations, self.args.tag])
    # End of _save_details()
# End of Evaluation()
