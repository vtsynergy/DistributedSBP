""" This Python script runs the graph-tool graph partition algorithm by Tiago Peixoto."""

import timeit
import argparse

import numpy as np
from graph_tool.inference import minimize_blockmodel_dl

from evaluate import evaluate_partition, evaluate_sampled_graph_partition
from evaluation import Evaluation
from samplestack import SampleStack
from util import load_graph


def parse_arguments():
    """Parses command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        the parsed command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parts", type=int, default=0,
                        help="""The number of streaming partitions to the dataset. If the dataset is static, 0.
                             Default = 0""")
    parser.add_argument("-o", "--overlap", type=str, default="low", help="(low|high). Default = low")
    parser.add_argument("-s", "--blockSizeVar", type=str, default="low", help="(low|high). Default = low")
    parser.add_argument("-t", "--type", type=str, default="static",
                        help="(static|streamingEdge|streamingSnowball). Default = static")
    parser.add_argument("-n", "--numNodes", type=int, default=1000, help="The size of the dataset. Default = 1000")
    parser.add_argument("-d", "--directory", type=str, default="../../data",
                        help="The location of the dataset directory. Default = ../../data")
    parser.add_argument("-v", "--verbose", action="store_true", help="If supplied, will print 'helpful' messages.")
    parser.add_argument("-b", "--blockProposals", type=int, default=10,
                        help="The number of block merge proposals per block. Default = 10")
    parser.add_argument("-i", "--iterations", type=int, default=100,
                        help="Maximum number of node reassignment iterations. Default = 100")
    parser.add_argument("-r", "--blockReductionRate", type=float, default=0.5,
                        help="The block reduction rate. Default = 0.5")
    parser.add_argument("--beta", type=int, default=3,
                        help="exploitation vs exploration: higher threshold = higher exploration. Default = 3")
    parser.add_argument("-c", "--csv", type=str, default="eval/benchmark",
                        help="The filepath to the csv file in which to store the evaluation results.")
    # Nodal Update Strategy
    parser.add_argument("-u", "--nodal_update_strategy", type=str, default="original",
                        help="(original|step|exponential|log). Default = original")
    parser.add_argument("--direction", type=str, default="growth", help="(growth|decay) Default = growth")
    parser.add_argument("-f", "--factor", type=float, default=0.0001,
                        help="""The factor by which to grow or decay the nodal update threshold.
                            If the nodal update strategy is step:
                                this value is added to or subtracted from the threshold with every iteration
                            If the nodal update strategy is exponential:
                                this (1 +/- this value) is multiplied by the threshold with every iteration
                            If the nodal update strategy is log:
                                this value is ignored
                            Default = 0.0001""")
    parser.add_argument("-e", "--threshold", type=float, default=5e-4,
                        help="The threshold at which to stop nodal block reassignment. Default = 5e-4")
    parser.add_argument("-z", "--sample_size", type=int, default=100,
                        help="The percent of total nodes to use as sample. Default = 100 (no sampling)")
    parser.add_argument("-m", "--sample_type", type=str, default="none",
                        choices=["degree_weighted", "expansion_snowball", "forest_fire", "max_degree", "random_jump",
                                 "random_node_neighbor", "random_walk", "uniform_random", "none"],
                        help="""Sampling algorithm to use. Default = none""")
    parser.add_argument("--sample_iterations", type=int, default=1,
                        help="The number of sampling iterations to perform. Default = 1")
    parser.add_argument("--degrees", action="store_true", help="Save vertex degrees and exit.")
    parser.add_argument("--tag", type=str, default="none", help="A custom tag for identifying particular runs")
    parser.add_argument("--delimiter", type=str, default="\t", help="Delimiter for reading in graph")
    parser.add_argument("--gtload", action="store_true",
                        help="""If true, will load the graph using graph tool's load graph function""")
    parser.add_argument("--undirected", action="store_true", help="If true, graph is symmetrical")
    args = parser.parse_args()
    return args
# End of parse_arguments()


if __name__ == "__main__":
    args = parse_arguments()
    t_start = timeit.default_timer()

    if args.sample_type != "none":
        samplestack = SampleStack(args)
        sampled_graph, sampled_graph_partition, vertex_mapping, block_mapping, evaluation = samplestack.unstack(args)
        full_graph, full_graph_partition, evaluation = samplestack.extrapolate_sample_partition(
            sampled_graph_partition, vertex_mapping, args, evaluation
        )
    else:
        graph, true_block_assignment = load_graph(args)
        t_load = timeit.default_timer()
        t_sample = timeit.default_timer()
        print("Performing stochastic block partitioning")
        evaluation = Evaluation(args, graph)
        # Please refer to the graph-tool documentation under graph-tool.inference for details on the input parameters
        partition = minimize_blockmodel_dl(graph,
                                           shrink_args={'parallel': True}, verbose=args.verbose,
                                           mcmc_equilibrate_args={'verbose': args.verbose, 'epsilon': 1e-4})
        t_partition = timeit.default_timer()

    t_end = timeit.default_timer()
    print('\nGraph partition took {} seconds'.format(t_end - t_start))
    evaluation.total_runtime(t_start, t_end)

    if args.sample_type != "none":
        print("===== Evaluating graph sampling =====")
        evaluation.evaluate_sampling(full_graph, sampled_graph, full_graph_partition, sampled_graph_partition,
                                     block_mapping, vertex_mapping, samplestack.true_block_assignment)
        evaluation.num_nodes = full_graph.num_vertices()
        evaluation.num_edges = full_graph.num_edges()
        # evaluate output partition against the true partition
        print("===== Evaluating sampled graph partition =====")
        evaluate_sampled_graph_partition(
            sampled_graph, samplestack.true_block_assignment[np.fromiter(vertex_mapping.keys(), dtype=np.int32)],
            sampled_graph_partition, evaluation, block_mapping)
        print("===== Evaluating full graph partition =====")
        evaluate_partition(full_graph, samplestack.true_block_assignment, full_graph_partition, evaluation)
    else:
        evaluation.loading = t_load - t_start
        evaluation.sampled_graph_partition_time = t_partition - t_sample
        evaluate_partition(graph, true_block_assignment, partition, evaluation)
