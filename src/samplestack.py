"""A Stack object for incremental sampling
"""
import argparse
import timeit
from typing import Dict, List, Tuple

from graph_tool import Graph
from graph_tool.inference import BlockState
from graph_tool.inference import minimize_blockmodel_dl

from evaluation import Evaluation
from sample import Sample
from samplestate import SampleState
from util import finetune_assignment
from util import load_graph
from util import partition_from_sample


class SampleStack(object):
    def __init__(self, args: argparse.Namespace) -> None:
        """Creates a SampleStack object.

        Parameters
        ---------
        args : argparse.Namespace
            the command-line arguments provided by the user
        """
        # Load graph
        # Create stack of samples
        # Use List as stack
        self.t_load_start = timeit.default_timer()
        self.full_graph, self.true_block_assignment = load_graph(args)
        self.t_load_end = timeit.default_timer()
        self.stack = list()  # type: List[Tuple[Graph, Sample]]
        self.create_sample_stack(args)
        self.t_sample_end = timeit.default_timer()
    # End of __init__()

    def create_sample_stack(self, args: argparse.Namespace):
        """Iteratively performs sampling to create the stack of samples.

        Parameters
        ----------
        args : argparse.Namespace
            the command-line arguments provided by the user
        """
        # Iteratively perform sampling
        for iteration in range(args.sample_iterations):
            if iteration == 0:
                sampled_graph, sample = self.sample(self.full_graph, args)
            else:
                sampled_graph, sample = self.sample(self.full_graph, args, sample.state)
            self.stack.append((sampled_graph, sample))
    # End of create_sample_stack()

    def sample(self, graph: Graph, args: argparse.Namespace, prev_state: SampleState = None) -> Tuple[Graph, Sample]:
        """Sample a set of vertices from the graph.

        Parameters
        ----------
        full_graph : Graph
            the graph from which to sample vertices
        args : Namespace
            the parsed command-line arguments
        prev_state : SampleState
            if prev_state is not None, sample will be conditioned on the previously selected vertices

        Returns
        ------
        sampled_graph : Graph
            the sampled graph created from the sampled Graph vertices
        sample : Sample
            the sample object containing the vertex and block mappings
        """
        sample_size = int((self.full_graph.num_vertices() * (args.sample_size / 100)) / args.sample_iterations)
        if prev_state is None:
            prev_state = SampleState(sample_size)
        sample_object = Sample.create_sample(self.full_graph, self.true_block_assignment, args, prev_state)
        return sample_object.graph, sample_object
    # End of sample()

    def _push(self):
        # Add a subsample to the stack
        raise NotImplementedError()
    # End of _push()

    def _pop(self) -> Tuple[Graph, Sample]:
        # Propagate a subsample's results up the stack
        return self.stack.pop(0)
    # End of _pop()

    def unstack(self, args: argparse.Namespace, sampled_graph_partition: BlockState = None,
                evaluation: Evaluation = None) -> Tuple[Graph, BlockState, Dict, Dict, Evaluation]:
        """Performs SBP on the first (innermost) sample. Merges said sample with the next in the stack, and performs
        SBP on the combined results. Repeats the process until all samples have been partitioned.

        Paramters
        ---------
        args : argparse.Namespace
            the command-line arguments supplied by the user
        sampled_graph_partition : BlockState
            the current partitioned state of the sampled graph. Default = None
        evaluation : Evaluation
            the current state of the evaluation of the algorithm. Default = None

        Returns
        -------
        sampled_graph : Graph
            the Graph object describing the combined samples
        sampled_graph_partition : BlockState
            the partition results of the combined samples
        vertex_mapping : Dict[int, int]
            the mapping of the vertices from the combined sample to the full graph
        block_mapping : Dict[int, int]
            the mapping of the communities/blocks from the combined sample to the full graph
        """
        # Propagate results back through the stack
        sampled_graph, sample = self._pop()
        min_num_blocks = -1
        # denominator = 2
        # if args.sample_iterations > 1:
        #     min_num_blocks = int(sampled_graph.num_nodes / denominator)
        #     min_num_blocks = 0
        if evaluation is None:
            evaluation = Evaluation(args, sampled_graph)
        print("Subgraph: V = {} E = {}".format(sampled_graph.num_vertices(), sampled_graph.num_edges()))
        t0 = timeit.default_timer()
        combined_partition = minimize_blockmodel_dl(sampled_graph,
                                                    shrink_args={'parallel': True}, verbose=args.verbose,
                                                    mcmc_equilibrate_args={'verbose': args.verbose, 'epsilon': 1e-4})
        evaluation.sampled_graph_partition_time += (timeit.default_timer() - t0)
        combined_sampled_graph = sampled_graph
        while len(self.stack) > 0:
            sampled_graph, next_sample = self._pop()
            t0 = timeit.default_timer()
            sample_partition = minimize_blockmodel_dl(sampled_graph,
                                                      shrink_args={'parallel': True}, verbose=args.verbose,
                                                      mcmc_equilibrate_args={'verbose': args.verbose, 'epsilon': 1e-4})
            evaluation.sampled_graph_partition_time += (timeit.default_timer() - t0)
            t1 = timeit.default_timer()
            # TODO: fix this to allow multi-sample strategies
            combined_partition, combined_sampled_graph, sample = self.combine_partition_with_sample(
                combined_partition, sample_partition, sample, next_sample, args
            )
            t2 = timeit.default_timer()
            # TODO: change to evaluation.merge_sample time?
            evaluation.propagate_membership += (t2 - t1)
        print("=====Performing final (combined) sample partitioning=====")
        if min_num_blocks > 0 or (args.sample_iterations > 1):
            combined_partition.num_blocks_to_merge = 0
            sampled_graph_partition = minimize_blockmodel_dl(combined_sampled_graph,
                                                             shrink_args={'parallel': True}, verbose=args.verbose,
                                                             mcmc_equilibrate_args={'verbose': False, 'epsilon': 1e-4})
        else:
            sampled_graph_partition = combined_partition
        return (
            combined_sampled_graph, sampled_graph_partition, sample.vertex_mapping, sample.true_blocks_mapping,
            evaluation
        )
    # End of unstack()

    def extrapolate_sample_partition(self, sampled_graph_partition: BlockState, vertex_mapping: Dict[int, int],
                                     args: argparse.Namespace,
                                     evaluation: Evaluation) -> Tuple[Graph, BlockState, Evaluation]:
        """Extrapolates the partitioning results from the sample to the full graph.

        This is done by first assigning to every unsampled vertex, the community to which it's most strongly
        connected. Then, a fine-tuning step (MCMC updates using a modified Metropolis-Hasting algorithm) is run
        on the result.

        Parameters
        ----------
        sampled_graph_partition : BlockState
            the current partitioned state of the sampled graph
        vertex_mapping : Dict[int, int]
            the mapping of sample vertices to full vertices
        args : argparse.Namespace
            the command-line arguments supplied by the user
        evaluation : Evaluation
            the current state of the evaluation of the algorithm

        Returns
        -------
        full_graph : Graph
            the graph object representing the entire (unsampled) graph
        full_graph_partition : BlockState
            the partition state of the full graph after extrapolation and fine-tuning
        evaluation : Evaluation
            the evaluation results of the algorithm
        """
        t1 = timeit.default_timer()
        full_graph_partition = partition_from_sample(sampled_graph_partition, self.full_graph, vertex_mapping)
        t2 = timeit.default_timer()
        full_graph_partition = finetune_assignment(full_graph_partition, args)
        t3 = timeit.default_timer()
        evaluation.loading = self.t_load_end - self.t_load_start
        evaluation.sampling = self.t_sample_end - self.t_load_end
        evaluation.propagate_membership += (t2 - t1)
        evaluation.finetune_membership += (t3 - t2)
        return self.full_graph, full_graph_partition, evaluation
    # End of extrapolate_sample_partition()

    def tail(self) -> Tuple[Graph, Sample]:
        # Get innermost sample
        return self.stack[0]
    # End of tail()
# End of SampleStack()
