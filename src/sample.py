"""Helper functions for performing different kinds of sampling.
"""

import argparse
from typing import List, Set
from copy import copy

from graph_tool import Graph
import numpy as np

from samplestate import DegreeWeightedSampleState
from samplestate import ExpansionSnowballSampleState
from samplestate import ForestFireSampleState
from samplestate import MaxDegreeSampleState
from samplestate import RandomWalkSampleState
from samplestate import RandomJumpSampleState
from samplestate import RandomNodeNeighborSampleState
from samplestate import SampleState
from samplestate import UniformRandomSampleState


class Sample():
    """Stores the variables needed to create a subgraph.
    """

    def __init__(self, state: SampleState, graph: Graph, old_true_block_assignment: np.ndarray) -> None:
        """Creates a new Sample object. Contains information about the sampled vertices and edges, the mapping of
        sampled vertices to the original graph vertices, and the true block membership for the sampled vertices.

        Parameters
        ----------
        state : SampleState
            contains the sampled vertices
        graph : Graph
            the graph from which the sample is taken
        old_true_block_assignment : np.ndarray[int]
            the vertex-to-community assignment array. Currently assumes that community assignment is non-overlapping.
        """
        self.state = state
        sampled_vertices = sorted(state.sample_idx[-state.sample_size:])
        self.vertex_mapping = dict([(v, k) for k, v in enumerate(sampled_vertices)])
        binary_filter = np.zeros(graph.num_vertices())
        binary_filter[sampled_vertices] = 1
        graph.set_vertex_filter(graph.new_vertex_property("bool", binary_filter))
        self.graph = Graph(graph, prune=True)  # If ordering is wacky, may need to play around with vorder
        graph.clear_filters()
        true_block_assignment = old_true_block_assignment[sampled_vertices]
        # Assuming the sample doesn't capture all the blocks, the block numbers in the sample may not be consecutive
        # The true_blocks_mapping ensures that they are consecutive
        true_blocks = list(set(true_block_assignment))
        self.true_blocks_mapping = dict([(v, k) for k, v in enumerate(true_blocks)])
        self.true_block_assignment = np.asarray([self.true_blocks_mapping[b] for b in true_block_assignment])
        self.sample_num = len(self.vertex_mapping)
    # End of __init__()

    @staticmethod
    def create_sample(graph: Graph, old_true_block_assignment: np.ndarray, args: argparse.Namespace,
                      prev_state: SampleState) -> 'Sample':
        """Performs sampling according to the sample type in args.

        TODO: either re-write how this method is used, or get rid of it - it seems to be a code smell.
        """
        # get rid of 1-degree vertices
        degrees = graph.get_total_degrees(np.arange(graph.num_vertices()))
        degree_filter = degrees > 2
        mapping = np.where(degrees > 2)[0]
        graph.set_vertex_filter(graph.new_vertex_property("bool", degree_filter))
        filtered_graph = Graph(graph, prune=True)
        print(filtered_graph.num_vertices())
        graph.clear_filters()
        # TODO: keep track of the mapping to original graph
        # TODO: below methods can return a SampleState, which we map back to original vertices here, then create the 
        # sample before return. This is brilliant! I am genius!
        if args.sample_type == "degree_weighted":
            state = Sample.degree_weighted_sample(filtered_graph, graph.num_vertices(), prev_state, args)
        elif args.sample_type == "expansion_snowball":
            state = Sample.expansion_snowball_sample(filtered_graph, graph.num_vertices(), prev_state, args)
        elif args.sample_type == "forest_fire":
            state = Sample.forest_fire_sample(filtered_graph, graph.num_vertices(), prev_state, args)
        elif args.sample_type == "max_degree":
            state = Sample.max_degree_sample(filtered_graph, graph.num_vertices(), prev_state, args)
        elif args.sample_type == "random_jump":
            state = Sample.random_jump_sample(filtered_graph, graph.num_vertices(), prev_state, args)
        elif args.sample_type == "random_node_neighbor":
            state = Sample.random_node_neighbor_sample(filtered_graph, graph.num_vertices(), prev_state, args)
        elif args.sample_type == "random_walk":
            state = Sample.random_walk_sample(filtered_graph, graph.num_vertices(), prev_state, args)
        elif args.sample_type == "uniform_random":
            state = Sample.uniform_random_sample(filtered_graph, graph.num_vertices(), prev_state, args)
        else:
            raise NotImplementedError("Sample type: {} is not implemented!".format(args.sample_type))
        state.sample_idx = mapping[state.sample_idx]
        return Sample(state, graph, old_true_block_assignment)
    # End of create_sample()

    @staticmethod
    def uniform_random_sample(graph: Graph, num_vertices: int, prev_state: UniformRandomSampleState,
                              args: argparse.Namespace) -> SampleState:
        """Uniform random sampling. All vertices are selected with the same probability.

        Parameters
        ----------
        graph : Graph
            the filtered graph from which to sample vertices
        num_vertices : int
            number of vertices in the unfiltered graph
        prev_state : UniformRandomSampleState
            the state of the previous sample in the stack. If there is no previous sample, an empty SampleState object
            should be passed in here.
        args : argparse.Namespace
            the command-line arguments provided by the user

        Returns
        -------
        state : SampleState
            the sample state with the sampled vertex ids (Note: these ids correspond to the filtered graph, and have
            to be mapped back to the unfiltered graph)
        """
        state = UniformRandomSampleState(graph.num_vertices(), prev_state)
        sample_num = int((num_vertices * (args.sample_size / 100)) / args.sample_iterations)
        choices = np.setdiff1d(np.asarray(range(graph.num_vertices())), state.sample_idx)
        state.sample_idx = np.concatenate(
            (state.sample_idx, np.random.choice(choices, sample_num, replace=False)),
            axis=None
        )
        return state
    # End of uniform_random_sampling()

    @staticmethod
    def random_walk_sample(graph: Graph, num_vertices: int, prev_state: RandomWalkSampleState,
                           args: argparse.Namespace) -> SampleState:
        """Random walk sampling. Start from a vertex and walk along the edges, sampling every vertex that is a part of
        the walk. With a probability of 0.15, restart the walk from the original vertex. To prevent getting stuck,
        after making N attempts, where N = the target number of vertices in the sample, change the starting vertex to a
        random vertex.

        Parameters
        ----------
        graph : Graph
            the filtered graph from which to sample vertices
        num_vertices : int
            number of vertices in the unfiltered graph
        prev_state : RandomWalkSampleState
            the state of the previous sample in the stack. If there is no previous sample, an empty SampleState object
            should be passed in here.
        args : argparse.Namespace
            the command-line arguments provided by the user

        Returns
        -------
        state : SampleState
            the sample state with the sampled vertex ids (Note: these ids correspond to the filtered graph, and have
            to be mapped back to the unfiltered graph)
        """
        state = RandomWalkSampleState(graph.num_vertices(), prev_state)
        sample_num = int((num_vertices * (args.sample_size / 100)) / args.sample_iterations)
        sample_num += len(state.sample_idx)
        num_tries = 0
        start = np.random.randint(sample_num)  # start with a random vertex
        vertex = start

        while len(state.index_set) == 0 or len(state.index_set) % sample_num != 0:
            num_tries += 1
            if not state.sampled_marker[vertex]:
                state.index_set.append(vertex)
                state.sampled_marker[vertex] = True
            if num_tries % sample_num == 0:  # If the number of tries is large, restart from new random vertex
                start = np.random.randint(sample_num)
                vertex = start
                num_tries = 0
            elif np.random.random() < 0.15:  # With a probability of 0.15, restart at original node
                vertex = start
            elif len(graph.get_out_neighbors(vertex)) > 0:  # If the vertex has out neighbors, go to one of them
                vertex = np.random.choice(graph.get_out_neighbors(vertex))
            else:  # Otherwise, restart from the original vertex
                if len(graph.get_out_neighbors(start)) == 0:  # if original vertex has no out neighbors, change it
                    start = np.random.randint(sample_num)
                vertex = start

        state.sample_idx = np.asarray(state.index_set)
        return state
    # End of random_walk_sample()

    @staticmethod
    def random_jump_sample(graph: Graph, num_vertices: int, prev_state: RandomJumpSampleState,
                           args: argparse.Namespace) -> SampleState:
        """Random jump sampling. Start from a vertex and walk along the edges, sampling every vertex that is a part of
        the walk. With a probability of 0.15, restart the walk from a new vertex.

        Parameters
        ----------
        graph : Graph
            the filtered graph from which to sample vertices
        num_vertices : int
            number of vertices in the unfiltered graph
        prev_state : RandomWalkSampleState
            the state of the previous sample in the stack. If there is no previous sample, an empty SampleState object
            should be passed in here.
        args : argparse.Namespace
            the command-line arguments provided by the user

        Returns
        -------
        state : SampleState
            the sample state with the sampled vertex ids (Note: these ids correspond to the filtered graph, and have
            to be mapped back to the unfiltered graph)
        """
        state = RandomJumpSampleState(graph.num_vertices(), prev_state)
        sample_num = int((num_vertices * (args.sample_size / 100)) / args.sample_iterations)
        sample_num += len(state.sample_idx)
        num_tries = 0
        start = np.random.randint(sample_num)  # start with a random vertex
        vertex = start

        while len(state.index_set) == 0 or len(state.index_set) % sample_num != 0:
            num_tries += 1
            if not state.sampled_marker[vertex]:
                state.index_set.append(vertex)
                state.sampled_marker[vertex] = True
            # If the number of tries is large, or with a probability of 0.15, start from new random vertex
            if num_tries % sample_num == 0 or np.random.random() < 0.15:
                start = np.random.randint(sample_num)
                vertex = start
                num_tries = 0
            elif graph.vertex(vertex).out_degree() > 0:
                # len(graph.get_out_neighbors(vertex)) > 0:  # If the vertex has out neighbors, go to one of them
                vertex = np.random.choice(graph.get_out_neighbors(vertex))
            else:  # Otherwise, restart from the original vertex
                if graph.vertex(start).out_degree() == 0:
                    # len(graph.get_out_neighbors(start)) == 0:  # if original vertex has no out neighbors, change it
                    start = np.random.randint(sample_num)
                vertex = start

        state.sample_idx = np.asarray(state.index_set)
        return state
    # End of random_jump_sample()

    @staticmethod
    def degree_weighted_sample(graph: Graph, num_vertices: int, prev_state: DegreeWeightedSampleState,
                               args: argparse.Namespace) -> SampleState:
        """Degree-weighted sampling. The probability of selecting a vertex is proportional to its degree.

        Parameters
        ----------
        graph : Graph
            the filtered graph from which to sample vertices
        num_vertices : int
            number of vertices in the unfiltered graph
        prev_state : UniformRandomSampleState
            the state of the previous sample in the stack. If there is no previous sample, an empty SampleState object
            should be passed in here.
        args : argparse.Namespace
            the command-line arguments provided by the user

        Returns
        -------
        state : SampleState
            the sample state with the sampled vertex ids (Note: these ids correspond to the filtered graph, and have
            to be mapped back to the unfiltered graph)
        """
        state = DegreeWeightedSampleState(graph.num_vertices(), prev_state)
        sample_num = int((num_vertices * (args.sample_size / 100)) / args.sample_iterations)
        vertex_degrees = graph.get_total_degrees(np.arange(graph.num_vertices()))
        vertex_degrees[state.sample_idx] = 0
        state.sample_idx = np.concatenate(
            (state.sample_idx, np.random.choice(graph.num_vertices(), sample_num, replace=False,
                                                p=vertex_degrees / np.sum(vertex_degrees)))
        )
        return state
    # End of degree_weighted_sample()

    @staticmethod
    def random_node_neighbor_sample(graph: Graph, num_vertices: int, prev_state: RandomNodeNeighborSampleState,
                                    args: argparse.Namespace) -> SampleState:
        """Random node neighbor sampling. Whenever a single vertex is selected, all its out neighbors are selected
        as well.

        Parameters
        ----------
        graph : Graph
            the filtered graph from which to sample vertices
        num_vertices : int
            number of vertices in the unfiltered graph
        prev_state : UniformRandomSampleState
            the state of the previous sample in the stack. If there is no previous sample, an empty SampleState object
            should be passed in here.
        args : argparse.Namespace
            the command-line arguments provided by the user

        Returns
        -------
        state : SampleState
            the sample state with the sampled vertex ids (Note: these ids correspond to the filtered graph, and have
            to be mapped back to the unfiltered graph)
        """
        state = RandomNodeNeighborSampleState(graph.num_vertices(), prev_state)
        sample_num = int((num_vertices * (args.sample_size / 100)) / args.sample_iterations)
        choices = np.setdiff1d(np.asarray(range(graph.num_vertices())), state.sample_idx)
        random_samples = np.random.choice(choices, sample_num, replace=False)
        sample_num += len(state.sample_idx)
        for vertex in random_samples:
            if not state.sampled_marker[vertex]:
                state.index_set.append(vertex)
                state.sampled_marker[vertex] = True
            for neighbor in graph.get_out_neighbors(vertex):
                if not state.sampled_marker[neighbor]:
                    state.index_set.append(neighbor)
                    state.sampled_marker[neighbor] = True
            if len(state.index_set) >= sample_num:
                break
        state.sample_idx = np.asarray(state.index_set[:sample_num])
        return state
    # End of random_node_neighbor_sample()

    @staticmethod
    def forest_fire_sample(graph: Graph, num_vertices: int, prev_state: ForestFireSampleState,
                           args: argparse.Namespace) -> SampleState:
        """Forest-fire sampling with forward probability = 0.7. At every stage, select 70% of the neighbors of the
        current sample. Vertices that were not selected are 'blacklisted', and no longer viable for future selection.
        If all vertices are thus 'burnt' before the target number of vertices has been selected, restart sampling from
        a new starting vertex.

        Parameters
        ----------
        graph : Graph
            the filtered graph from which to sample vertices
        num_vertices : int
            number of vertices in the unfiltered graph
        prev_state : UniformRandomSampleState
            the state of the previous sample in the stack. If there is no previous sample, an empty SampleState object
            should be passed in here.
        args : argparse.Namespace
            the command-line arguments provided by the user

        Returns
        -------
        state : SampleState
            the sample state with the sampled vertex ids (Note: these ids correspond to the filtered graph, and have
            to be mapped back to the unfiltered graph)
        """
        state = ForestFireSampleState(graph.num_vertices(), prev_state)
        sample_num = int((num_vertices * (args.sample_size / 100)) / args.sample_iterations)
        sample_num += len(state.sample_idx)
        while len(state.index_set) == 0 or len(state.index_set) % sample_num != 0:
            for vertex in state.current_fire_front:
                # add vertex to index set
                if not state.sampled_marker[vertex]:
                    state.sampled_marker[vertex] = True
                    state.burnt_marker[vertex] = True
                    state.num_burnt += 1
                    state.index_set.append(vertex)
                # select edges to burn
                num_to_choose = np.random.geometric(0.7)
                out_neighbors = graph.get_out_neighbors(vertex)
                if len(out_neighbors) < 1:  # If there are no outgoing neighbors
                    continue
                if len(out_neighbors) <= num_to_choose:
                    num_to_choose = len(out_neighbors)
                mask = np.zeros(len(out_neighbors))
                indexes = np.random.choice(np.arange(len(out_neighbors)), num_to_choose, replace=False)
                mask[indexes] = 1
                for index, value in enumerate(mask):
                    neighbor = out_neighbors[index]
                    if value == 1:  # if chosen, add to next frontier
                        if not state.burnt_marker[neighbor]:
                            state.next_fire_front.append(neighbor)
                    state.burnt_marker[neighbor] = True  # mark all neighbors as visited
            if state.num_burnt == graph.num_vertices():  # all samples are burnt, restart
                state.num_burnt = 0
                state.burnt_marker = [False] * graph.num_vertices()
                state.current_fire_front = [np.random.randint(graph.num_vertices())]
                state.next_fire_front = list()
                continue
            if len(state.next_fire_front) == 0:  # if fire is burnt-out
                state.current_fire_front = [np.random.randint(graph.num_vertices())]
            else:
                state.current_fire_front = list(state.next_fire_front)
                state.next_fire_front = list()
        state.sample_idx = np.asarray(state.index_set[:sample_num])
        return state
    # End of forest_fire_sample()

    @staticmethod
    def max_degree_sample(graph: Graph, num_vertices: int, prev_state: DegreeWeightedSampleState,
                          args: argparse.Namespace) -> SampleState:
        """Max-degree sampling. Simply samples the highest-degree vertices.

        Parameters
        ----------
        graph : Graph
            the filtered graph from which to sample vertices
        num_vertices : int
            number of vertices in the unfiltered graph
        prev_state : UniformRandomSampleState
            the state of the previous sample in the stack. If there is no previous sample, an empty SampleState object
            should be passed in here.
        args : argparse.Namespace
            the command-line arguments provided by the user

        Returns
        -------
        state : SampleState
            the sample state with the sampled vertex ids (Note: these ids correspond to the filtered graph, and have
            to be mapped back to the unfiltered graph)
        """
        state = MaxDegreeSampleState(graph.num_vertices(), prev_state)
        sample_num = int((num_vertices * (args.sample_size / 100)) / args.sample_iterations)
        vertex_degrees = graph.get_total_degrees(np.arange(graph.num_vertices()))
        vertex_degrees[state.sample_idx] = 0
        top_indices = np.argpartition(vertex_degrees, -sample_num)[-sample_num:]
        state.sample_idx = np.concatenate((state.sample_idx, top_indices))
        return state
    # End of max_degree_sample()

    @staticmethod
    def expansion_snowball_sample(graph: Graph, num_vertices: int, prev_state: ExpansionSnowballSampleState,
                                  args: argparse.Namespace) -> SampleState:
        """Expansion snowball sampling. At every iteration, picks a vertex adjacent to the current sample that
        contributes the most new neighbors.

        Parameters
        ----------
        graph : Graph
            the filtered graph from which to sample vertices
        num_vertices : int
            number of vertices in the unfiltered graph
        prev_state : UniformRandomSampleState
            the state of the previous sample in the stack. If there is no previous sample, an empty SampleState object
            should be passed in here.
        args : argparse.Namespace
            the command-line arguments provided by the user

        Returns
        -------
        state : SampleState
            the sample state with the sampled vertex ids (Note: these ids correspond to the filtered graph, and have
            to be mapped back to the unfiltered graph)
        """
        state = ExpansionSnowballSampleState(graph.num_vertices(), prev_state)
        sample_num = int((num_vertices * (args.sample_size / 100)) / args.sample_iterations)
        sample_num += len(state.sample_idx)
        if not state.neighbors:  # If there are no neighbors, start with the state.start vertex
            state.index_flag[state.start] = True
            state.neighbors = set(graph.get_out_neighbors(state.start))
            for neighbor in graph.get_out_neighbors(state.start):
                if neighbor == state.start:
                    state.neighbors.remove(neighbor)
                else:
                    state.neighbors_flag[neighbor] = True
                    new_neighbors = 0
                    for _neighbor in graph.get_out_neighbors(neighbor):
                        if not (state.index_flag[_neighbor] or state.neighbors_flag[_neighbor]):
                            new_neighbors += 1
                    state.contribution[neighbor] += new_neighbors
        while len(state.index_set) == 0 or len(state.index_set) % sample_num != 0:
            if len(state.neighbors) == 0:  # choose random vertex not in index set
                vertex = np.random.choice(np.setxor1d(np.arange(graph.num_vertices()), state.index_set))
                state.index_set.append(vertex)
                state.index_flag[vertex] = True
                for neighbor in graph.get_out_neighbors(vertex):
                    if not (state.neighbors_flag[neighbor] or state.index_flag[neighbor]):
                        Sample._add_neighbor(neighbor, state.contribution, state.index_flag, state.neighbors_flag,
                                             graph.get_out_neighbors(neighbor), graph.get_in_neighbors(neighbor),
                                             state.neighbors)
                continue
            elif np.max(state.contribution) == 0:  # choose random neighbors from neighbor set
                num_choices = min(len(state.neighbors), sample_num - len(state.index_set))
                vertices = np.random.choice(np.fromiter(state.neighbors, int, len(state.neighbors)),
                                            num_choices, replace=False)
                for vertex in vertices:
                    state.index_set.append(vertex)
                    state.index_flag[vertex] = True
                    state.neighbors.remove(vertex)
                    for neighbor in graph.get_out_neighbors(vertex):
                        if not (state.neighbors_flag[neighbor] or state.index_flag[neighbor]):
                            Sample._add_neighbor(neighbor, state.contribution, state.index_flag, state.neighbors_flag,
                                                 graph.get_out_neighbors(neighbor), graph.get_in_neighbors(neighbor),
                                                 state.neighbors)
                continue
            vertex = np.argmax(state.contribution)
            state.index_set.append(vertex)
            state.index_flag[vertex] = True
            state.neighbors.remove(vertex)
            state.contribution[vertex] = 0
            for neighbor in graph.get_in_neighbors(vertex):
                if not (state.neighbors_flag[neighbor] or state.index_flag[neighbor]):
                    Sample._add_neighbor(neighbor, state.contribution, state.index_flag, state.neighbors_flag,
                                         graph.get_out_neighbors(neighbor), graph.get_in_neighbors(neighbor),
                                         state.neighbors)
        state.sample_idx = np.asarray(state.index_set)
        return state
    # End of expansion_snowball_sample()

    @staticmethod
    def _add_neighbor(vertex: int, contribution: List[int], index_flag: List[bool], neighbor_flag: List[bool],
                      out_neighbors: np.ndarray, in_neighbors: np.ndarray, neighbors: Set[int]):
        #    -> Tuple[List[int], List[bool]]:
        """Updates the expansion contribution for neighbors of a single vertex.
        """
        neighbors.add(vertex)
        neighbor_flag[vertex] = True
        if contribution[vertex] == 0:
            Sample._calculate_contribution(vertex, contribution, index_flag, neighbor_flag, out_neighbors, in_neighbors)
    # End of _add_neighbor()

    @staticmethod
    def _calculate_contribution(vertex: int, contribution: List[int], index_flag: List[bool], neighbor_flag: List[bool],
                                out_neighbors: np.ndarray, in_neighbors: np.ndarray):
        """Updates the contributions of the sample neighbors. This function is currently the bottleneck for the
        expansion snowball sampling technique.
        TODO: change calculation so it only has to subtract contributions. i.e. start with contributions = out_degrees,
        and whenever a vertex is added to the sample, subtract 1 from the relevant contributions.
        """
        # Compute contribution of this vertex
        for out_neighbor in out_neighbors:
            if not (index_flag[out_neighbor] or neighbor_flag[out_neighbor]):
                contribution[vertex] += 1
        # Decrease contribution of all neighbors with out links to this vertex
        for in_neighbor in in_neighbors:
            if contribution[in_neighbor] > 0:
                contribution[in_neighbor] -= 1
    # End of _calculate_contribution()
# End of Sample()
