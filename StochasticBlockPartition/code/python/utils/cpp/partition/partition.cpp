#include "partition.hpp"

void Partition::initialize_edge_counts(std::vector<Matrix2Column> neighbors) {
    this->blockmodel = BoostMappedMatrix(this->num_blocks, this->num_blocks);
    // Initialize the blockmodel
    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
        Matrix2Column vertex_neighbors = neighbors[vertex];
        if (vertex_neighbors.rows() == 0) {
            continue;
        }
        for (int i = 0; i < vertex_neighbors.rows(); ++i) {
            // Get count
            int block = this->block_assignment[vertex];
            int neighbor = vertex_neighbors(i, 0);
            int neighbor_block = this->block_assignment[neighbor];
            int weight = vertex_neighbors(i, 1);
            // Update blockmodel
            this->blockmodel.add(block, neighbor_block, weight);
            // Update degrees
            this->block_degrees_out[block] += weight;
            this->block_degrees_in[neighbor] += weight;
        }
    }
    // Count block degrees
    this->block_degrees = this->block_degrees_out + this->block_degrees_in;
}

Partition Partition::clone_with_true_block_membership(std::vector<Matrix2Column> neighbors,
                                                      Vector true_block_membership) {
    int num_blocks = 0;
    Vector uniques = Vector::Zero(true_block_membership.size());
    for (uint i = 0; i < true_block_membership.size(); ++i) {
        int membership = true_block_membership[i];
        uniques[membership] = 1; // mark as used
    }
    for (uint block = 0; block < uniques.size(); ++block) {
        if (uniques[block] == 1) {
            num_blocks++;
        }
    }
    return Partition(num_blocks, neighbors, this->block_reduction_rate, true_block_membership);
}

Partition Partition::copy() {
    Partition partition_copy = Partition(this->num_blocks, std::vector<Matrix2Column>(), this->block_reduction_rate);
    partition_copy.block_assignment = Vector(this->block_assignment);
    partition_copy.overall_entropy = this->overall_entropy;
    partition_copy.blockmodel = this->blockmodel.copy();
    partition_copy.block_degrees = Vector(this->block_degrees);
    partition_copy.block_degrees_out = Vector(this->block_degrees_out);
    partition_copy.block_degrees_in = Vector(this->block_degrees_in);
    partition_copy.num_blocks_to_merge = 0;
    return partition_copy;
}

Partition Partition::from_sample(int num_blocks, std::vector<Matrix2Column> neighbors, Vector sample_block_membership,
                                 std::map<int, int> mapping, float block_reduction_rate) {
    // Fill in initial block assignment
    Vector block_assignment = Vector::Constant(num_blocks, -1);
    for (const auto& item : mapping) {
        block_assignment[item.first] = sample_block_membership[item.second];
    }
    // Every unassigned block gets assigned to the next block number
    int next_block = num_blocks;
    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
        if (block_assignment[vertex] >= 0) {
            continue;
        }
        block_assignment[vertex] = next_block;
        next_block++;
    }
    // Every previously unassigned block gets assigned to the block it's most connected to
    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
        if (block_assignment[vertex] < num_blocks) {
            continue;
        }
        Vector block_counts = Vector::Zero(num_blocks);
        Matrix2Column vertex_neighbors = neighbors[vertex];
        for (uint i = 0; i < vertex_neighbors.rows(); ++i) {
            int neighbor = vertex_neighbors(i, 0);
            int neighbor_block = block_assignment[neighbor];
            if (neighbor_block < num_blocks) {
                block_counts[neighbor_block]++;
            }
        }
        int new_block;
        block_counts.maxCoeff(&new_block);
        block_assignment[vertex] = new_block;
    }
    return Partition(num_blocks, neighbors, block_reduction_rate, block_assignment);
}

void Partition::merge_blocks(int from_block, int to_block) {
    for (uint index = 0; index < this->block_assignment.size(); ++index) {
        if (this->block_assignment[index] == from_block) {
            this->block_assignment[index] = to_block;
        }
    }
};

Vector sort_indices(Eigen::VectorXd unsorted) {
    // initialize original index locations
    Vector indices = Vector::LinSpaced(1, 0, unsorted.size() - 1);

    // sort indexes based on comparing values in v
    // std::sort(indices.begin(), indices.end(), [unsorted](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    std::sort(indices.data(), indices.data() + indices.size(), 
              [unsorted](size_t i1, size_t i2) { return unsorted[i1] > unsorted[i2]; }
    );

    return indices;
}

void Partition::carry_out_best_merges(Eigen::VectorXd delta_entropy_for_each_block, Vector best_merge_for_each_block) {
    Vector best_merges = sort_indices(delta_entropy_for_each_block);
    Vector block_map = Vector::LinSpaced(1, 0, this->num_blocks - 1);
    int num_merged = 0;
    int counter = 0;
    while (num_merged < this->num_blocks_to_merge) {
        int merge_from = best_merges[counter];
        int merge_to = block_map[best_merge_for_each_block[merge_from]];
        counter++;
        if (merge_to != merge_from) {
            for (uint i = 0; i < block_map.size(); ++i) {
                int block = block_map[i];
                if (block == merge_from) {
                    block_map[i] = merge_to;
                }
            }
            this->merge_blocks(merge_from, merge_to);
            num_merged++;
        }
    }
    Vector remaining_blocks = Vector(this->block_assignment);
    auto end = std::unique(remaining_blocks.data(), remaining_blocks.data() + remaining_blocks.size());
    remaining_blocks.resize(std::distance(remaining_blocks.data(), end));
    Vector mapping = Vector::Constant(this->num_blocks, -1);
    // mapping[remaining_blocks] = np.arange(len(remaining_blocks))
    counter = 0;
    for (uint i = 0; i < remaining_blocks.size(); ++i) {
        int rem_block = remaining_blocks[i];
        mapping[rem_block] = counter;
        counter++;
    }
    // partition.block_assignment = mapping[partition.block_assignment]
    for (uint i = 0; i < this->block_assignment.size(); ++i) {
        int block = this->block_assignment[i];
        int new_block = mapping[block];
        this->block_assignment[block] = new_block;
    }
    this->num_blocks -= this->num_blocks_to_merge;
}