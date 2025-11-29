# FastKCNA

## Experimental Environment

All experiments are conducted on a server equipped with 2 Intel(R) Xeon(R) Silver 4210R CPUs, each of which has 10 cores, and 256 GB DRAM as the main memory. The OS version is CentOS 7.9.2009. All codes were written by {C++} and compiled by {g++ 11.3}. The SIMD instructions are enabled to accelerate the distance computations.

## Building Instruction

### Prerequisites

cmake g++ OpenMP Boost

### Compile

```
cd code
cmake .
make
```

## Usage

```
./build_index -data_path datapath.lshkit -index_path indexpath.fastnsg -log_path logpath.csv -K 500 -L 500 -S 12 -R 100 -iter 6 -search_L 80 -nsg_R 50 -search_K 500 -step 10 -loop_i 2 -alpha 60 -tau 0 -pg_type 1
```

-pg_type: 0(KNNG); 1(NSG); 2(HNSW); 3($\tau$-MNG); 4(NSW); 5($\alpha$-PG)

Note : You can use fvec2lshkit.cpp to convert fvec format data into lshkit format data. In FastHNSW, nsg_R has the same effect as M in the original HNSW.

The candidate neighbor set size will be the largest of K, L, search_L, search_K.

The index structure of FastHNSW is the same as that of HNSW in hnswlib.

## HNSW Implementation Details

This section provides a detailed analysis of the HNSW (Hierarchical Navigable Small World) implementation in FastKCNA.

### Level Assignment (`code/include/graph_utils.h`)

Each node is assigned a random level using an exponential distribution:

```cpp
int getRandomLevel(double reverse_size, std::default_random_engine &level_generator_) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (int)r;
}
```

- The formula `-log(random) * (1/log(M))` ensures level `l` contains approximately `n/M^l` nodes
- Level 0 contains all nodes, higher levels contain exponentially fewer nodes
- Data is reordered so higher-level nodes come first for efficient hierarchical construction

### Multi-Level Graph Building (`code/build_index.cpp`)

HNSW construction follows these steps:

1. **Top-down construction**: Builds from the highest level down to level 0
2. **Level 0 has 2× connections**: Following the original HNSW paper, level 0 uses `2*M` connections while upper levels use `M` connections
3. **Cumulative levels**: Each level contains nodes from that level and all higher levels

```cpp
for (int level = all_level - 1; level >= 0; --level) {
    oracle.set_size(num_perlevel[level]);
    if (level == 0) {
        params.nsg_R *= 2;  // Double connections at level 0
    }
    index->build(oracle, params, &info, graphs[level], temp_entry_point);
}
```

### NN-Descent Algorithm (`code/src/kgraph.cpp`)

NN-Descent is the core algorithm for constructing approximate k-nearest neighbor graphs. It works by iteratively refining neighbor estimates through a "local join" operation.

**Key insight**: "A neighbor of my neighbor is likely my neighbor too."

#### Algorithm Steps:

1. **Initialization** (lines 370-410): Random neighbors are assigned to each node
2. **Join** (lines 413-432): For each pair of neighbors `(i, j)` of node `n`, compute `dist(i, j)` and update both `i`'s and `j`'s neighbor lists if this distance is better than existing neighbors
3. **Update** (lines 435-517): Mark new/old neighbors for the next iteration to avoid redundant computations

The algorithm converges when few new neighbors are discovered (controlled by the `delta` parameter).

### RNG Pruning

The pruning uses an angle-based criterion (α-RNG):

```cpp
float cos_ij = (nhood.nn_nsg[t].dist + djk - p.dist) / 2 / sqrt(nhood.nn_nsg[t].dist * djk);
check = djk < p.dist && cos_ij < threshold;
```

This implements the **law of cosines** to determine if a candidate neighbor should be pruned:
- If an existing neighbor `t` is closer to candidate `p` than the query `q` is, and the angle is below a threshold, `p` is pruned
- This creates a sparse but high-quality graph structure

### BridgeView Search

A novel neighbor refinement technique (`BridgeView`) exploits the graph structure to discover better neighbors by examining reverse neighbors of proximity graph edges.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nsg_R` (M) | 50 | Max connections per node (upper levels) |
| `maxM0` | 2×M | Max connections at level 0 |
| `search_K` (ef_construction) | 500 | Size of dynamic candidate list during construction |
| `search_L` | 40 | Search list size |
| `alpha` | 60° | Angle threshold for RNG pruning |
| `loop_i` | 2 | Number of refinement iterations |

### Serialization (`code/include/graph_utils.h`)

The HNSW index is saved in binary format **compatible with hnswlib**:

- **Level 0 memory**: Contains node connections + raw data + labels
- **Higher level links**: Stored separately for nodes with level > 0
- **Metadata**: max_elements, M, maxM0 (=2M), efConstruction, entry point, etc.

This allows direct loading and querying with the hnswlib library.
