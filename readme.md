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
