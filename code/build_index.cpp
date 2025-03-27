#include "kgraph.h"
#include "kgraph-data.h"
#include "utils.h"
#include "graph_utils.h"
#include <omp.h>

int main(int argc, char *argv[])
{

    kgraph::KGraph::IndexParams params;
    char *ptr = nullptr;
    std::vector<std::string> input_params;
    for (int i = 0; i < argc; ++i)
    {
        string arg = argv[i];
        input_params.push_back(arg);
        if (arg == "-data_path")
        {
            params.data_path = std::string(argv[i + 1]);
            std::cout << "data_path: " << params.data_path << std::endl;
        }
        if (arg == "-index_path")
        {
            params.index_path = std::string(argv[i + 1]);
            std::cout << "index_path: " << params.index_path << std::endl;
        }
        if (arg == "-log_path")
        {
            params.log_path = std::string(argv[i + 1]);
            std::cout << "log_path: " << params.log_path << std::endl;
        }
        if (arg == "-K")
        {
            params.K = (unsigned)strtoul(argv[i + 1], &ptr, 10);
            std::cout << "params.K: " << params.K << std::endl;
        }
        if (arg == "-L")
        {
            params.L = (unsigned)strtoul(argv[i + 1], &ptr, 10);
            std::cout << "params.L: " << params.L << std::endl;
        }
        if (arg == "-S")
        {
            params.S = (unsigned)strtoul(argv[i + 1], &ptr, 10);
            std::cout << "params.S: " << params.S << std::endl;
        }
        if (arg == "-R")
        {
            params.R = (unsigned)strtoul(argv[i + 1], &ptr, 10);
            std::cout << "params.R: " << params.R << std::endl;
        }
        if (arg == "-iter")
        {
            params.iterations = (unsigned)strtoul(argv[i + 1], &ptr, 10);
            std::cout << "params.iterations: " << params.iterations << std::endl;
        }
        if (arg == "-search_L")
        {
            params.search_L = (unsigned)strtoul(argv[i + 1], &ptr, 10);
            std::cout << "params.search_L: " << params.search_L << std::endl;
        }
        if (arg == "-search_K")
        {
            params.search_K = (unsigned)strtoul(argv[i + 1], &ptr, 10);
            std::cout << "params.search_K: " << params.search_K << std::endl;
        }
        if (arg == "-nsg_R")
        {
            params.nsg_R = (unsigned)strtoul(argv[i + 1], &ptr, 10);
            std::cout << "params.nsg_R: " << params.nsg_R << std::endl;
        }
        if (arg == "-step")
        {
            params.step = (unsigned)strtoul(argv[i + 1], &ptr, 10);
            std::cout << "params.step: " << params.step << std::endl;
        }
        if (arg == "-alpha")
        {
            params.alpha = atof(argv[i + 1]);
            std::cout << "params.alpha: " << params.alpha << std::endl;
        }
        if (arg == "-tau")
        {
            params.tau = atof(argv[i + 1]);
            std::cout << "params.tau: " << params.tau << std::endl;
        }
        if (arg == "-loop_i")
        {
            params.loop_i = (unsigned)strtoul(argv[i + 1], &ptr, 10);
            std::cout << "params.loop_i: " << params.loop_i << std::endl;
        }
        if (arg == "-nthreads")
        {
            params.nthreads = (unsigned)strtoul(argv[i + 1], &ptr, 10);
            std::cout << "params.nthreads: " << params.nthreads << std::endl;
        }
        if (arg == "-controls")
        {
            params.controls = (unsigned)strtoul(argv[i + 1], &ptr, 10);
            std::cout << "params.controls: " << params.controls << std::endl;
        }
        if (arg == "-recall")
        {
            params.recall = atof(argv[i + 1]);
            std::cout << "params.recall: " << params.recall << std::endl;
        }
        if (arg == "-pg_type")
        {
            params.pg_type = (unsigned)strtoul(argv[i + 1], &ptr, 10);
            std::cout << "params.pg_type: " << params.pg_type << std::endl;
        }
    }
    {
        auto it = std::find(input_params.begin(), input_params.end(), "-K");
        if (it == input_params.end())
        {
            params.K = params.search_K;
        }
        it = std::find(input_params.begin(), input_params.end(), "-L");
        if (it == input_params.end())
        {
            params.L = params.search_K;
        }
    }
    params.printparams();
    omp_set_num_threads(params.nthreads);
    kgraph::Matrix<float> matrix;
    std::vector<int> levels;
    std::vector<int> num_perlevel;
    std::vector<unsigned> labels;
    if (params.pg_type == kgraph::INDEX_HNSW || params.pg_type == kgraph::INDEX_NSW)
    {
        unsigned point_num = kgraph::getPointNum(params.data_path.c_str());
        if (params.pg_type == kgraph::INDEX_HNSW)
        {
            kgraph::getLevel(point_num, levels, params.nsg_R, params.seed);
        }
        else
        {
            levels.resize(point_num, 0);
        }

        std::vector<unsigned> file2data;
        kgraph::getMapping(point_num, levels, labels, file2data, num_perlevel);
        matrix.load_lshkit(params.data_path, file2data.data());
    }
    else
    {
        matrix.load_lshkit(params.data_path);
    }

    kgraph::MatrixOracle<float, kgraph::metric::l2sqr> oracle(matrix);

    std::vector<std::vector<std::vector<unsigned>>> graphs;

    auto start = std::chrono::high_resolution_clock::now();
    kgraph::KGraph *index = kgraph::KGraph::create();
    unsigned entry_point;

    if (params.pg_type == kgraph::INDEX_HNSW || params.pg_type == kgraph::INDEX_NSW)
    {
        unsigned all_level = num_perlevel.size();
        graphs.resize(all_level);
        unsigned temp_entry_point;
        for (int level = all_level - 1; level >= 0; --level)
        {
            oracle.set_size(num_perlevel[level]);
            if (level == 0)
            {
                params.nsg_R *= 2;
            }
            kgraph::KGraph::IndexInfo info;
            index->build(oracle, params, &info, graphs[level], temp_entry_point);
            kgraph::save_log(params.log_path.c_str(), params, info);
            if (level == 0)
            {
                params.nsg_R /= 2;
            }
            if (level == all_level - 1)
            {
                entry_point = temp_entry_point;
            }
        }
    }
    else
    {
        graphs.resize(1);
        kgraph::KGraph::IndexInfo info;
        index->build(oracle, params, &info, graphs[0], entry_point);
        kgraph::save_log(params.log_path.c_str(), params, info);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double index_time = diff.count();
    std::cout << "Indexing time: " << index_time << "[s]" << std::endl;

    delete index;
    if (params.pg_type == kgraph::INDEX_HNSW || params.pg_type == kgraph::INDEX_NSW)
    {
        kgraph::save_hnsw<float>(matrix, graphs, params, entry_point, levels, labels, params.index_path.c_str());
    }
    else
    {
        kgraph::save_pg(graphs, entry_point, params.index_path.c_str());
    }
    long peakmemory = kgraph::getPeakMemoryUsage();
    std::cout << "Peak memory: " << peakmemory << "[MB]" << std::endl;
    kgraph::write_sep(params.log_path.c_str(), index_time, peakmemory);
    return 0;
}