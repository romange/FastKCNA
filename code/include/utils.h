#ifndef KGRAPH_UTILS_H
#define KGRAPH_UTILS_H

#include <thread>
#include <mutex>
#include <sys/resource.h>
#include <memory>
#include <algorithm>
#include <vector>

namespace kgraph
{
    // Get peak memory usages
    inline long getPeakMemoryUsage()
    {
        struct rusage r_usage;
        getrusage(RUSAGE_SELF, &r_usage);
        return r_usage.ru_maxrss / 1024;
    }

    // Arrange the indexes in ascending order of elements
    template <typename T>
    std::vector<unsigned> sort_indexes(const std::vector<T> &v)
    {

        // initialize original index locations
        std::vector<unsigned> idx(v.size());
        for (size_t i = 0; i != idx.size(); ++i)
            idx[i] = i;

        // sort indexes based on comparing values in v
        std::sort(idx.begin(), idx.end(),
                  [&v](unsigned i1, unsigned i2)
                  {
                      if (v[i1] == v[i2])
                      {
                          return i1 < i2;
                      }
                      return v[i1] < v[i2];
                  });

        return idx;
    }

    // Arrange the indexes in descending order of elements
    template <typename T>
    std::vector<unsigned> sort_indexes_greater(const std::vector<T> &v)
    {

        // initialize original index locations
        std::vector<unsigned> idx(v.size());
        for (size_t i = 0; i != idx.size(); ++i)
            idx[i] = i;

        // sort indexes based on comparing values in v
        std::sort(idx.begin(), idx.end(),
                  [&v](unsigned i1, unsigned i2)
                  {
                      if (v[i1] == v[i2])
                      {
                          return i1 < i2;
                      }
                      return v[i1] > v[i2];
                  });

        return idx;
    }

    // generate size distinct random numbers < N
    template <typename RNG>
    void GenRandom(RNG &rng, unsigned *addr, unsigned size, unsigned N)
    {
        if (N == size)
        {
            for (unsigned i = 0; i < size; ++i)
            {
                addr[i] = i;
            }
            return;
        }
        for (unsigned i = 0; i < size; ++i)
        {
            addr[i] = rng() % (N - size);
        }
        std::sort(addr, addr + size);
        for (unsigned i = 1; i < size; ++i)
        {
            if (addr[i] <= addr[i - 1])
            {
                addr[i] = addr[i - 1] + 1;
            }
        }
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i)
        {
            addr[i] = (addr[i] + off) % N;
        }
    }

    inline void save_log(const char *log_file, kgraph::KGraph::IndexParams &params, kgraph::KGraph::IndexInfo info)
    {
        ofstream ofs(log_file, std::ios::out | std::ios::app);
        if (ofs.is_open())
        {
            ofs << "pg_type,K,L,S,R,I,controls,recall,searchL,searchK,nsgR,massqS,tau,alpha,step" << endl;
            ofs << params.pg_type << "," << params.K << "," << params.L << "," << params.S << "," << params.R << ","
                << params.iterations << "," << params.controls << "," << params.recall << "," << params.search_L << ","
                << params.search_K << "," << params.nsg_R << "," << params.massq_S << "," << params.tau << ","
                << params.alpha << "," << params.step << "," << endl;
            ofs << "kgraph time" << ",";
            for (unsigned i = 0; i < info.kgraph_time.size(); i++)
            {
                ofs << info.kgraph_time[i] << ",";
            }
            ofs << endl;
            ofs << "kgraph recall" << ",";
            for (unsigned i = 0; i < info.kgraph_recall.size(); i++)
            {
                ofs << info.kgraph_recall[i] << ",";
            }
            ofs << endl;
            ofs << "kcna time" << ",";
            for (unsigned i = 0; i < info.kcna_time.size(); i++)
            {
                ofs << info.kcna_time[i] << ",";
            }
            ofs << endl;
            ofs << "kcna recall" << ",";
            for (unsigned i = 0; i < info.kcna_recall.size(); i++)
            {
                ofs << info.kcna_recall[i] << ",";
            }
            ofs << endl;
            ofs << "buildpg time" << ",";
            for (unsigned i = 0; i < info.buildpg_time.size(); i++)
            {
                ofs << info.buildpg_time[i] << ",";
            }
            ofs << endl;
            ofs << "refinement time" << ",";
            for (unsigned i = 0; i < info.refinement_time.size(); i++)
            {
                ofs << info.refinement_time[i] << ",";
            }
            ofs << endl;
            ofs << "search time" << ",";
            for (unsigned i = 0; i < info.search_time.size(); i++)
            {
                ofs << info.search_time[i] << ",";
            }
            ofs << endl;
            ofs << "pruning time" << ",";
            for (unsigned i = 0; i < info.prune_time.size(); i++)
            {
                ofs << info.prune_time[i] << ",";
            }
            ofs << endl;
            ofs << "add reverse edges time" << ",";
            for (unsigned i = 0; i < info.add_reverse_time.size(); i++)
            {
                ofs << info.add_reverse_time[i] << ",";
            }
            ofs << endl;
            ofs << "tree grow time" << ",";
            for (unsigned i = 0; i < info.tree_grow_time.size(); i++)
            {
                ofs << info.tree_grow_time[i] << ",";
            }
            ofs << endl;
            ofs << "prune scan_rate" << ",";
            for (unsigned i = 0; i < info.prune_cost.size(); i++)
            {
                ofs << info.prune_cost[i] << ",";
            }
            ofs << endl;
            ofs << "search scan_rate" << ",";
            for (unsigned i = 0; i < info.search_cost.size(); i++)
            {
                ofs << info.search_cost[i] << ",";
            }
            ofs << endl;
            ofs << "AOD" << "," << info.AOD;
            ofs << endl;
            ofs.close();
        }
    }

    inline void write_sep(const char *log_file, float index_time, float peakmemory)
    {
        ofstream ofs(log_file, std::ios::out | std::ios::app);
        if (ofs.is_open())
        {
            ofs << "indexing time" << "," << index_time << endl;
            ofs << "peak memoey" << "," << peakmemory << endl;
            ofs << "***************************************************************************************************************" << endl
                << endl;
            ofs.close();
        }
    }

} // namespace kgraph

#endif