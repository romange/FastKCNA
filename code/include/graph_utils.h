#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "kgraph.h"
#include "kgraph-data.h"

namespace kgraph
{
    int getRandomLevel(double reverse_size, std::default_random_engine &level_generator_)
    {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int)r;
    }

    void getLevel(unsigned point_num, std::vector<int> &levels, size_t M_, unsigned random_seed)
    {
        levels.resize(point_num);
        std::default_random_engine level_generator_;
        double muti_ = 1 / log(1.0 * M_);
        level_generator_.seed(random_seed);
        for (unsigned i = 0; i < point_num; ++i)
        {
            int level = getRandomLevel(muti_, level_generator_);
            levels[i] = level;
        }
    }

    void getMapping(unsigned point_num, std::vector<int> &levels, std::vector<unsigned> &labels, std::vector<unsigned> &file2data, std::vector<int> &num_perlevel)
    {
        labels.resize(point_num);
        labels.clear();
        file2data.resize(point_num);
        auto it = std::max_element(levels.begin(), levels.end());
        int max_level = *it;
        num_perlevel.resize(max_level + 1, 0);
        for (int i = max_level; i >= 0; --i)
        {
            if (i != max_level)
            {
                num_perlevel[i] += num_perlevel[i + 1];
            }
            for (unsigned j = 0; j < point_num; ++j)
            {
                if (levels[j] == i)
                {
                    labels.push_back(j);
                    num_perlevel[i]++;
                }
            }
        }
        for (unsigned i = 0; i < point_num; ++i)
        {
            file2data[labels[i]] = i;
        }
    }

    unsigned getPointNum(const char *data_path)
    {
        std::ifstream is(data_path, std::ios::binary);
        if (!is.is_open())
        {
            std::cout << "can't open data file!" << std::endl;
            return 0;
        }
        unsigned point_num = 0;
        is.read(reinterpret_cast<char *>(&point_num), sizeof(point_num));
        is.read(reinterpret_cast<char *>(&point_num), sizeof(point_num));
        is.close();
        return point_num;
    }

    template <typename T>
    static void writeBinaryPOD(std::ostream &out, const T &podRef)
    {
        out.write((char *)&podRef, sizeof(T));
    }

    template <typename T>
    void save_hnsw(kgraph::Matrix<T> &data, std::vector<std::vector<std::vector<unsigned>>> &graphs,
                   KGraph::IndexParams &params, unsigned ep, std::vector<int> &levels,
                   std::vector<unsigned> &labels, const char *output_path)
    {
        size_t max_elements_ = data.size();
        size_t cur_element_count = data.size();
        size_t M_ = params.nsg_R;
        size_t maxM_ = params.nsg_R;
        size_t maxM0_ = 2 * params.nsg_R;
        size_t ef_construction_ = params.search_K;
        size_t size_links_per_element_ = maxM_ * sizeof(unsigned) + sizeof(unsigned);
        size_t size_links_level0_ = maxM0_ * sizeof(unsigned) + sizeof(unsigned);
        size_t size_data_per_element_ = size_links_level0_ + sizeof(T) * data.dim() + sizeof(size_t);
        size_t offsetData_ = size_links_level0_;
        size_t label_offset_ = size_links_level0_ + sizeof(T) * data.dim();
        size_t offsetLevel0_ = 0;
        char *data_level0_memory_ = nullptr;
        char **linkLists_ = (char **)malloc(sizeof(void *) * data.size());
        int maxlevel_ = *(std::max_element(levels.begin(), levels.end()));
        double mult_ = 1 / log(1.0 * M_);
        unsigned int enterpoint_node_ = ep;
        data_level0_memory_ = (char *)malloc(data.size() * size_data_per_element_);
        memset(data_level0_memory_, 0, data.size() * size_data_per_element_);
        for (unsigned i = 0; i < data.size(); i++)
        {
            unsigned *cll = (unsigned *)(data_level0_memory_ + i * size_data_per_element_ + offsetLevel0_);
            int size = graphs[0][i].size();
            unsigned short linkCount = static_cast<unsigned short>(size);
            *((unsigned short int *)(cll)) = *((unsigned short int *)&linkCount);
            unsigned *nei = (unsigned *)(cll + 1);
            for (unsigned j = 0; j < size; j++)
            {
                nei[j] = graphs[0][i][j];
            }
            size_t label = labels[i];
            char *c_data = data.getdata(i);
            memcpy((data_level0_memory_ + i * size_data_per_element_ + offsetData_), c_data, sizeof(float) * data.dim());
            memcpy((data_level0_memory_ + i * size_data_per_element_ + label_offset_), &label, sizeof(label));
        }

        std::ofstream output(output_path, std::ios::binary);
        std::streampos position;
        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);
        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);
        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);
        free(data_level0_memory_);

        for (int cur_c = 0; cur_c < levels.size(); cur_c++)
        {
            if (levels[labels[cur_c]] > 0)
            {
                linkLists_[cur_c] = (char *)malloc(size_links_per_element_ * levels[labels[cur_c]] + 1);
                memset(linkLists_[cur_c], 0, size_links_per_element_ * levels[labels[cur_c]] + 1);
            }
        }

        for (int level = 1; level <= maxlevel_; level++)
        {
            for (unsigned i = 0; i < data.size(); i++)
            {
                if (levels[labels[i]] < level)
                {
                    continue;
                }
                unsigned *cll = (unsigned *)(linkLists_[i] + (level - 1) * size_links_per_element_);
                int size = graphs[level][i].size();
                unsigned short linkCount = static_cast<unsigned short>(size);
                *((unsigned short int *)(cll)) = *((unsigned short int *)&linkCount);
                unsigned *nei = (unsigned *)(cll + 1);
                for (unsigned j = 0; j < size; j++)
                {
                    nei[j] = graphs[level][i][j];
                }
            }
        }

        for (size_t i = 0; i < cur_element_count; i++)
        {
            unsigned int linkListSize = levels[labels[i]] > 0 ? size_links_per_element_ * levels[labels[i]] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
        for (int i = 0; i < cur_element_count; i++)
        {
            if (levels[labels[i]] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
    }

    void save_pg(std::vector<std::vector<std::vector<unsigned>>> &graphs, unsigned ep, const char *output_file)
    {
        auto &graph = graphs[0];
        unsigned nd_ = graph.size();
        std::ofstream out(output_file, std::ios::binary | std::ios::out);
        unsigned width = 0;
        for (int i = 0; i < nd_; i++)
        {
            if (graph[i].size() > width)
            {
                width = graph[i].size();
            }
        }
        out.write((char *)&width, sizeof(unsigned));
        out.write((char *)&ep, sizeof(unsigned));
        for (unsigned i = 0; i < nd_; i++)
        {
            unsigned GK = graph[i].size();
            out.write((char *)&GK, sizeof(unsigned));
            vector<unsigned> res;
            res.resize(GK);
            for (int j = 0; j < GK; j++)
            {
                res[j] = graph[i][j];
            }
            out.write((char *)res.data(), GK * sizeof(unsigned));
        }
        out.close();
    }

} // namespace kgraph

#endif