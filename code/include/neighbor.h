#ifndef KGRAPH_NEIGHBOR_H
#define KGRAPH_NEIGHBOR_H

#include <unordered_set>
#include <mutex>
#include <stack>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <cstring>

namespace kgraph
{
    struct Neighbor
    {
        uint32_t id;
        float dist;
        bool flag;  // whether this entry is a newly found one
        bool isnew; // whether the new one in (i)th compare by (i-1)th
        short pid;
        Neighbor() : dist(std::numeric_limits<float>::max()), isnew(true), pid(-1) {}
        Neighbor(unsigned i, float d, bool f = true) : id(i), dist(d), isnew(true), flag(f), pid(-1) {}
        Neighbor(bool is, unsigned i, float d, bool f = true) : id(i), dist(d), flag(f), isnew(is), pid(-1) {}
        inline bool operator<(Neighbor const &other)
        {
            return dist < other.dist;
        }
        inline bool operator==(Neighbor const &other)
        {
            return id == other.id;
        }
    };

    // extended neighbor structure for search time
    struct NeighborX : public Neighbor
    {
        uint16_t m;
        uint16_t M; // actual M used
        NeighborX() {}
        NeighborX(unsigned i, float d) : Neighbor(i, d, true), m(0), M(0)
        {
        }
    };

    struct RankNeighbor
    {
        unsigned id;
        unsigned rank;
        RankNeighbor() {}
        RankNeighbor(unsigned i, unsigned r) : id(i), rank(r) {}
        inline bool operator<(RankNeighbor const &other)
        {
            return rank < other.rank;
        }
    };

    static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn)
    {
        // find the location to insert
        int left = 0, right = K - 1;
        if (addr[left].dist > nn.dist)
        {
            memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
            addr[left] = nn;
            return left;
        }
        if (addr[right].dist < nn.dist)
        {
            addr[K] = nn;
            return K;
        }
        while (left < right - 1)
        {
            int mid = (left + right) / 2;
            if (addr[mid].dist > nn.dist)
            {
                right = mid;
            }
            else
            {
                left = mid;
            }
        }

        while (left > 0)
        {
            if (addr[left].dist < nn.dist)
            {
                break;
            }
            if (addr[left].id == nn.id)
            {
                return K + 1;
            }
            left--;
        }
        if (addr[left].id == nn.id || addr[right].id == nn.id)
        {
            return K + 1;
        }
        memmove((char *)&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
        addr[right] = nn;
        return right;
    }

} // namespace kgraph

#endif