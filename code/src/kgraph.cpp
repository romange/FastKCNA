#ifndef KGRAPH_VERSION
#define KGRAPH_VERSION unknown
#endif
#ifndef KGRAPH_BUILD_NUMBER
#define KGRAPH_BUILD_NUMBER
#endif
#ifndef KGRAPH_BUILD_ID
#define KGRAPH_BUILD_ID
#endif
#define STRINGIFY(x) STRINGIFY_HELPER(x)
#define STRINGIFY_HELPER(x) #x
static char const *kgraph_version = STRINGIFY(KGRAPH_VERSION) "-" STRINGIFY(KGRAPH_BUILD_NUMBER) "," STRINGIFY(KGRAPH_BUILD_ID);
#define mymean(x, n) x / n

#ifdef _OPENMP
#include <omp.h>
#endif
#include <unordered_set>
#include <mutex>
#include <stack>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include "kgraph.h"
#include <sys/resource.h>
#include <queue>
#include <assert.h>
#include <thread>
#include "utils.h"
#include <unistd.h>
#include "boost/smart_ptr/detail/spinlock.hpp"

constexpr double kPi = 3.14159265358979323846264;

namespace kgraph
{

    typedef std::vector<Neighbor> Neighbors;

    using namespace std;
    unsigned verbosity = default_verbosity;
    typedef boost::detail::spinlock Lock;
    typedef std::lock_guard<Lock> LockGuard;

    struct myIndegree
    {
        int value;
        int index;
        myIndegree(int value = -1, int index = 0) : value(value), index(index) {}
    };

    myIndegree maxIndegree(myIndegree &a, myIndegree &b)
    {
        if (a.value > b.value)
        {
            return a;
        }
        else
        {
            return b;
        }
    }
    int findMaxElementIndex(const std::vector<int> &vec)
    {
        int max_index = 0;

        myIndegree max_value2(-1, 0);

        int num_elements = vec.size();
#pragma omp declare reduction(MyMax:myIndegree : omp_out = maxIndegree(omp_out, omp_in))
        {
#pragma omp parallel for shared(vec) reduction(MyMax : max_value2)
            for (int i = 0; i < num_elements; ++i)
            {
                if (vec[i] > max_value2.value)
                {
                    max_value2.value = vec[i];
                    max_value2.index = i;
                }
            }
        }

        return max_value2.index;
    }

    // both pool and knn should be sorted in ascending order
    static float EvaluateRecall(Neighbors const &pool, unsigned K0, Neighbors const &knn, unsigned K)
    {
        if (knn.empty())
            return 1.0;
        unsigned found = 0;
        for (unsigned i = 0; i < min(K0, K); i++)
        {
            unsigned id = pool[i].id;
            for (unsigned j = 0; j < K; j++)
            {
                if (id == knn[j].id)
                {
                    found++;
                    break;
                }
            }
        }
        return float(found) / K;
    }

    static float EvaluateAccuracy(Neighbors const &pool, Neighbors const &knn)
    {
        unsigned m = std::min(pool.size(), knn.size());
        float sum = 0;
        unsigned cnt = 0;
        for (unsigned i = 0; i < m; ++i)
        {
            if (knn[i].dist > 0)
            {
                sum += abs(pool[i].dist - knn[i].dist) / knn[i].dist;
                ++cnt;
            }
        }
        return cnt > 0 ? sum / cnt : 0;
    }

    static float EvaluateOneRecall(Neighbors const &pool, Neighbors const &knn)
    {
        if (pool[0].dist == knn[0].dist)
            return 1.0;
        return 0;
    }

    static float EvaluateDelta(Neighbors const &pool, unsigned K)
    {
        unsigned c = 0;
        unsigned N = K;
        if (pool.size() < N)
            N = pool.size();
        for (unsigned i = 0; i < N; ++i)
        {
            if (pool[i].flag)
                ++c;
        }
        return float(c) / K;
    }

    struct Control
    {
        unsigned id;
        Neighbors neighbors;
    };

    // try insert nn into the list
    // the array addr must contain at least K+1 entries:
    //      addr[0..K-1] is a sorted list
    //      addr[K] is as output parameter
    // * if nn is already in addr[0..K-1], return K+1
    // * Otherwise, do the equivalent of the following
    //      put nn into addr[K]
    //      make addr[0..K] sorted
    //      return the offset of nn's index in addr (could be K)
    //
    // Special case:  K == 0
    //      addr[0] <- nn
    //      return 0
    template <typename NeighborT>
    unsigned UpdateKnnListHelper(NeighborT *addr, unsigned K, NeighborT nn)
    {
        // find the location to insert
        unsigned j;
        unsigned i = K;
        while (i > 0)
        {
            j = i - 1;
            if (addr[j].dist <= nn.dist)
                break;
            i = j;
        }
        // check for equal ID
        unsigned l = i;
        while (l > 0)
        {
            j = l - 1;
            if (addr[j].dist < nn.dist)
                break;
            if (addr[j].id == nn.id)
                return K + 1;
            l = j;
        }
        // i <= K-1
        j = K;
        while (j > i)
        {
            addr[j] = addr[j - 1];
            --j;
        }
        addr[i] = nn;
        return i;
    }

    static inline unsigned UpdateKnnList(Neighbor *addr, unsigned K, Neighbor nn)
    {
        return UpdateKnnListHelper<Neighbor>(addr, K, nn);
    }

    static inline unsigned UpdateKnnList(NeighborX *addr, unsigned K, NeighborX nn)
    {
        return UpdateKnnListHelper<NeighborX>(addr, K, nn);
    }

    void LinearSearch(IndexOracle const &oracle, unsigned i, unsigned K, vector<Neighbor> *pnns)
    {
        vector<Neighbor> nns(K + 1);
        unsigned N = oracle.size();
        Neighbor nn;
        nn.id = 0;
        // nn.flag = true; // we don't really use this
        unsigned k = 0;
        while (nn.id < N)
        {
            if (nn.id != i)
            {
                nn.dist = oracle(i, nn.id);
                UpdateKnnList(&nns[0], k, nn);
                if (k < K)
                    ++k;
            }
            ++nn.id;
        }
        nns.resize(K);
        pnns->swap(nns);
    }

    unsigned SearchOracle::search(unsigned K, float epsilon, unsigned *ids, float *dists) const
    {
        return 0;
    }

    void GenerateControl(IndexOracle const &oracle, unsigned C, unsigned K, vector<Control> *pcontrols)
    {
        auto s_computegt = std::chrono::high_resolution_clock::now();
        vector<Control> controls(C);
        {
            vector<unsigned> index(oracle.size());
            int i = 0;
            for (unsigned &v : index)
            {
                v = i++;
            }
            random_shuffle(index.begin(), index.end());
#pragma omp parallel for
            for (unsigned i = 0; i < C; ++i)
            {
                controls[i].id = index[i];
                LinearSearch(oracle, index[i], K, &controls[i].neighbors);
            }
        }
        pcontrols->swap(controls);
        auto e_computegt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_computegt = e_computegt - s_computegt;
        double Time_computegt = diff_computegt.count();
        cout << "Generating control time: " << Time_computegt << "[s]" << endl;
    }

    static char const *KGRAPH_MAGIC = "KNNGRAPH";
    static unsigned constexpr KGRAPH_MAGIC_SIZE = 8;
    static uint32_t constexpr SIGNATURE_VERSION = 2;

    class KGraphImpl : public KGraph
    {
    protected:
        vector<unsigned> M;
        bool no_dist; // Distance & flag information in Neighbor is not valid.

        // actual M for a node that should be used in search time
        unsigned actual_M(unsigned pM, unsigned i) const
        {
            return std::min(std::max(M[i], pM), unsigned(graph[i].size()));
        }

    public:
        vector<vector<Neighbor>> graph;
        virtual ~KGraphImpl()
        {
        }

        virtual void build(IndexOracle const &oracle, IndexParams const &params, IndexInfo *info, vector<vector<unsigned>> &graph, unsigned &entry_point);

        virtual unsigned search(SearchOracle const &oracle, SearchParams const &params, unsigned *ids, float *dists, SearchInfo *pinfo) const
        {
            std::cout << "KGraph::KgraphImpl::search" << std::endl;
            return 0;
        }

        virtual void get_nn(unsigned id, unsigned *nns, float *dist, unsigned *pM, unsigned *pL) const
        {
            std::cout << "KGraph::KgraphImpl::get_nn" << std::endl;
        }
    };

    class KGraphConstructor : public KGraphImpl
    {
        // The neighborhood structure maintains a pool of near neighbors of an object.
        // The neighbors are stored in the pool.  "n" (<="params.L") is the number of valid entries
        // in the pool, with the beginning "k" (<="n") entries sorted.
        struct Nhood
        {               // neighborhood
            bool found; // helped found new NN in this round
            Lock lock;
            unsigned L;   // # valid items in the pool,  L + 1 <= pool.size()
            unsigned M;   // we only join items in pool[0..M)
            float radius; // distance of interesting range
            float radiusM;
            Neighbors pool;
            vector<Neighbor> nn_nsg;
            vector<unsigned> nn_old;
            vector<unsigned> nn_new;
            vector<unsigned> rnn_old;
            vector<unsigned> rnn_new;
            vector<unsigned> old_nsg;
            vector<bool> rnn_new_flag;
            // only non-readonly method which is supposed to be called in parallel
            unsigned parallel_try_insert(unsigned id, float dist)
            {
                if (dist > radius)
                    return pool.size();
                LockGuard guard(lock);
                // unsigned l = UpdateKnnList(&pool[0], L, Neighbor(id, dist, true));
                Neighbor nn(id, dist);
                unsigned l = InsertIntoPool(&pool[0], L, nn);
                if (l <= L)
                { // inserted
                    if (L + 1 < pool.size())
                    { // if l == L + 1, there's a duplicate
                        ++L;
                    }
                    else
                    {
                        radius = pool[L - 1].dist;
                    }
                }
                return l;
            }

            // join should not be conflict with insert
            template <typename C>
            void join(C callback) const
            {
                for (unsigned const i : nn_new)
                {
                    for (unsigned const j : nn_new)
                    {
                        if (i < j)
                        {
                            callback(i, j);
                        }
                    }
                    for (unsigned j : nn_old)
                    {
                        callback(i, j);
                    }
                }
            }
        };

        IndexOracle const &oracle;
        vector<Nhood> nhoods;
        size_t n_comps;
        vector<Control> controls;
        unsigned MAXL = 0;

        void init()
        {
            unsigned N = oracle.size();
            unsigned seed = params.seed;
            mt19937 rng(seed);
            for (auto &nhood : nhoods)
            {
                nhood.nn_new.resize(params.S * 2);
                nhood.pool.resize(params.L + 1);
                nhood.radius = numeric_limits<float>::max();
            }
#pragma omp parallel
            {
#ifdef _OPENMP
                mt19937 rng(seed ^ omp_get_thread_num());
#else
                mt19937 rng(seed);
#endif
                vector<unsigned> random(params.S + 1);
#pragma omp for
                for (unsigned n = 0; n < N; ++n)
                {
                    auto &nhood = nhoods[n];
                    Neighbors &pool = nhood.pool;
                    kgraph::GenRandom(rng, &nhood.nn_new[0], nhood.nn_new.size(), N);
                    kgraph::GenRandom(rng, &random[0], random.size(), N);
                    nhood.L = params.S;
                    nhood.M = params.S;
                    unsigned i = 0;
                    for (unsigned l = 0; l < nhood.L; ++l)
                    {
                        if (random[i] == n)
                            ++i;
                        auto &nn = nhood.pool[l];
                        nn.id = random[i++];
                        nn.dist = oracle(nn.id, n);
                        nn.flag = true;
                    }
                    sort(pool.begin(), pool.begin() + nhood.L);
                }
            }
        }

        void join()
        {
            size_t cc = 0;
#pragma omp parallel for default(shared) schedule(dynamic, 100) reduction(+ : cc)
            for (unsigned n = 0; n < oracle.size(); ++n)
            {
                size_t uu = 0;
                nhoods[n].found = false;
                nhoods[n].join([&](unsigned i, unsigned j)
                               {
                        float dist = oracle(i, j);
                        ++cc;
                        unsigned r;
                        r = nhoods[i].parallel_try_insert(j, dist);
                        if (r < params.K) ++uu;
                        r = nhoods[j].parallel_try_insert(i, dist);
                        if (r < params.K) ++uu; });
                nhoods[n].found = uu > 0;
            }
            n_comps += cc;
        }

        void update()
        {
            unsigned N = oracle.size();
            for (auto &nhood : nhoods)
            {
                nhood.nn_new.clear();
                nhood.nn_old.clear();
                nhood.rnn_new.clear();
                nhood.rnn_old.clear();
                nhood.radius = nhood.pool.back().dist;
            }
            //!!! compute radius2
#pragma omp parallel for
            for (unsigned n = 0; n < N; ++n)
            {
                auto &nhood = nhoods[n];
                if (nhood.found)
                {
                    unsigned maxl = std::min(nhood.M + params.S, nhood.L);
                    unsigned c = 0;
                    unsigned l = 0;
                    while ((l < maxl) && (c < params.S))
                    {
                        if (nhood.pool[l].flag)
                            ++c;
                        ++l;
                    }
                    nhood.M = l;
                }
                assert(nhood.M > 0);
                nhood.radiusM = nhood.pool[nhood.M - 1].dist;
            }
#pragma omp parallel for
            for (unsigned n = 0; n < N; ++n)
            {
                auto &nhood = nhoods[n];
                auto &nn_new = nhood.nn_new;
                auto &nn_old = nhood.nn_old;
                for (unsigned l = 0; l < nhood.M; ++l)
                {
                    auto &nn = nhood.pool[l];
                    auto &nhood_o = nhoods[nn.id]; // nhood on the other side of the edge
                    if (nn.flag)
                    {
                        nn_new.push_back(nn.id);
                        if (nn.dist > nhood_o.radiusM)
                        {
                            LockGuard guard(nhood_o.lock);
                            nhood_o.rnn_new.push_back(n);
                        }
                        nn.flag = false;
                    }
                    else
                    {
                        nn_old.push_back(nn.id);
                        if (nn.dist > nhood_o.radiusM)
                        {
                            LockGuard guard(nhood_o.lock);
                            nhood_o.rnn_old.push_back(n);
                        }
                    }
                }
            }
            for (unsigned i = 0; i < N; ++i)
            {
                auto &nn_new = nhoods[i].nn_new;
                auto &nn_old = nhoods[i].nn_old;
                auto &rnn_new = nhoods[i].rnn_new;
                auto &rnn_old = nhoods[i].rnn_old;
                if (params.R && (rnn_new.size() > params.R))
                {
                    random_shuffle(rnn_new.begin(), rnn_new.end());
                    rnn_new.resize(params.R);
                }
                nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
                if (params.R && (rnn_old.size() > params.R))
                {
                    random_shuffle(rnn_old.begin(), rnn_old.end());
                    rnn_old.resize(params.R);
                }
                nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
            }
        }

    public:
        IndexParams params;
        IndexInfo *pinfo;
        unsigned ep_ = 0;
        unsigned loopcount = 0;
        vector<bool> pruneAgain;

        void init_nhoods()
        {
            unsigned nd_ = oracle.size();
            unsigned range = params.nsg_R;
            for (unsigned i = 0; i < nd_; ++i)
            {
                auto &nhood = nhoods[i];
                std::vector<unsigned>().swap(nhood.nn_new);
                std::vector<unsigned>().swap(nhood.nn_old);
                std::vector<unsigned>().swap(nhood.rnn_new);
                std::vector<unsigned>().swap(nhood.rnn_old);
                nhood.nn_nsg.reserve(range / 2);
                nhood.pool.resize(MAXL + 1);
            }
        }

        void ComputeEp()
        {
            unsigned nd_ = oracle.size();
            if (nd_ == 1)
            {
                ep_ = 0;
                return;
            }
            float *center = nullptr;
            center = oracle.Calcenter();
            std::vector<Neighbor> pool;
            unsigned init_point = rand() % nd_;
            get_neighbors(center, init_point, params.search_L, pool);
            ep_ = pool[0].id;
        }

        void tree_grow()
        {
            cout << "Tree grow..." << endl;
            auto s_dfs = std::chrono::high_resolution_clock::now();
            unsigned nd_ = oracle.size();
            unsigned root = ep_;
            vector<bool> flags(nd_, false);
            unsigned unlinked_cnt = 0;
            unsigned start = 0;
            while (unlinked_cnt < nd_)
            {
                DFS(flags, root, unlinked_cnt);
                if (unlinked_cnt >= nd_)
                    break;
                findroot(flags, root, start);
            }
            auto e_dfs = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_dfs = e_dfs - s_dfs;
            double Time_dfs = diff_dfs.count();
            pinfo->tree_grow_time.push_back(Time_dfs);
            std::cout << "Tree grow time: " << Time_dfs << "[s]" << std::endl;
        }

        void DFS(vector<bool> &flag, unsigned root, unsigned &cnt)
        {
            unsigned nd_ = oracle.size();
            unsigned tmp = root;
            std::stack<unsigned> s;
            s.push(root);
            if (!flag[root])
                cnt++;
            flag[root] = true;
            while (!s.empty())
            {
                unsigned next = nd_ + 1;
                for (unsigned i = 0; i < nhoods[tmp].nn_nsg.size(); i++)
                {
                    unsigned temp_next_id = nhoods[tmp].nn_nsg[i].id;
                    if (flag[temp_next_id] == false)
                    { //
                        next = temp_next_id;
                        break;
                    }
                }
                if (next == (nd_ + 1))
                {
                    s.pop();
                    if (s.empty())
                        break;
                    tmp = s.top();
                    continue;
                }
                tmp = next;
                flag[tmp] = true;
                s.push(tmp);
                cnt++;
            }
        }

        void findroot(vector<bool> &flag, unsigned &root, unsigned &start)
        {
            unsigned nd_ = oracle.size();
            unsigned id = nd_;
            for (unsigned i = start; i < nd_; i++)
            {
                if (flag[i] == false)
                {
                    id = i;
                    start = i;
                    break;
                }
            }
            if (id == nd_)
                return;
            unsigned found = 0;
            for (unsigned i = 0; i < nhoods[id].L; i++)
            {
                unsigned nn_id = nhoods[id].pool[i].id;
                if (flag[nn_id] && nhoods[nn_id].nn_nsg.size() < params.nsg_R)
                {
                    root = nn_id;
                    found = 1;
                    break;
                }
            }

            if (found == 0)
            {
                unsigned temp_count = 0;
                int factor = 1;
                while (true)
                {
                    temp_count++;
                    factor = int(temp_count / nd_) + 1;
                    unsigned rid = rand() % nd_;
                    if (flag[rid] && nhoods[rid].nn_nsg.size() < params.nsg_R * factor)
                    {
                        root = rid;
                        break;
                    }
                }
            }
            Neighbor temp_nn(id, oracle(root, id));
            nhoods[root].nn_nsg.push_back(temp_nn);
        }

        float pg_prune_normal(unsigned q)
        {
            unsigned nd_ = oracle.size();
            unsigned range = params.nsg_R;
            unsigned maxc = params.search_K;
            float tau_ = params.tau;
            float alpha = params.alpha;
            float threshold = std::cos(alpha / 180 * kPi);
            unsigned loop_i_ = params.loop_i;
            unsigned pg_type_ = params.pg_type;
            unsigned cc = 0;
            unsigned start = 0;
            auto &nhood = nhoods[q];
            nhood.nn_nsg.clear();
            if (nhood.L == 0)
            {
                return 0;
            }
            if (nhood.pool[start].id == q)
            {
                start++;
            }
            nhood.pool[start].pid = -1;
            nhood.nn_nsg.push_back(nhood.pool[start]);
            while (nhood.nn_nsg.size() < range && (++start) < nhood.L && start < maxc)
            {
                auto &p = nhood.pool[start];
                unsigned p_id = p.id;
                if (p_id >= nd_)
                {
                    continue;
                }
                bool occlude = false;
                for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                {
                    unsigned t_id = nhood.nn_nsg[t].id;
                    if (p_id == t_id)
                    {
                        occlude = true;
                        break;
                    }
                    cc++;
                    float djk = oracle(t_id, p_id);
                    bool check = false;
                    if (loopcount != loop_i_ || pg_type_ == INDEX_ALPHAPG)
                    {
                        float cos_ij = (nhood.nn_nsg[t].dist + djk - p.dist) / 2 /
                                       sqrt(nhood.nn_nsg[t].dist * djk);
                        check = djk < p.dist && cos_ij < threshold;
                    }
                    else if (pg_type_ == INDEX_TAUMNG)
                    {
                        check = djk < p.dist - 3 * tau_;
                    }
                    else
                    {
                        check = djk < p.dist;
                    }
                    if (check)
                    {
                        occlude = true;
                        p.pid = t;
                        break;
                    }
                }
                if (!occlude)
                {
                    p.pid = -1;
                    nhood.nn_nsg.push_back(p);
                }
            }
            while (start < nhood.L)
            {
                auto &p = nhood.pool[start];
                p.pid = -1;
                start++;
            }
            for (unsigned i = 0; i < nhood.nn_nsg.size(); i++)
            {
                nhood.nn_nsg[i].flag = false;
            }
            return float(cc) / float(nd_ * float(nd_ - 1) / 2);
        }

        float pg_prune_fast(unsigned q)
        {
            unsigned nd_ = oracle.size();
            unsigned range = params.nsg_R;
            unsigned maxc = params.search_K;
            float tau_ = params.tau;
            float alpha = params.alpha;
            float threshold = std::cos(alpha / 180 * kPi);
            unsigned loop_i_ = params.loop_i;
            unsigned pg_type_ = params.pg_type;
            unsigned cc = 0;
            unsigned start = 0;
            auto &nhood = nhoods[q];
            vector<unsigned> temp_nsg;
            unsigned pass = 0;
            for (unsigned i = 0; i < nhood.nn_nsg.size(); i++)
            {
                if (!nhood.nn_nsg[i].flag)
                {
                    temp_nsg.push_back(nhood.nn_nsg[i].id);
                }
            }
            nhood.nn_nsg.clear();
            if (nhood.L == 0)
            {
                return 0;
            }
            if (nhood.pool[start].id == q)
            {
                start++;
            }
            nhood.pool[start].pid = -1;
            nhood.nn_nsg.push_back(nhood.pool[start]);
            if (find(temp_nsg.begin(), temp_nsg.end(), nhood.nn_nsg.back().id) == temp_nsg.end())
            {
                nhood.nn_nsg.back().isnew = true;
            }
            else
            {
                nhood.nn_nsg.back().isnew = false;
            }
            while (nhood.nn_nsg.size() < range && (++start) < nhood.L && start < maxc)
            {
                auto &p = nhood.pool[start];
                unsigned p_id = p.id;
                if (p_id >= nd_)
                {
                    continue;
                }
                bool occlude = false;
                if (p.isnew)
                {
                    for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                    {
                        unsigned t_id = nhood.nn_nsg[t].id;
                        if (p_id == t_id)
                        {
                            occlude = true;
                            break;
                        }
                        cc++;
                        float djk = oracle(t_id, p_id);
                        bool check = false;
                        if (loopcount != loop_i_ || pg_type_ == INDEX_ALPHAPG)
                        {
                            float cos_ij = (nhood.nn_nsg[t].dist + djk - p.dist) / 2 /
                                           sqrt(nhood.nn_nsg[t].dist * djk);
                            check = djk < p.dist && cos_ij < threshold;
                        }
                        else if (pg_type_ == INDEX_TAUMNG)
                        {
                            check = djk < p.dist - 3 * tau_;
                        }
                        else
                        {
                            check = djk < p.dist;
                        }
                        if (check)
                        {
                            occlude = true;
                            p.pid = t;
                            break;
                        }
                    }
                    if (!occlude)
                    {
                        p.pid = -1;
                        nhood.nn_nsg.push_back(p);
                        nhood.nn_nsg.back().isnew = true;
                    }
                }
                else
                {
                    if (find(temp_nsg.begin(), temp_nsg.end(), p_id) == temp_nsg.end() && ((p.pid >= 0) && (p.pid < temp_nsg.size())))
                    {
                        unsigned prune_p = temp_nsg[p.pid];
                        auto iter = find_if(nhood.nn_nsg.begin(), nhood.nn_nsg.end(), [prune_p](Neighbor &neighbor)
                                            { return neighbor.id == prune_p; });
                        if (iter != nhood.nn_nsg.end())
                        {
                            pass++;
                            p.pid = distance(nhood.nn_nsg.begin(), iter);
                            continue;
                        }
                        else
                        {
                            for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                            {
                                unsigned t_id = nhood.nn_nsg[t].id;
                                if (p_id == t_id)
                                {
                                    occlude = true;
                                    break;
                                }
                                cc++;
                                float djk = oracle(t_id, p_id);
                                bool check = false;
                                if (loopcount != loop_i_ || pg_type_ == INDEX_ALPHAPG)
                                {
                                    float cos_ij = (nhood.nn_nsg[t].dist + djk - p.dist) / 2 /
                                                   sqrt(nhood.nn_nsg[t].dist * djk);
                                    check = djk < p.dist && cos_ij < threshold;
                                }
                                else if (pg_type_ == INDEX_TAUMNG)
                                {
                                    check = djk < p.dist - 3 * tau_;
                                }
                                else
                                {
                                    check = djk < p.dist;
                                }
                                if (check)
                                {
                                    occlude = true;
                                    p.pid = t;
                                    break;
                                }
                            }
                            if (!occlude)
                            {
                                p.pid = -1;
                                nhood.nn_nsg.push_back(p);
                                nhood.nn_nsg.back().isnew = true;
                            }
                        }
                    }
                    else if (find(temp_nsg.begin(), temp_nsg.end(), p_id) == temp_nsg.end() && ((p.pid < 0) || (p.pid >= temp_nsg.size())))
                    {
                        for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                        {
                            unsigned t_id = nhood.nn_nsg[t].id;
                            if (p_id == t_id)
                            {
                                occlude = true;
                                break;
                            }
                            cc++;
                            float djk = oracle(t_id, p_id);
                            bool check = false;
                            if (loopcount != loop_i_ || pg_type_ == INDEX_ALPHAPG)
                            {
                                float cos_ij = (nhood.nn_nsg[t].dist + djk - p.dist) / 2 /
                                               sqrt(nhood.nn_nsg[t].dist * djk);
                                check = djk < p.dist && cos_ij < threshold;
                            }
                            else if (pg_type_ == INDEX_TAUMNG)
                            {
                                check = djk < p.dist - 3 * tau_;
                            }
                            else
                            {
                                check = djk < p.dist;
                            }
                            if (check)
                            {
                                occlude = true;
                                p.pid = t;
                                break;
                            }
                        }
                        if (!occlude)
                        {
                            p.pid = -1;
                            nhood.nn_nsg.push_back(p);
                            nhood.nn_nsg.back().isnew = true;
                        }
                    }
                    else
                    {
                        for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                        {
                            unsigned t_id = nhood.nn_nsg[t].id;
                            if (p_id == t_id)
                            {
                                occlude = true;
                                break;
                            }
                            if (!nhood.nn_nsg[t].isnew)
                            {
                                continue;
                            }
                            cc++;
                            float djk = oracle(t_id, p_id);
                            bool check = false;
                            if (loopcount != loop_i_ || pg_type_ == INDEX_ALPHAPG)
                            {
                                float cos_ij = (nhood.nn_nsg[t].dist + djk - p.dist) / 2 /
                                               sqrt(nhood.nn_nsg[t].dist * djk);
                                check = djk < p.dist && cos_ij < threshold;
                            }
                            else if (pg_type_ == INDEX_TAUMNG)
                            {
                                check = djk < p.dist - 3 * tau_;
                            }
                            else
                            {
                                check = djk < p.dist;
                            }
                            if (check)
                            {
                                occlude = true;
                                p.pid = t;
                                break;
                            }
                        }
                        if (!occlude)
                        {
                            p.pid = -1;
                            nhood.nn_nsg.push_back(p);
                            nhood.nn_nsg.back().isnew = false;
                        }
                    }
                }
            }
            while (start < nhood.L)
            {
                auto &p = nhood.pool[start];
                p.pid = -1;
                start++;
            }
            for (unsigned i = 0; i < nhoods[q].nn_nsg.size(); i++)
            {
                nhoods[q].nn_nsg[i].flag = false;
            }
            return float(cc) / float(nd_ * float(nd_ - 1) / 2);
        }

        void add_reverse_edges(unsigned n, unsigned range, std::vector<Lock> &locks)
        {
            unsigned nd_ = oracle.size();
            for (size_t i = 0; i < nhoods[n].nn_nsg.size(); ++i)
            {
                size_t des = nhoods[n].nn_nsg[i].id;
                if (des >= nd_)
                {
                    continue;
                }
                int dup = 0;
                {
                    LockGuard guard(locks[des]);
                    for (size_t j = 0; j < nhoods[des].nn_nsg.size(); ++j)
                    {
                        if (n == nhoods[des].nn_nsg[j].id)
                        {
                            dup = 1;
                            break;
                        }
                    }
                }
                if (dup)
                {
                    continue;
                }
                Neighbor nn(n, oracle(n, des), true);
                {
                    LockGuard guard(locks[des]);
                    nhoods[des].nn_nsg.push_back(nn);
                }
            }
        }

        void reverse_prune(unsigned q)
        {
            unsigned range = params.nsg_R;
            float tau_ = params.tau;
            float alpha = params.alpha;
            float threshold = std::cos(alpha / 180 * kPi);
            unsigned loop_i_ = params.loop_i;
            unsigned pg_type_ = params.pg_type;
            auto &nhood = nhoods[q];
            auto temp_pool = nhood.nn_nsg;
            sort(temp_pool.begin(), temp_pool.end());
            nhood.nn_nsg.clear();
            unsigned start = 0;
            if (temp_pool.size() == 0)
            {
                return;
            }
            if (temp_pool[start].id == q)
            {
                start++;
            }
            nhood.nn_nsg.push_back(temp_pool[start]);
            while (nhood.nn_nsg.size() < range && (++start) < temp_pool.size())
            {
                auto &p = temp_pool[start];
                unsigned p_id = p.id;
                bool occlude = false;
                for (unsigned t = 0; t < nhood.nn_nsg.size(); t++)
                {
                    unsigned t_id = nhood.nn_nsg[t].id;
                    if (p_id == t_id)
                    {
                        occlude = true;
                        break;
                    }
                    float djk = oracle(t_id, p_id);
                    bool check = false;
                    if (loopcount != loop_i_ || pg_type_ == INDEX_ALPHAPG)
                    {
                        float cos_ij = (nhood.nn_nsg[t].dist + djk - p.dist) / 2 /
                                       sqrt(nhood.nn_nsg[t].dist * djk);
                        check = djk < p.dist && cos_ij < threshold;
                    }
                    else if (pg_type_ == INDEX_TAUMNG)
                    {
                        check = djk < p.dist - 3 * tau_;
                    }
                    else
                    {
                        check = djk < p.dist;
                    }
                    if (check)
                    {
                        occlude = true;
                        break;
                    }
                }
                if (!occlude)
                {
                    nhood.nn_nsg.push_back(p);
                }
            }
        }

        void pg_link()
        {
            unsigned range = params.nsg_R;
            unsigned nd_ = oracle.size();
            unsigned loop_i_ = params.loop_i;
            unsigned pg_type_ = params.pg_type;

            std::cout << "Pruning..." << std::endl;
            auto s_prune = std::chrono::high_resolution_clock::now();
            float scan = 0;
#pragma omp parallel
            {
#pragma omp for schedule(dynamic, 100) reduction(+ : scan)
                for (unsigned n = 0; n < nd_; ++n)
                {
                    if (pruneAgain[n]||(loopcount == loop_i_ && pg_type_ == INDEX_TAUMNG))
                    {
                        scan += pg_prune_normal(n);
                    }
                    else
                    {
                        scan += pg_prune_fast(n);
                    }
                }
            }
            auto e_prune = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_prune = e_prune - s_prune;
            double time_prune = diff_prune.count();
            pinfo->prune_time.push_back(time_prune);
            pinfo->prune_cost.push_back(scan);
            std::cout << "Pruning time: " << time_prune << "[s]" << std::endl;

            std::cout << "Add reverse edges..." << std::endl;
            auto s_addedge = std::chrono::high_resolution_clock::now();
            std::vector<Lock> locks(nd_);
#pragma omp parallel for schedule(dynamic, 100)
            for (unsigned n = 0; n < nd_; ++n)
            {
                add_reverse_edges(n, range, locks);
            }
            std::fill(pruneAgain.begin(), pruneAgain.end(), false);
#pragma omp parallel for schedule(dynamic, 100)
            for (unsigned n = 0; n < nd_; ++n)
            {
                if (nhoods[n].nn_nsg.size() > range)
                {
                    pruneAgain[n] = true;
                    reverse_prune(n);
                }
            }
            auto e_addedge = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_addedge = e_addedge - s_addedge;
            double time_addedge = diff_addedge.count();
            pinfo->add_reverse_time.push_back(time_addedge);
            std::cout << "Add reverse edges time: " << time_addedge << "[s]" << std::endl;
        }

        void buildPG()
        {
            std::cout << "Refining KCNA..." << std::endl;
            auto s_refine = std::chrono::high_resolution_clock::now();
            unsigned pg_type_ = params.pg_type;
            unsigned loop_i_ = params.loop_i;
            pg_link();
            ComputeEp();
            cout << "Ep: " << ep_ << endl;
            if (pg_type_ != INDEX_HNSW && pg_type_ != INDEX_NSW)
            {
                tree_grow();
            }
            UpdateNhoods();
            auto e_refine = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_refine = e_refine - s_refine;
            double time_refine = diff_refine.count();
            pinfo->refinement_time.push_back(time_refine);
            std::cout << "Refining KCNA time: " << time_refine << "[s]" << std::endl;
        }

        unsigned BridgeView()
        {
            cout << "Bridge view search..." << endl;
            auto s_search = std::chrono::high_resolution_clock::now();
            float delta = 1.0;
            float scan_rate = 0;
            unsigned cnt = 0;
            bool flag = true;
            unsigned nd_ = oracle.size();
            unsigned S_ = params.massq_S;
            BridgeView_init();
            while (true)
            {
                cnt = BridgeView_update(true);
                delta = (float)cnt / nd_ / S_;
                if (delta < 0.5)
                {
                    cnt += BridgeView_update(false);
                    delta = (float)cnt / nd_ / S_;
                    flag = false;
                }
                scan_rate += BridgeView_join();
                if (!flag)
                    break;
            }
            BridgeView_clear();
            auto e_search = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_search = e_search - s_search;
            double time_search = diff_search.count();
            pinfo->search_time.push_back(time_search);
            pinfo->search_cost.push_back(scan_rate);
            cout << "Bridge view search time: " << time_search << "[s]" << endl;
            return 0;
        }
        void BridgeView_init()
        {
            unsigned nd_ = oracle.size();
            unsigned S_ = params.massq_S;
            unsigned cur_loop = loopcount - 1;
            unsigned cur_L = min(params.search_L + cur_loop * params.step, MAXL);
#pragma omp parallel for
            for (unsigned i = 0; i < nd_; i++)
            {
                auto &nhood = nhoods[i];
                if (cur_loop == 0)
                {
                    for (unsigned j = 0; j < nhood.L; j++)
                    {
                        nhood.pool[j].flag = true;
                        nhood.pool[j].isnew = false;
                    }
                }
                else
                {
                    for (unsigned j = 0; j < cur_L; j++)
                    {
                        auto isexpand = nhood.pool[j].flag;
                        nhood.pool[j].isnew = isexpand;
                        nhood.pool[j].flag = true;
                    }
                    for (unsigned j = cur_L; j < nhood.L; j++)
                    {
                        nhood.pool[j].flag = true;
                        nhood.pool[j].isnew = false;
                    }
                }
                nhood.radius = nhood.pool.back().dist;
                nhood.rnn_new.clear();
                nhood.rnn_new.resize(S_ / 2);
                nhood.rnn_new_flag.clear();
                nhood.rnn_new_flag.resize(S_);
            }
        }
        unsigned BridgeView_update(bool flag)
        {
            unsigned count = 0;
            unsigned nd_ = oracle.size();
            unsigned S_ = params.massq_S;
            unsigned L = min(params.search_L + (loopcount - 1) * params.step, MAXL);
            if (flag)
            {
#pragma omp parallel for
                for (unsigned i = 0; i < nd_; i++)
                {
                    auto &nhood = nhoods[i];
                    nhood.rnn_new.clear();
                    nhood.rnn_new_flag.clear();
                }
            }
#pragma omp parallel for reduction(+ : count)
            for (unsigned i = 0; i < nd_; i++)
            {
                auto &nhood = nhoods[i];
                unsigned cnt = 0;
                for (unsigned j = 0; j < L && j < nhood.L && (cnt < S_ || !flag); j++)
                {
                    if (nhood.pool[j].flag)
                    {

                        unsigned id = nhood.pool[j].id;
                        {
                            LockGuard guard(nhoods[id].lock);
                            nhoods[id].rnn_new.push_back(i);
                            nhoods[id].rnn_new_flag.push_back(nhood.pool[j].isnew);
                        }
                        nhood.pool[j].flag = false;
                        cnt++;
                    }
                }
                count += cnt;
            }
            return count;
        }
        float BridgeView_join()
        {
            unsigned cc = 0;
            unsigned nd_ = oracle.size();
            unsigned dim_ = oracle.dim();
#pragma omp parallel for default(shared) schedule(dynamic, 100) reduction(+ : cc)
            for (unsigned i = 0; i < nd_; i++)
            {
                unsigned q = nd_ - i - 1;
                auto &rnn_new = nhoods[q].rnn_new;
                auto &nhood = nhoods[q];
                auto &rnn_new_flag = nhoods[q].rnn_new_flag;
                auto &nn_nsg = nhoods[q].nn_nsg;
                if (rnn_new.empty())
                {
                    continue;
                }
                for (unsigned k = 0; k < rnn_new.size(); k++)
                {
                    unsigned id = rnn_new[k];
                    bool ifnew = rnn_new_flag[k];
                    for (unsigned j = 0; j < nn_nsg.size(); j++)
                    {
                        unsigned nsg_id = nn_nsg[j].id;
                        if (id == nsg_id)
                            continue;
                        if (ifnew == false && nn_nsg[j].isnew == false)
                            continue;
                        cc++;
                        float dist = oracle(id, nsg_id);
                        parallel_InsertIntoPool(nhoods[id].pool, nhoods[id].L, nsg_id, dist, nhoods[id].radius, nhoods[id].lock);
                    }
                }
            }
            return (float)cc / (nd_ * float(nd_ - 1) / 2);
        }

        unsigned BridgeView_clear()
        {
            unsigned nd_ = oracle.size();
#pragma omp parallel for
            for (unsigned i = 0; i < nd_; i++)
            {
                auto &nhood = nhoods[i];
                vector<unsigned>().swap(nhood.rnn_new);
            }
            return 0;
        }

        void get_neighbors(const float *query, unsigned init_point, unsigned L, std::vector<Neighbor> &pool)
        {
            unsigned nd_ = oracle.size();
            pool.resize(L + 1);
            std::vector<unsigned> init_ids(L);
            vector<bool> flags(nd_, false);
            L = 0;
            for (unsigned i = 0; i < init_ids.size() && i < nhoods[init_point].pool.size(); i++)
            {
                init_ids[i] = nhoods[init_point].pool[i].id;
                flags[init_ids[i]] = true;
                L++;
            }
            while (L < init_ids.size())
            {
                unsigned id = rand() % nd_;
                if (flags[id])
                {
                    continue;
                }
                init_ids[L] = id;
                L++;
                flags[id] = true;
            }
            L = 0;

            for (unsigned i = 0; i < init_ids.size(); i++)
            {
                unsigned id = init_ids[i];
                if (id >= nd_)
                {
                    continue;
                }
                float dist = oracle(query, id);
                pool[i] = Neighbor(id, dist, true);
                L++;
            }
            std::sort(pool.begin(), pool.begin() + L);
            int k = 0;
            while (k < (int)L)
            {
                int nk = L;
                if (pool[k].flag)
                {
                    pool[k].flag = false;
                    unsigned n = pool[k].id;
                    for (unsigned m = 0; m < nhoods[n].nn_nsg.size(); ++m)
                    {
                        unsigned id = nhoods[n].nn_nsg[m].id;
                        if (flags[id])
                        {
                            continue;
                        }
                        flags[id] = 1;
                        float dist = oracle(query, id);
                        if (dist >= pool[L - 1].dist)
                        {
                            continue;
                        }
                        Neighbor nn(id, dist, true);
                        int r = InsertIntoPool(pool.data(), L, nn);

                        if (L + 1 < pool.size())
                        {
                            ++L;
                        }
                        if (r < nk)
                        {
                            nk = r;
                        }
                    }
                }
                if (nk <= k)
                    k = nk;
                else
                    ++k;
            }
        }

        unsigned parallel_InsertIntoPool(vector<Neighbor> &pool, unsigned &L, unsigned id, float dist, float &radius, Lock &lock)
        {
            if (dist > radius)
            {
                return pool.size();
            }
            LockGuard guard(lock);
            unsigned l = InsertIntoPool(pool.data(), L, Neighbor(true, id, dist, true));
            if (l <= L)
            { // inserted
                if (L + 1 < pool.size())
                { // if l == L + 1, there's a duplicate
                    ++L;
                }
                else
                {
                    radius = pool[L - 1].dist;
                }
            }
            return l;
        }
        float CalculatePG()
        {
            cout << "Computing AOD..." << endl;
            unsigned nd_ = oracle.size();
            unsigned max = 0, min = 1000000000;
            unsigned total = 0;
            for (size_t i = 0; i < nd_; i++)
            {
                unsigned size = nhoods[i].nn_nsg.size();
                if (size > max)
                {
                    max = size;
                }
                if (size < min)
                {
                    min = size;
                }
                total += size;
            }
            float avg = static_cast<float>(total) / nd_;
            cout << "Degree statistics: max = " << max << ", min = " << min << ", avg = " << avg << endl;
            return avg;
        }

        void UpdateNhoods()
        {
            if (loopcount != 0)
            {
#pragma omp parallel for schedule(dynamic, 100)
                for (unsigned i = 0; i < oracle.size(); i++)
                {
                    auto &nhood = nhoods[i];
                    for (unsigned j = 0; j < nhood.nn_nsg.size(); j++)
                    {
                        unsigned id = nhood.nn_nsg[j].id;
                        if (find(nhood.old_nsg.begin(), nhood.old_nsg.end(), id) == nhood.old_nsg.end())
                        {
                            nhood.nn_nsg[j].isnew = true;
                        }
                        else
                        {
                            nhood.nn_nsg[j].isnew = false;
                        }
                    }
                }
            }
            else
            {
#pragma omp parallel for schedule(dynamic, 100)
                for (unsigned i = 0; i < oracle.size(); i++)
                {
                    auto &nhood = nhoods[i];
                    for (unsigned j = 0; j < nhood.nn_nsg.size(); j++)
                    {
                        nhood.nn_nsg[j].isnew = true;
                    }
                }
            }
#pragma omp parallel for schedule(dynamic, 100)
            for (unsigned i = 0; i < oracle.size(); i++)
            {
                auto &nhood = nhoods[i];
                nhood.old_nsg.clear();
                nhood.old_nsg.reserve(nhood.nn_nsg.size());
                for (unsigned j = 0; j < nhood.nn_nsg.size(); j++)
                {
                    unsigned id = nhood.nn_nsg[j].id;
                    nhood.old_nsg.emplace_back(id);
                }
            }
        }

        float evaluate()
        {
            auto s_computegt = std::chrono::high_resolution_clock::now();
            double recall = 0.0;
#pragma omp parallel for reduction(+ : recall)
            for (size_t i = 0; i < controls.size(); ++i)
            {
                auto &c = controls[i];
                recall += EvaluateRecall(nhoods[c.id].pool, nhoods[c.id].L, c.neighbors, MAXL);
            }
            float rec = recall / controls.size();
            cout << "KCNA recall@" << MAXL << ":  " << rec << endl;
            auto e_computegt = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_computegt = e_computegt - s_computegt;
            double Time_computegt = diff_computegt.count();
            cout << "Compute KCNA recall time: " << Time_computegt << "[s]" << endl;
            return rec;
        }

        void swapGraph(std::vector<std::vector<unsigned>> &graph)
        {
            unsigned nd_ = oracle.size();
            graph.resize(nd_);
            for (unsigned i = 0; i < nd_; ++i)
            {
                auto &nei = graph[i];
                unsigned size = 0;
                if (params.pg_type == INDEX_KNNG)
                {
                    auto &pool = nhoods[i].pool;
                    size = params.K;
                    nei.resize(size);
                    for (unsigned j = 0; j < size; ++j)
                    {
                        nei[j] = pool[j].id;
                    }
                }
                else
                {
                    auto &pool = nhoods[i].nn_nsg;
                    size = pool.size();
                    nei.resize(size);
                    for (unsigned j = 0; j < size; ++j)
                    {
                        nei[j] = pool[j].id;
                    }
                }
            }
        }

        KGraphConstructor(IndexOracle const &o, IndexParams const &p, IndexInfo *r)
            : oracle(o), params(p), pinfo(r), nhoods(o.size()), n_comps(0)
        {
            no_dist = false;
            auto s_kgraph_iter = std::chrono::high_resolution_clock::now();
            unsigned N = oracle.size();
            if (N <= params.controls)
            {
                cerr << "Warning: small dataset, shrinking control size to " << (N - 1) << "." << endl;
                params.controls = N - 1;
            }
            if (N <= params.L)
            {
                cerr << "Warning: small dataset, shrinking L to " << (N - 1) << "." << endl;
                params.L = N - 1;
            }
            if (N <= params.S)
            {
                cerr << "Warning: small dataset, shrinking S to " << (N - 1) << "." << endl;
                params.S = N - 1;
            }
            if (params.iterations < 0)
            {
                cout << params.iterations << endl;
                printf("The iterations must be larger than 0.\n");
                exit(-1);
            }
            if (N <= params.search_L)
            {
                cerr << "Warning: small dataset, shrinking search_L to " << (N - 1) << "." << endl;
                params.search_L = N - 1;
            }
            if (N <= params.search_K)
            {
                cerr << "Warning: small dataset, shrinking search_K to " << (N - 1) << "." << endl;
                params.search_K = N - 1;
            }
            if (N <= params.K)
            {
                cerr << "Warning: small dataset, shrinking K to " << (N - 1) << "." << endl;
                params.K = N - 1;
                params.loop_i = 0;
                params.S = N - 1;
                params.iterations = 0;
            }
            MAXL = 0;
            if (params.L > MAXL)
                MAXL = params.L;
            if (params.K > MAXL)
                MAXL = params.K;
            if (params.search_L > MAXL)
                MAXL = params.search_L;
            if (params.search_K > MAXL)
                MAXL = params.search_K;

            if (verbosity > 0)
                cout << "Generating control..." << endl;
            GenerateControl(oracle, params.controls, MAXL, &controls);
            if (verbosity > 0)
                cout << "Initializing..." << endl;

            cout << "K: " << params.K << endl;
            // initialize nhoods
            init();

            // iterate until converge
            float total = N * float(N - 1) / 2;
            IndexInfo info;
            info.stop_condition = IndexInfo::ITERATION;
            info.recall = 0;
            info.accuracy = numeric_limits<float>::max();
            info.cost = 0;
            info.iterations = 0;
            info.delta = 1.0;

            for (unsigned it = 0; it < params.iterations; ++it)
            {
                ++info.iterations;
                {
                    float M = 0;
                    for (auto const &nhood : nhoods)
                    {
                        M += nhood.M;
                    }
                    int nhoods_size = nhoods.size();
                    info.M = mymean(M, nhoods_size);
                }
                join();
                {
                    info.cost = n_comps / total;
                    float one_exact = 0, one_approx = 0, one_recall = 0, recall = 0, accuracy = 0;
                    float delta = 0;
                    for (auto const &nhood : nhoods)
                    {
                        delta += EvaluateDelta(nhood.pool, params.K);
                    }
                    for (auto const &c : controls)
                    {
                        one_approx += nhoods[c.id].pool[0].dist;
                        one_exact += c.neighbors[0].dist;
                        one_recall += EvaluateOneRecall(nhoods[c.id].pool, c.neighbors);
                        recall += EvaluateRecall(nhoods[c.id].pool, nhoods[c.id].L, c.neighbors, params.K);
                        accuracy += EvaluateAccuracy(nhoods[c.id].pool, c.neighbors);
                    }
                    int nhoods_size = nhoods.size();
                    int controls_size = controls.size();
                    info.recall = mymean(recall, controls_size);
                    info.accuracy = mymean(accuracy, controls_size);
                    info.delta = mymean(delta, nhoods_size);
                    auto e_kgraph_iter = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> diff_kgraph_iter = e_kgraph_iter - s_kgraph_iter;
                    double time_kgraph_iter = diff_kgraph_iter.count();
                    info.kgraph_recall.push_back(info.recall);
                    info.kgraph_time.push_back(time_kgraph_iter);
                    if (verbosity > 0)
                    {
                        cout << "iteration: " << info.iterations
                             << " recall: " << info.recall
                             << " accuracy: " << info.accuracy
                             << " cost: " << info.cost
                             << " M: " << info.M
                             << " delta: " << info.delta
                             << " time: " << time_kgraph_iter
                             << " one-recall: " << mymean(one_recall, controls_size)
                             << " one-ratio: " << mymean(one_approx, controls_size) / mymean(one_exact, controls_size)
                             << endl;
                    }
                }
                if (info.delta <= params.delta)
                {
                    info.stop_condition = IndexInfo::DELTA;
                    break;
                }
                if (info.recall >= params.recall)
                {
                    info.stop_condition = IndexInfo::RECALL;
                    break;
                }
                if (info.iterations >= params.iterations)
                {
                    break;
                }
                update();
            }
            if (pinfo)
            {
                *pinfo = info;
            }
        }
    };

    void KGraphImpl::build(IndexOracle const &oracle, IndexParams const &params, IndexInfo *info, vector<vector<unsigned>> &graph, unsigned &entry_point)
    {
        if (params.pg_type == kgraph::INDEX_KNNG)
        {
            KGraphConstructor con(oracle, params, info);
            con.swapGraph(graph);
            entry_point = con.ep_;
        }
        else
        {
            auto s = std::chrono::high_resolution_clock::now();

            KGraphConstructor con(oracle, params, info);

            con.pruneAgain.resize(oracle.size());
            std::fill(con.pruneAgain.begin(), con.pruneAgain.end(), true);

            con.init_nhoods();

            float kcna_recall = 0;
            con.loopcount = 0;
            unsigned pg_iteration = con.params.loop_i + 1;
            for (unsigned iter = 1; iter <= pg_iteration; iter++)
            {

                kcna_recall = con.evaluate();
                auto e_kcna = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff_kcna = e_kcna - s;
                double Time_kcna = diff_kcna.count();
                con.pinfo->kcna_recall.push_back(kcna_recall);
                con.pinfo->kcna_time.push_back(Time_kcna);

                if (kcna_recall >= con.params.recall)
                {
                    con.loopcount = con.params.loop_i;
                }

                con.buildPG();

                auto e = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = e - s;
                double Time = diff.count();
                con.pinfo->buildpg_time.push_back(Time);
                cout << "Construct time: " << Time << "[s]" << endl;

                if (iter == pg_iteration || con.loopcount == con.params.loop_i)
                {
                    float aod = con.CalculatePG();
                    con.pinfo->AOD = aod;
                    break;
                }
                con.loopcount++;
                con.BridgeView();
            }
            con.swapGraph(graph);
            entry_point = con.ep_;
        }
        return;
    }

    KGraph *KGraph::create()
    {
        return new KGraphImpl;
    }

    char const *KGraph::version()
    {
        return kgraph_version;
    }
}
