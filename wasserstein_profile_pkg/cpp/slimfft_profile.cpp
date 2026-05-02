#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fftw3.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace slimfft {

struct CompactInterval {
    int red_start = 0;
    int red_end = -1;
    int blue_start = 0;
    int blue_end = -1;
    double cost = 0.0;
    int size() const { return red_end - red_start + 1; }
};

struct ColoredPoint {
    double position = 0.0;
    char color = 'R';
    int uid = -1;
};

struct ProfileResult {
    std::vector<double> costs;
};

struct IntervalLifetime {
    int red_start = 0;
    int red_end = -1;
    int blue_start = 0;
    int blue_end = -1;
    int born_k = 0;
    int dead_k_exclusive = 0;
};

struct ProfileWithLifetimesResult {
    std::vector<double> costs;
    std::vector<IntervalLifetime> intervals;
};

static double ipow(double x, int p) {
    double r = 1.0;
    while (p > 0) {
        if (p & 1) r *= x;
        x *= x;
        p >>= 1;
    }
    return r;
}

static double abs_ipow(double x, int p) { return ipow(std::abs(x), p); }

static std::vector<double> make_binom_row(int p) {
    std::vector<double> b(p + 1, 0.0);
    b[0] = 1.0;
    for (int i = 1; i <= p; ++i)
        b[i] = b[i-1] * static_cast<double>(p - i + 1) / static_cast<double>(i);
    return b;
}

static std::vector<double> direct_convolution(const std::vector<double>& a,
                                              const std::vector<double>& b) {
    if (a.empty() || b.empty()) return {};
    std::vector<double> out(a.size() + b.size() - 1, 0.0);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j)
            out[i+j] += a[i] * b[j];
    return out;
}

static std::vector<double> fftw_convolution(const std::vector<double>& a,
                                            const std::vector<double>& b) {
    if (a.empty() || b.empty()) return {};
    const int out_len = static_cast<int>(a.size() + b.size() - 1);
    const int n = out_len;
    const int nc = n / 2 + 1;

    double* in_a = static_cast<double*>(fftw_malloc(sizeof(double) * n));
    double* in_b = static_cast<double*>(fftw_malloc(sizeof(double) * n));
    double* out_time = static_cast<double*>(fftw_malloc(sizeof(double) * n));
    fftw_complex* fa = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * nc));
    fftw_complex* fb = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * nc));
    fftw_complex* fc = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * nc));

    std::fill(in_a, in_a + n, 0.0);
    std::fill(in_b, in_b + n, 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) in_a[i] = a[i];
    for (std::size_t i = 0; i < b.size(); ++i) in_b[i] = b[i];

    fftw_plan pa = fftw_plan_dft_r2c_1d(n, in_a, fa, FFTW_ESTIMATE);
    fftw_plan pb = fftw_plan_dft_r2c_1d(n, in_b, fb, FFTW_ESTIMATE);
    fftw_plan pc = fftw_plan_dft_c2r_1d(n, fc, out_time, FFTW_ESTIMATE);

    fftw_execute(pa);
    fftw_execute(pb);

    for (int i = 0; i < nc; ++i) {
        const double ar = fa[i][0], ai = fa[i][1];
        const double br = fb[i][0], bi = fb[i][1];
        fc[i][0] = ar*br - ai*bi;
        fc[i][1] = ar*bi + ai*br;
    }

    fftw_execute(pc);

    std::vector<double> out(out_len);
    const double scale = 1.0 / static_cast<double>(n);
    for (int i = 0; i < out_len; ++i) {
        double v = out_time[i] * scale;
        out[i] = std::abs(v) < 1e-12 ? 0.0 : v;
    }

    fftw_destroy_plan(pa); fftw_destroy_plan(pb); fftw_destroy_plan(pc);
    fftw_free(in_a); fftw_free(in_b); fftw_free(out_time);
    fftw_free(fa); fftw_free(fb); fftw_free(fc);
    return out;
}

static std::vector<double> convolve(const std::vector<double>& a,
                                    const std::vector<double>& b,
                                    int direct_threshold = 2 << 10) {
    const int out_len = static_cast<int>(a.size() + b.size() - 1);
    if (out_len <= 0) return {};
    return out_len <= direct_threshold ? direct_convolution(a, b) : fftw_convolution(a, b);
}

static double shift_array_value(int shift_min, const std::vector<double>& values, int d) {
    const int idx = d - shift_min;
    if (idx < 0 || idx >= static_cast<int>(values.size())) return 0.0;
    return values[idx];
}

static std::pair<int, std::vector<double>> combine_shift_arrays(
        const std::vector<std::pair<int, const std::vector<double>*>>& parts) {
    bool any = false;
    int lo = 0, hi = -1;
    for (const auto& [start, vec] : parts) {
        if (!vec || vec->empty()) continue;
        const int vhi = start + static_cast<int>(vec->size()) - 1;
        if (!any) { lo = start; hi = vhi; any = true; }
        else      { lo = std::min(lo, start); hi = std::max(hi, vhi); }
    }
    if (!any) return {0, {}};

    std::vector<double> out(hi - lo + 1, 0.0);
    for (const auto& [start, vec] : parts) {
        if (!vec || vec->empty()) continue;
        const int off = start - lo;
        for (std::size_t i = 0; i < vec->size(); ++i)
            out[off + i] += (*vec)[i];
    }
    return {lo, std::move(out)};
}

static std::vector<double> pointwise_power(const std::vector<double>& vals, int p) {
    std::vector<double> out(vals.size(), 1.0);
    if (p == 0) return out;
    for (std::size_t i = 0; i < vals.size(); ++i) out[i] = ipow(vals[i], p);
    return out;
}

static std::vector<double> prefix_sums(const std::vector<double>& vals) {
    std::vector<double> pref(vals.size() + 1, 0.0);
    for (std::size_t i = 0; i < vals.size(); ++i)
        pref[i+1] = pref[i] + vals[i];
    return pref;
}

static double prefix_range_sum(const std::vector<double>& pref, int l, int r) {
    if (l > r) return 0.0;
    return pref[r+1] - pref[l];
}

static std::pair<int, std::vector<double>> build_cross_values_general(
        const std::vector<double>& red_forward,
        const std::vector<double>& blue_forward,
        int shift_min,
        int p,
        bool red_left_blue_right,
        const std::vector<double>& binom) {
    if (red_forward.empty() || blue_forward.empty()) return {0, {}};

    const int nr = static_cast<int>(red_forward.size());
    const int nb = static_cast<int>(blue_forward.size());
    const int len = nr + nb - 1;

    std::vector<double> red_rev(red_forward.rbegin(), red_forward.rend());
    std::vector<double> values(len, 0.0);

    for (int c = 1; c < p; ++c) {
        std::vector<double> a = pointwise_power(red_rev, c);
        std::vector<double> b = pointwise_power(blue_forward, p - c);
        std::vector<double> conv = convolve(a, b);
        const int sign_parity = red_left_blue_right ? c : (p - c);
        const double coeff = ((sign_parity & 1) ? -1.0 : 1.0) * binom[c];
        for (std::size_t i = 0; i < conv.size(); ++i)
            values[i] += coeff * conv[i];
    }

    const double coeff_blue = red_left_blue_right ? 1.0 : ((p & 1) ? -1.0 : 1.0);
    const double coeff_red  = red_left_blue_right ? ((p & 1) ? -1.0 : 1.0) : 1.0;

    std::vector<double> red_pref = prefix_sums(pointwise_power(red_rev, p));
    std::vector<double> blue_pref = prefix_sums(pointwise_power(blue_forward, p));

    for (int q = 0; q < len; ++q) {
        const int a_lo = std::max(0, q - (nb - 1)), a_hi = std::min(q, nr - 1);
        const int b_lo = std::max(0, q - (nr - 1)), b_hi = std::min(q, nb - 1);
        values[q] += coeff_red  * prefix_range_sum(red_pref,  a_lo, a_hi);
        values[q] += coeff_blue * prefix_range_sum(blue_pref, b_lo, b_hi);
    }

    for (double& v : values)
        if (std::abs(v) < 1e-10) v = 0.0;

    return {shift_min, std::move(values)};
}

class BalancedIntervalCostDataStructure {
public:
    BalancedIntervalCostDataStructure(std::vector<double> red, std::vector<double> blue, int p)
        : R_(std::move(red)), B_(std::move(blue)), p_(p), binom_(make_binom_row(p)) {
        build_merged_order();
        build_prefix_counts();
        root_ = build_node(0, static_cast<int>(all_points_.size()) - 1);
    }
    ~BalancedIntervalCostDataStructure() { destroy(root_); }

    const std::vector<ColoredPoint>& all_points() const { return all_points_; }

    CompactInterval match_interval_by_indices(int merged_left, int merged_right,
                                              int direct_pair_threshold = 2 << 10) const {
        const int red_count  = prefix_red_in_merged_[merged_right+1] - prefix_red_in_merged_[merged_left];
        const int blue_count = prefix_blue_in_merged_[merged_right+1] - prefix_blue_in_merged_[merged_left];
        const int red_start  = prefix_red_in_merged_[merged_left];
        const int blue_start = prefix_blue_in_merged_[merged_left];

        double cost;
        if (red_count <= direct_pair_threshold) {
            cost = 0.0;
            for (int a = 0; a < red_count; ++a)
                cost += abs_ipow(R_[red_start + a] - B_[blue_start + a], p_);
        } else {
            const int shift = blue_start - red_start;
            cost = query_cost(root_, merged_left, merged_right, shift);
            if (std::abs(cost) < 1e-10) cost = 0.0;
        }

        return CompactInterval{
            red_start, red_start + red_count - 1,
            blue_start, blue_start + blue_count - 1,
            cost
        };
    }

private:
    struct Node {
        int l = 0, r = 0, mid = 0;
        Node* left = nullptr;
        Node* right = nullptr;
        int red_start = -1, red_end = -1;
        int blue_start = -1, blue_end = -1;
        int total_shift_min = 0;
        std::vector<double> total_values;
        int cross_lr_shift_min = 0;
        std::vector<double> cross_lr_values;
        int cross_rl_shift_min = 0;
        std::vector<double> cross_rl_values;
    };

    std::vector<double> R_, B_;
    int p_;
    std::vector<double> binom_;
    std::vector<ColoredPoint> all_points_;
    std::vector<int> prefix_red_in_merged_;
    std::vector<int> prefix_blue_in_merged_;
    Node* root_ = nullptr;

    void build_merged_order() {
        std::vector<ColoredPoint> raw;
        raw.reserve(R_.size() + B_.size());
        int uid = 0;
        for (double x : R_) raw.push_back({x, 'R', uid++});
        for (double x : B_) raw.push_back({x, 'B', uid++});
        std::sort(raw.begin(), raw.end(), [](const ColoredPoint& a, const ColoredPoint& b) {
            if (a.position != b.position) return a.position < b.position;
            if (a.color != b.color) return a.color < b.color;
            return a.uid < b.uid;
        });
        all_points_.resize(raw.size());
        for (int i = 0; i < static_cast<int>(raw.size()); ++i) {
            raw[i].uid = i;
            all_points_[i] = raw[i];
        }
    }

    void build_prefix_counts() {
        const int m = static_cast<int>(all_points_.size());
        prefix_red_in_merged_.assign(m + 1, 0);
        prefix_blue_in_merged_.assign(m + 1, 0);
        for (int i = 0; i < m; ++i) {
            prefix_red_in_merged_[i+1]  = prefix_red_in_merged_[i]  + (all_points_[i].color == 'R');
            prefix_blue_in_merged_[i+1] = prefix_blue_in_merged_[i] + (all_points_[i].color == 'B');
        }
    }

    std::pair<int, int> range_color_indices(int l, int r, char color) const {
        const auto& prefix = (color == 'R') ? prefix_red_in_merged_ : prefix_blue_in_merged_;
        const int start = prefix[l];
        const int after_end = prefix[r+1];
        if (after_end == start) return {-1, -1};
        return {start, after_end - 1};
    }

    std::pair<int, std::vector<double>> build_cross(const Node* ln, const Node* rn,
                                                    bool red_left) const {
        const int u = red_left ? ln->red_start  : rn->red_start;
        const int v = red_left ? ln->red_end    : rn->red_end;
        const int s = red_left ? rn->blue_start : ln->blue_start;
        const int t = red_left ? rn->blue_end   : ln->blue_end;
        if (u == -1 || s == -1) return {0, {}};
        return build_cross_values_general(
            {R_.begin()+u, R_.begin()+v+1},
            {B_.begin()+s, B_.begin()+t+1},
            s - v, p_, red_left, binom_);
    }

    Node* build_node(int l, int r) {
        Node* node = new Node();
        node->l = l; node->r = r; node->mid = (l + r) / 2;
        std::tie(node->red_start, node->red_end)   = range_color_indices(l, r, 'R');
        std::tie(node->blue_start, node->blue_end) = range_color_indices(l, r, 'B');
        if (l == r) return node;

        node->left  = build_node(l, node->mid);
        node->right = build_node(node->mid + 1, r);

        auto [lr_shift, lr_vals] = build_cross(node->left, node->right, true);
        node->cross_lr_shift_min = lr_shift;
        node->cross_lr_values = std::move(lr_vals);

        auto [rl_shift, rl_vals] = build_cross(node->left, node->right, false);
        node->cross_rl_shift_min = rl_shift;
        node->cross_rl_values = std::move(rl_vals);

        auto [tot_shift, tot_vals] = combine_shift_arrays({
            {node->left->total_shift_min,  &node->left->total_values},
            {node->cross_lr_shift_min,     &node->cross_lr_values},
            {node->cross_rl_shift_min,     &node->cross_rl_values},
            {node->right->total_shift_min, &node->right->total_values},
        });
        node->total_shift_min = tot_shift;
        node->total_values = std::move(tot_vals);
        return node;
    }

    static void destroy(Node* node) {
        if (!node) return;
        destroy(node->left);
        destroy(node->right);
        delete node;
    }

    double query_cost(const Node* node, int ql, int qr, int shift) const {
        if (ql <= node->l && node->r <= qr)
            return shift_array_value(node->total_shift_min, node->total_values, shift);
        if (!node->left) return 0.0;
        if (qr <= node->mid) return query_cost(node->left,  ql, qr, shift);
        if (ql >  node->mid) return query_cost(node->right, ql, qr, shift);
        return shift_array_value(node->cross_lr_shift_min, node->cross_lr_values, shift)
             + shift_array_value(node->cross_rl_shift_min, node->cross_rl_values, shift)
             + query_cost(node->left,  ql, qr, shift)
             + query_cost(node->right, ql, qr, shift);
    }
};

class DynamicIntervalSetSummary {
public:
    struct ApplyWithIdsResult {
        double swallowed_cost = 0.0;
        int swallowed_pairs = 0;
        std::vector<int> swallowed_ids;
    };

    DynamicIntervalSetSummary() : root_(nullptr), total_cost_(0.0), total_size_(0), rng_(123456789) {}
    ~DynamicIntervalSetSummary() { destroy(root_); }

    double total_cost() const { return total_cost_; }
    int total_size() const { return total_size_; }

    std::pair<double, int> range_summary(int red_start, int red_end) {
        auto [a, bc] = split(root_, red_start);
        auto [b, c] = split(bc, red_end + 1);
        const double sc = subtree_cost(b);
        const int sp = subtree_pairs(b);
        root_ = merge(a, merge(b, c));
        return {sc, sp};
    }

    std::pair<double, int> apply_interval(const CompactInterval& iv) {
        auto [a, bc] = split(root_, iv.red_start);
        auto [b, c] = split(bc, iv.red_end + 1);
        const double sc = subtree_cost(b);
        const int sp = subtree_pairs(b);
        destroy(b);
        Node* nn = new Node(iv, -1, static_cast<std::uint32_t>(rng_()));
        root_ = merge(a, merge(nn, c));
        total_cost_ += iv.cost - sc;
        total_size_ += iv.size() - sp;
        return {sc, sp};
    }

    ApplyWithIdsResult apply_interval_with_ids(const CompactInterval& iv, int new_id) {
        auto [a, bc] = split(root_, iv.red_start);
        auto [b, c] = split(bc, iv.red_end + 1);
        ApplyWithIdsResult res;
        res.swallowed_cost = subtree_cost(b);
        res.swallowed_pairs = subtree_pairs(b);
        collect_interval_ids(b, res.swallowed_ids);
        destroy(b);
        Node* nn = new Node(iv, new_id, static_cast<std::uint32_t>(rng_()));
        root_ = merge(a, merge(nn, c));
        total_cost_ += iv.cost - res.swallowed_cost;
        total_size_ += iv.size() - res.swallowed_pairs;
        return res;
    }

private:
    struct Node {
        CompactInterval interval;
        int interval_id;
        std::uint32_t prio;
        Node* left;
        Node* right;
        double subtree_cost_sum;
        int subtree_pair_sum;
        int subtree_max_red_end;

        Node(const CompactInterval& iv, int id, std::uint32_t p)
            : interval(iv), interval_id(id), prio(p),
              left(nullptr), right(nullptr),
              subtree_cost_sum(iv.cost),
              subtree_pair_sum(iv.size()),
              subtree_max_red_end(iv.red_end) {}
    };

    Node* root_;
    double total_cost_;
    int total_size_;
    std::mt19937 rng_;

    static double subtree_cost(Node* n)  { return n ? n->subtree_cost_sum    : 0.0; }
    static int    subtree_pairs(Node* n) { return n ? n->subtree_pair_sum    : 0; }
    static int    max_red_end(Node* n)   { return n ? n->subtree_max_red_end : std::numeric_limits<int>::min(); }

    static void pull(Node* n) {
        if (!n) return;
        n->subtree_cost_sum    = n->interval.cost   + subtree_cost(n->left)  + subtree_cost(n->right);
        n->subtree_pair_sum    = n->interval.size()  + subtree_pairs(n->left) + subtree_pairs(n->right);
        n->subtree_max_red_end = std::max({n->interval.red_end, max_red_end(n->left), max_red_end(n->right)});
    }

    static Node* merge(Node* a, Node* b) {
        if (!a) return b;
        if (!b) return a;
        if (a->prio < b->prio) { a->right = merge(a->right, b); pull(a); return a; }
        else                   { b->left  = merge(a, b->left);  pull(b); return b; }
    }

    static std::pair<Node*, Node*> split(Node* root, int key) {
        if (!root) return {nullptr, nullptr};
        if (root->interval.red_start < key) {
            auto [a, b] = split(root->right, key);
            root->right = a; pull(root);
            return {root, b};
        } else {
            auto [a, b] = split(root->left, key);
            root->left = b; pull(root);
            return {a, root};
        }
    }

    static void collect_interval_ids(Node* n, std::vector<int>& out) {
        if (!n) return;
        collect_interval_ids(n->left, out);
        out.push_back(n->interval_id);
        collect_interval_ids(n->right, out);
    }

    static void destroy(Node* n) {
        if (!n) return;
        destroy(n->left);
        destroy(n->right);
        delete n;
    }
};

struct LightCandidate {
    int left_uid = -1;
    int right_uid = -1;
    CompactInterval merged_interval;
    double delta_cost = 0.0;
    double swallowed_cost = 0.0;
    int swallowed_pairs = 0;
};

static bool candidate_valid(const LightCandidate& c,
                            const std::vector<unsigned char>& alive,
                            const std::vector<int>& prev_idx,
                            const std::vector<int>& next_idx) {
    return c.left_uid  >= 0 && c.left_uid  < static_cast<int>(alive.size()) &&
           c.right_uid >= 0 && c.right_uid < static_cast<int>(alive.size()) &&
           alive[c.left_uid] && alive[c.right_uid] &&
           next_idx[c.left_uid] == c.right_uid &&
           prev_idx[c.right_uid] == c.left_uid;
}

static LightCandidate evaluate_candidate(const BalancedIntervalCostDataStructure& qds,
                                         DynamicIntervalSetSummary& ivset,
                                         const ColoredPoint& lp,
                                         const ColoredPoint& rp) {
    CompactInterval merged = qds.match_interval_by_indices(lp.uid, rp.uid);
    auto [sc, sp] = ivset.range_summary(merged.red_start, merged.red_end);
    return LightCandidate{lp.uid, rp.uid, merged, merged.cost - sc, sc, sp};
}

static LightCandidate refresh_candidate(const LightCandidate& c,
                                        DynamicIntervalSetSummary& ivset) {
    auto [sc, sp] = ivset.range_summary(c.merged_interval.red_start, c.merged_interval.red_end);
    LightCandidate out = c;
    out.swallowed_cost = sc;
    out.swallowed_pairs = sp;
    out.delta_cost = c.merged_interval.cost - sc;
    return out;
}

static bool same_priority(const LightCandidate& a, const LightCandidate& b, double atol = 1e-12) {
    return std::abs(a.delta_cost - b.delta_cost) <= atol &&
           a.left_uid == b.left_uid &&
           a.right_uid == b.right_uid &&
           a.swallowed_pairs == b.swallowed_pairs;
}

struct CandidateCompare {
    bool operator()(const LightCandidate& a, const LightCandidate& b) const {
        if (a.delta_cost != b.delta_cost) return a.delta_cost > b.delta_cost;
        if (a.left_uid != b.left_uid) return a.left_uid > b.left_uid;
        return a.right_uid > b.right_uid;
    }
};

using CandidateHeap = std::priority_queue<LightCandidate, std::vector<LightCandidate>, CandidateCompare>;

static void push_candidate(CandidateHeap& heap,
                           const std::vector<ColoredPoint>& pts,
                           DynamicIntervalSetSummary& ivset,
                           int left_id, int right_id,
                           const BalancedIntervalCostDataStructure& qds) {
    if (left_id < 0 || right_id < 0 ||
        left_id  >= static_cast<int>(pts.size()) ||
        right_id >= static_cast<int>(pts.size())) return;
    if (pts[left_id].color == pts[right_id].color) return;
    heap.push(evaluate_candidate(qds, ivset, pts[left_id], pts[right_id]));
}

static void validate_instance(const std::vector<double>& R, const std::vector<double>& B, int p) {
    if (R.size() != B.size())                throw std::invalid_argument("R and B must have equal size.");
    if (!std::is_sorted(R.begin(), R.end())) throw std::invalid_argument("R must be sorted.");
    if (!std::is_sorted(B.begin(), B.end())) throw std::invalid_argument("B must be sorted.");
    if (p < 1)                               throw std::invalid_argument("p must be a positive integer.");
}

ProfileResult compute_profile_p(const std::vector<double>& R,
                                const std::vector<double>& B,
                                int p) {
    validate_instance(R, B, p);
    BalancedIntervalCostDataStructure qds(R, B, p);
    const auto& pts = qds.all_points();
    const int m = static_cast<int>(pts.size());

    std::vector<int> prev_idx(m), next_idx(m);
    std::vector<unsigned char> alive(m, 1);
    for (int i = 0; i < m; ++i) {
        prev_idx[i] = i - 1;
        next_idx[i] = (i + 1 < m) ? i + 1 : -1;
    }

    DynamicIntervalSetSummary ivset;
    CandidateHeap heap;
    for (int i = 0; i + 1 < m; ++i)
        push_candidate(heap, pts, ivset, i, i + 1, qds);

    ProfileResult result;
    result.costs.reserve(R.size() + 1);
    result.costs.push_back(0.0);

    for (int k = 1; k <= static_cast<int>(R.size()); ++k) {
        LightCandidate chosen;
        while (true) {
            LightCandidate c = heap.top(); heap.pop();
            if (!candidate_valid(c, alive, prev_idx, next_idx)) continue;
            LightCandidate rc = refresh_candidate(c, ivset);
            if (same_priority(c, rc)) { chosen = rc; break; }
            heap.push(rc);
        }

        ivset.apply_interval(chosen.merged_interval);

        const int li = chosen.left_uid, ri = chosen.right_uid;
        const int ln = prev_idx[li], rn = next_idx[ri];
        alive[li] = alive[ri] = 0;
        if (ln != -1) next_idx[ln] = rn;
        if (rn != -1) prev_idx[rn] = ln;
        prev_idx[li] = next_idx[li] = prev_idx[ri] = next_idx[ri] = -1;
        push_candidate(heap, pts, ivset, ln, rn, qds);

        result.costs.push_back(ivset.total_cost());
    }
    return result;
}

ProfileWithLifetimesResult compute_profile_p_with_lifetimes(const std::vector<double>& R,
                                                            const std::vector<double>& B,
                                                            int p) {
    validate_instance(R, B, p);
    BalancedIntervalCostDataStructure qds(R, B, p);
    const auto& pts = qds.all_points();
    const int m = static_cast<int>(pts.size());

    std::vector<int> prev_idx(m), next_idx(m);
    std::vector<unsigned char> alive(m, 1);
    for (int i = 0; i < m; ++i) {
        prev_idx[i] = i - 1;
        next_idx[i] = (i + 1 < m) ? i + 1 : -1;
    }

    DynamicIntervalSetSummary ivset;
    CandidateHeap heap;
    for (int i = 0; i + 1 < m; ++i)
        push_candidate(heap, pts, ivset, i, i + 1, qds);

    ProfileWithLifetimesResult result;
    result.costs.reserve(R.size() + 1);
    result.costs.push_back(0.0);
    result.intervals.reserve(R.size());

    const int n = static_cast<int>(R.size());
    for (int k = 1; k <= n; ++k) {
        LightCandidate chosen;
        while (true) {
            LightCandidate c = heap.top(); heap.pop();
            if (!candidate_valid(c, alive, prev_idx, next_idx)) continue;
            LightCandidate rc = refresh_candidate(c, ivset);
            if (same_priority(c, rc)) { chosen = rc; break; }
            heap.push(rc);
        }

        const int new_id = static_cast<int>(result.intervals.size());
        auto res = ivset.apply_interval_with_ids(chosen.merged_interval, new_id);

        for (int old_id : res.swallowed_ids)
            if (old_id >= 0) result.intervals[old_id].dead_k_exclusive = k;

        result.intervals.push_back({
            chosen.merged_interval.red_start,
            chosen.merged_interval.red_end,
            chosen.merged_interval.blue_start,
            chosen.merged_interval.blue_end,
            k, n + 1
        });

        const int li = chosen.left_uid, ri = chosen.right_uid;
        const int ln = prev_idx[li], rn = next_idx[ri];
        alive[li] = alive[ri] = 0;
        if (ln != -1) next_idx[ln] = rn;
        if (rn != -1) prev_idx[rn] = ln;
        prev_idx[li] = next_idx[li] = prev_idx[ri] = next_idx[ri] = -1;
        push_candidate(heap, pts, ivset, ln, rn, qds);

        result.costs.push_back(ivset.total_cost());
    }
    return result;
}

ProfileResult compute_profile_squared(const std::vector<double>& R, const std::vector<double>& B) {
    return compute_profile_p(R, B, 2);
}
ProfileWithLifetimesResult compute_profile_squared_with_lifetimes(const std::vector<double>& R,
                                                                   const std::vector<double>& B) {
    return compute_profile_p_with_lifetimes(R, B, 2);
}

} // namespace slimfft

#ifndef SLIMFFT_BUILD_PYTHON
int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n;
    if (!(std::cin >> n)) { std::cerr << "Expected n on stdin.\n"; return 1; }

    std::vector<double> R(n), B(n);
    for (int i = 0; i < n; ++i) std::cin >> R[i];
    for (int i = 0; i < n; ++i) std::cin >> B[i];

    int p = 2;
    if (!(std::cin >> p)) std::cin.clear();

    slimfft::ProfileResult result = slimfft::compute_profile_p(R, B, p);
    std::cout << std::setprecision(17);
    for (std::size_t i = 0; i < result.costs.size(); ++i) {
        if (i) std::cout << ' ';
        std::cout << result.costs[i];
    }
    std::cout << '\n';
    return 0;
}
#endif