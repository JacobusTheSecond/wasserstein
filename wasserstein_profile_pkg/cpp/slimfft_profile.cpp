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

static void validate_instance(const std::vector<double>& R, const std::vector<double>& B) {
    if (R.size() != B.size()) {
        throw std::invalid_argument("R and B must have equal size.");
    }
    if (!std::is_sorted(R.begin(), R.end())) {
        throw std::invalid_argument("R must be sorted.");
    }
    if (!std::is_sorted(B.begin(), B.end())) {
        throw std::invalid_argument("B must be sorted.");
    }
}

static std::vector<double> direct_convolution(const std::vector<double>& a,
                                              const std::vector<double>& b) {
    if (a.empty() || b.empty()) return {};
    std::vector<double> out(a.size() + b.size() - 1, 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double ai = a[i];
        for (std::size_t j = 0; j < b.size(); ++j) {
            out[i + j] += ai * b[j];
        }
    }
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

    if (!in_a || !in_b || !out_time || !fa || !fb || !fc) {
        if (in_a) fftw_free(in_a);
        if (in_b) fftw_free(in_b);
        if (out_time) fftw_free(out_time);
        if (fa) fftw_free(fa);
        if (fb) fftw_free(fb);
        if (fc) fftw_free(fc);
        throw std::runtime_error("FFTW allocation failed.");
    }

    std::fill(in_a, in_a + n, 0.0);
    std::fill(in_b, in_b + n, 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) in_a[i] = a[i];
    for (std::size_t i = 0; i < b.size(); ++i) in_b[i] = b[i];

    fftw_plan pa = fftw_plan_dft_r2c_1d(n, in_a, fa, FFTW_ESTIMATE);
    fftw_plan pb = fftw_plan_dft_r2c_1d(n, in_b, fb, FFTW_ESTIMATE);
    fftw_plan pc = fftw_plan_dft_c2r_1d(n, fc, out_time, FFTW_ESTIMATE);

    if (!pa || !pb || !pc) {
        if (pa) fftw_destroy_plan(pa);
        if (pb) fftw_destroy_plan(pb);
        if (pc) fftw_destroy_plan(pc);
        fftw_free(in_a);
        fftw_free(in_b);
        fftw_free(out_time);
        fftw_free(fa);
        fftw_free(fb);
        fftw_free(fc);
        throw std::runtime_error("FFTW plan creation failed.");
    }

    fftw_execute(pa);
    fftw_execute(pb);

    for (int i = 0; i < nc; ++i) {
        const double ar = fa[i][0];
        const double ai = fa[i][1];
        const double br = fb[i][0];
        const double bi = fb[i][1];
        fc[i][0] = ar * br - ai * bi;
        fc[i][1] = ar * bi + ai * br;
    }

    fftw_execute(pc);

    std::vector<double> out(out_len);
    const double scale = 1.0 / static_cast<double>(n);
    for (int i = 0; i < out_len; ++i) {
        double v = out_time[i] * scale;
        if (std::abs(v) < 1e-12) v = 0.0;
        out[i] = v;
    }

    fftw_destroy_plan(pa);
    fftw_destroy_plan(pb);
    fftw_destroy_plan(pc);
    fftw_free(in_a);
    fftw_free(in_b);
    fftw_free(out_time);
    fftw_free(fa);
    fftw_free(fb);
    fftw_free(fc);

    return out;
}

static std::vector<double> real_convolution_hybrid(const std::vector<double>& a,
                                                   const std::vector<double>& b,
                                                   int direct_threshold = 2 << 10) {
    const int out_len = static_cast<int>(a.size() + b.size() - 1);
    if (out_len <= 0) return {};
    if (out_len <= direct_threshold) return direct_convolution(a, b);
    return fftw_convolution(a, b);
}

static double interval_squared_cost_direct(const std::vector<double>& R,
                                           const std::vector<double>& B,
                                           int red_start,
                                           int blue_start,
                                           int count) {
    double total = 0.0;
    for (int a = 0; a < count; ++a) {
        const double diff = R[red_start + a] - B[blue_start + a];
        total += diff * diff;
    }
    return total;
}

static double shift_array_value(int shift_min, const std::vector<double>& values, int d) {
    const int idx = d - shift_min;
    if (idx < 0 || idx >= static_cast<int>(values.size())) return 0.0;
    return values[idx];
}

static std::pair<int, std::vector<double>> combine_shift_arrays(
    const std::vector<std::pair<int, const std::vector<double>*>>& parts) {
    bool any = false;
    int min_shift = 0;
    int max_shift = -1;

    for (const auto& part : parts) {
        if (part.second == nullptr || part.second->empty()) continue;
        const int lo = part.first;
        const int hi = part.first + static_cast<int>(part.second->size()) - 1;
        if (!any) {
            min_shift = lo;
            max_shift = hi;
            any = true;
        } else {
            min_shift = std::min(min_shift, lo);
            max_shift = std::max(max_shift, hi);
        }
    }

    if (!any) return {0, {}};

    std::vector<double> out(max_shift - min_shift + 1, 0.0);
    for (const auto& part : parts) {
        if (part.second == nullptr || part.second->empty()) continue;
        const int start = part.first - min_shift;
        for (std::size_t i = 0; i < part.second->size(); ++i) {
            out[start + static_cast<int>(i)] += (*(part.second))[i];
        }
    }
    return {min_shift, std::move(out)};
}

class BalancedIntervalSquaredCostDataStructure {
public:
    explicit BalancedIntervalSquaredCostDataStructure(std::vector<double> red,
                                                      std::vector<double> blue)
        : R_(std::move(red)), B_(std::move(blue)) {
        validate_instance(R_, B_);
        build_merged_order();
        build_prefixes();
        root_ = build_node(0, static_cast<int>(all_points_.size()) - 1);
    }

    ~BalancedIntervalSquaredCostDataStructure() { destroy(root_); }

    const std::vector<ColoredPoint>& all_points() const { return all_points_; }

    CompactInterval match_interval_by_indices(int merged_left,
                                              int merged_right,
                                              int direct_pair_threshold = 2 << 10) const {
        const int red_count =
            prefix_red_in_merged_[merged_right + 1] - prefix_red_in_merged_[merged_left];
        const int blue_count =
            prefix_blue_in_merged_[merged_right + 1] - prefix_blue_in_merged_[merged_left];

        if (red_count != blue_count) {
            throw std::invalid_argument("Queried merged interval is not balanced.");
        }
        if (red_count == 0) {
            throw std::invalid_argument("Queried merged interval is empty.");
        }

        const int red_start = prefix_red_in_merged_[merged_left];
        const int blue_start = prefix_blue_in_merged_[merged_left];
        const int red_end = red_start + red_count - 1;
        const int blue_end = blue_start + blue_count - 1;

        if (red_count <= direct_pair_threshold) {
            const double cost =
                interval_squared_cost_direct(R_, B_, red_start, blue_start, red_count);
            return CompactInterval{red_start, red_end, blue_start, blue_end, cost};
        }

        const double sum_r2 = prefix_r2_[red_end + 1] - prefix_r2_[red_start];
        const double sum_b2 = prefix_b2_[blue_end + 1] - prefix_b2_[blue_start];
        const double prod_sum =
            query_product_sum(root_, merged_left, merged_right, blue_start - red_start);

        return CompactInterval{
            red_start,
            red_end,
            blue_start,
            blue_end,
            sum_r2 + sum_b2 - 2.0 * prod_sum
        };
    }

private:
    struct Node {
        int l = 0;
        int r = 0;
        int mid = 0;
        Node* left = nullptr;
        Node* right = nullptr;

        int red_start = -1;
        int red_end = -1;
        int blue_start = -1;
        int blue_end = -1;

        int total_shift_min = 0;
        std::vector<double> total_values;

        int cross_lr_shift_min = 0;
        std::vector<double> cross_lr_values;

        int cross_rl_shift_min = 0;
        std::vector<double> cross_rl_values;
    };

    std::vector<double> R_;
    std::vector<double> B_;
    std::vector<ColoredPoint> all_points_;
    std::vector<int> prefix_red_in_merged_;
    std::vector<int> prefix_blue_in_merged_;
    std::vector<double> prefix_r2_;
    std::vector<double> prefix_b2_;
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

    void build_prefixes() {
        const int m = static_cast<int>(all_points_.size());
        prefix_red_in_merged_.assign(m + 1, 0);
        prefix_blue_in_merged_.assign(m + 1, 0);

        for (int i = 0; i < m; ++i) {
            prefix_red_in_merged_[i + 1] =
                prefix_red_in_merged_[i] + (all_points_[i].color == 'R');
            prefix_blue_in_merged_[i + 1] =
                prefix_blue_in_merged_[i] + (all_points_[i].color == 'B');
        }

        prefix_r2_.assign(R_.size() + 1, 0.0);
        prefix_b2_.assign(B_.size() + 1, 0.0);

        for (std::size_t i = 0; i < R_.size(); ++i) {
            prefix_r2_[i + 1] = prefix_r2_[i] + R_[i] * R_[i];
        }
        for (std::size_t i = 0; i < B_.size(); ++i) {
            prefix_b2_[i + 1] = prefix_b2_[i] + B_[i] * B_[i];
        }
    }

    std::pair<int, int> range_color_indices(int l, int r, char color) const {
        const auto& prefix = (color == 'R') ? prefix_red_in_merged_ : prefix_blue_in_merged_;
        const int start = prefix[l];
        const int after_end = prefix[r + 1];
        if (after_end == start) return {-1, -1};
        return {start, after_end - 1};
    }

    std::pair<int, std::vector<double>>
    build_cross_left_red_right_blue(const Node* left, const Node* right) const {
        if (left->red_start == -1 || right->blue_start == -1) return {0, {}};

        const int u = left->red_start;
        const int v = left->red_end;
        const int s = right->blue_start;
        const int t = right->blue_end;

        std::vector<double> red_vals;
        red_vals.reserve(v - u + 1);
        for (int i = v; i >= u; --i) red_vals.push_back(R_[i]);

        std::vector<double> blue_vals(B_.begin() + s, B_.begin() + t + 1);
        std::vector<double> values = real_convolution_hybrid(red_vals, blue_vals);
        return {s - v, std::move(values)};
    }

    std::pair<int, std::vector<double>>
    build_cross_right_red_left_blue(const Node* left, const Node* right) const {
        if (right->red_start == -1 || left->blue_start == -1) return {0, {}};

        const int u = right->red_start;
        const int v = right->red_end;
        const int s = left->blue_start;
        const int t = left->blue_end;

        std::vector<double> red_vals;
        red_vals.reserve(v - u + 1);
        for (int i = v; i >= u; --i) red_vals.push_back(R_[i]);

        std::vector<double> blue_vals(B_.begin() + s, B_.begin() + t + 1);
        std::vector<double> values = real_convolution_hybrid(red_vals, blue_vals);
        return {s - v, std::move(values)};
    }

    Node* build_node(int l, int r) {
        Node* node = new Node();
        node->l = l;
        node->r = r;
        node->mid = (l + r) / 2;

        {
            auto [rs, re] = range_color_indices(l, r, 'R');
            node->red_start = rs;
            node->red_end = re;
        }
        {
            auto [bs, be] = range_color_indices(l, r, 'B');
            node->blue_start = bs;
            node->blue_end = be;
        }

        if (l == r) return node;

        node->left = build_node(l, node->mid);
        node->right = build_node(node->mid + 1, r);

        {
            auto [shift_min, vals] = build_cross_left_red_right_blue(node->left, node->right);
            node->cross_lr_shift_min = shift_min;
            node->cross_lr_values = std::move(vals);
        }
        {
            auto [shift_min, vals] = build_cross_right_red_left_blue(node->left, node->right);
            node->cross_rl_shift_min = shift_min;
            node->cross_rl_values = std::move(vals);
        }
        {
            std::vector<std::pair<int, const std::vector<double>*>> parts;
            parts.push_back({node->left->total_shift_min, &node->left->total_values});
            parts.push_back({node->cross_lr_shift_min, &node->cross_lr_values});
            parts.push_back({node->cross_rl_shift_min, &node->cross_rl_values});
            parts.push_back({node->right->total_shift_min, &node->right->total_values});

            auto [shift_min, vals] = combine_shift_arrays(parts);
            node->total_shift_min = shift_min;
            node->total_values = std::move(vals);
        }

        return node;
    }

    static void destroy(Node* node) {
        if (node == nullptr) return;
        destroy(node->left);
        destroy(node->right);
        delete node;
    }

    double query_product_sum(const Node* node, int ql, int qr, int shift) const {
        if (ql <= node->l && node->r <= qr) {
            return shift_array_value(node->total_shift_min, node->total_values, shift);
        }
        if (node->left == nullptr || node->right == nullptr) return 0.0;
        if (qr <= node->mid) return query_product_sum(node->left, ql, qr, shift);
        if (ql > node->mid) return query_product_sum(node->right, ql, qr, shift);

        return shift_array_value(node->cross_lr_shift_min, node->cross_lr_values, shift)
             + shift_array_value(node->cross_rl_shift_min, node->cross_rl_values, shift)
             + query_product_sum(node->left, ql, qr, shift)
             + query_product_sum(node->right, ql, qr, shift);
    }
};

class DynamicIntervalSetSummary {
public:
    struct ApplyWithIdsResult {
        double swallowed_cost = 0.0;
        int swallowed_pairs = 0;
        std::vector<int> swallowed_ids;
    };

    DynamicIntervalSetSummary()
        : root_(nullptr), total_cost_(0.0), total_size_(0), rng_(123456789) {}

    ~DynamicIntervalSetSummary() { destroy(root_); }

    double total_cost() const { return total_cost_; }
    int total_size() const { return total_size_; }

    std::pair<double, int> range_summary(int red_start, int red_end) {
        auto [a, bc] = split(root_, red_start);
        auto [b, c] = split(bc, red_end + 1);

        Node* pred = rightmost(a);
        if (pred != nullptr && pred->interval.red_end >= red_start) {
            root_ = merge(a, merge(b, c));
            throw std::runtime_error("Candidate partially overlaps existing interval on the left.");
        }
        if (b != nullptr && max_red_end(b) > red_end) {
            root_ = merge(a, merge(b, c));
            throw std::runtime_error("Candidate partially overlaps existing interval in the middle block.");
        }

        const double swallowed_cost = subtree_cost(b);
        const int swallowed_pairs = subtree_pairs(b);

        root_ = merge(a, merge(b, c));
        return {swallowed_cost, swallowed_pairs};
    }

    std::pair<double, int> apply_interval(const CompactInterval& merged_interval) {
        auto [a, bc] = split(root_, merged_interval.red_start);
        auto [b, c] = split(bc, merged_interval.red_end + 1);

        Node* pred = rightmost(a);
        if (pred != nullptr && pred->interval.red_end >= merged_interval.red_start) {
            root_ = merge(a, merge(b, c));
            throw std::runtime_error("Chosen interval partially overlaps existing interval on the left.");
        }
        if (b != nullptr && max_red_end(b) > merged_interval.red_end) {
            root_ = merge(a, merge(b, c));
            throw std::runtime_error("Chosen interval partially overlaps existing interval in the middle block.");
        }

        const double swallowed_cost = subtree_cost(b);
        const int swallowed_pairs = subtree_pairs(b);

        Node* new_node = new Node(merged_interval, -1, static_cast<std::uint32_t>(rng_()));
        destroy(b);
        root_ = merge(a, merge(new_node, c));

        total_cost_ = total_cost_ - swallowed_cost + merged_interval.cost;
        total_size_ = total_size_ - swallowed_pairs + merged_interval.size();
        return {swallowed_cost, swallowed_pairs};
    }

    ApplyWithIdsResult apply_interval_with_ids(const CompactInterval& merged_interval,
                                               int new_interval_id) {
        auto [a, bc] = split(root_, merged_interval.red_start);
        auto [b, c] = split(bc, merged_interval.red_end + 1);

        Node* pred = rightmost(a);
        if (pred != nullptr && pred->interval.red_end >= merged_interval.red_start) {
            root_ = merge(a, merge(b, c));
            throw std::runtime_error("Chosen interval partially overlaps existing interval on the left.");
        }
        if (b != nullptr && max_red_end(b) > merged_interval.red_end) {
            root_ = merge(a, merge(b, c));
            throw std::runtime_error("Chosen interval partially overlaps existing interval in the middle block.");
        }

        ApplyWithIdsResult res;
        res.swallowed_cost = subtree_cost(b);
        res.swallowed_pairs = subtree_pairs(b);
        collect_interval_ids(b, res.swallowed_ids);

        Node* new_node = new Node(merged_interval, new_interval_id,
                                  static_cast<std::uint32_t>(rng_()));
        destroy(b);
        root_ = merge(a, merge(new_node, c));

        total_cost_ = total_cost_ - res.swallowed_cost + merged_interval.cost;
        total_size_ = total_size_ - res.swallowed_pairs + merged_interval.size();
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

        Node(const CompactInterval& it, int id, std::uint32_t p)
            : interval(it),
              interval_id(id),
              prio(p),
              left(nullptr),
              right(nullptr),
              subtree_cost_sum(it.cost),
              subtree_pair_sum(it.size()),
              subtree_max_red_end(it.red_end) {}
    };

    Node* root_;
    double total_cost_;
    int total_size_;
    std::mt19937 rng_;

    static double subtree_cost(Node* node) { return node ? node->subtree_cost_sum : 0.0; }
    static int subtree_pairs(Node* node) { return node ? node->subtree_pair_sum : 0; }
    static int max_red_end(Node* node) {
        return node ? node->subtree_max_red_end : std::numeric_limits<int>::min();
    }

    static void pull(Node* node) {
        if (!node) return;
        node->subtree_cost_sum =
            node->interval.cost + subtree_cost(node->left) + subtree_cost(node->right);
        node->subtree_pair_sum =
            node->interval.size() + subtree_pairs(node->left) + subtree_pairs(node->right);
        node->subtree_max_red_end =
            std::max({node->interval.red_end, max_red_end(node->left), max_red_end(node->right)});
    }

    static Node* merge(Node* a, Node* b) {
        if (!a) return b;
        if (!b) return a;
        if (a->prio < b->prio) {
            a->right = merge(a->right, b);
            pull(a);
            return a;
        } else {
            b->left = merge(a, b->left);
            pull(b);
            return b;
        }
    }

    static std::pair<Node*, Node*> split(Node* root, int key) {
        if (!root) return {nullptr, nullptr};
        if (root->interval.red_start < key) {
            auto [a, b] = split(root->right, key);
            root->right = a;
            pull(root);
            return {root, b};
        } else {
            auto [a, b] = split(root->left, key);
            root->left = b;
            pull(root);
            return {a, root};
        }
    }

    static Node* rightmost(Node* node) {
        if (!node) return nullptr;
        while (node->right) node = node->right;
        return node;
    }

    static void collect_interval_ids(Node* node, std::vector<int>& out) {
        if (!node) return;
        collect_interval_ids(node->left, out);
        out.push_back(node->interval_id);
        collect_interval_ids(node->right, out);
    }

    static void destroy(Node* node) {
        if (!node) return;
        destroy(node->left);
        destroy(node->right);
        delete node;
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

static bool candidate_valid(const LightCandidate& candidate,
                            const std::vector<unsigned char>& alive,
                            const std::vector<int>& prev_idx,
                            const std::vector<int>& next_idx) {
    const int left_id = candidate.left_uid;
    const int right_id = candidate.right_uid;
    return 0 <= left_id && left_id < static_cast<int>(alive.size()) &&
           0 <= right_id && right_id < static_cast<int>(alive.size()) &&
           alive[left_id] && alive[right_id] &&
           next_idx[left_id] == right_id &&
           prev_idx[right_id] == left_id;
}

static LightCandidate evaluate_candidate(const BalancedIntervalSquaredCostDataStructure& query_ds,
                                         DynamicIntervalSetSummary& interval_set,
                                         const ColoredPoint& left_endpoint,
                                         const ColoredPoint& right_endpoint) {
    if (left_endpoint.position > right_endpoint.position) {
        throw std::invalid_argument("Candidate endpoints must be ordered.");
    }
    if (left_endpoint.color == right_endpoint.color) {
        throw std::invalid_argument("Candidate endpoints must have opposite colors.");
    }

    CompactInterval merged =
        query_ds.match_interval_by_indices(left_endpoint.uid, right_endpoint.uid);

    auto [swallowed_cost, swallowed_pairs] =
        interval_set.range_summary(merged.red_start, merged.red_end);

    if (merged.size() != swallowed_pairs + 1) {
        throw std::runtime_error("Unexpected candidate size.");
    }

    LightCandidate cand;
    cand.left_uid = left_endpoint.uid;
    cand.right_uid = right_endpoint.uid;
    cand.merged_interval = merged;
    cand.swallowed_cost = swallowed_cost;
    cand.swallowed_pairs = swallowed_pairs;
    cand.delta_cost = merged.cost - swallowed_cost;
    return cand;
}

static LightCandidate refresh_candidate(const LightCandidate& candidate,
                                        DynamicIntervalSetSummary& interval_set) {
    auto [swallowed_cost, swallowed_pairs] =
        interval_set.range_summary(candidate.merged_interval.red_start,
                                   candidate.merged_interval.red_end);

    if (candidate.merged_interval.size() != swallowed_pairs + 1) {
        throw std::runtime_error("Unexpected refreshed candidate size.");
    }

    LightCandidate out = candidate;
    out.swallowed_cost = swallowed_cost;
    out.swallowed_pairs = swallowed_pairs;
    out.delta_cost = candidate.merged_interval.cost - swallowed_cost;
    return out;
}

static bool same_priority(const LightCandidate& a,
                          const LightCandidate& b,
                          double atol = 1e-12) {
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

static bool push_candidate(std::priority_queue<LightCandidate,
                                               std::vector<LightCandidate>,
                                               CandidateCompare>& heap,
                           const std::vector<ColoredPoint>& all_points,
                           DynamicIntervalSetSummary& interval_set,
                           int left_id,
                           int right_id,
                           const BalancedIntervalSquaredCostDataStructure& query_ds) {
    if (left_id < 0 || right_id < 0) return false;
    if (left_id >= static_cast<int>(all_points.size()) ||
        right_id >= static_cast<int>(all_points.size())) {
        return false;
    }

    const ColoredPoint& left_point = all_points[left_id];
    const ColoredPoint& right_point = all_points[right_id];
    if (left_point.color == right_point.color) return false;

    heap.push(evaluate_candidate(query_ds, interval_set, left_point, right_point));
    return true;
}

ProfileResult compute_profile_squared(const std::vector<double>& R,
                                      const std::vector<double>& B) {
    validate_instance(R, B);

    BalancedIntervalSquaredCostDataStructure query_ds(R, B);
    const auto& all_points = query_ds.all_points();
    const int m_points = static_cast<int>(all_points.size());

    std::vector<int> prev_idx(m_points), next_idx(m_points);
    std::vector<unsigned char> alive(m_points, 1);
    for (int i = 0; i < m_points; ++i) {
        prev_idx[i] = i - 1;
        next_idx[i] = (i + 1 < m_points) ? i + 1 : -1;
    }

    DynamicIntervalSetSummary interval_set;
    std::priority_queue<LightCandidate,
                        std::vector<LightCandidate>,
                        CandidateCompare> heap;

    for (int left_id = 0; left_id + 1 < m_points; ++left_id) {
        push_candidate(heap, all_points, interval_set, left_id, left_id + 1, query_ds);
    }

    ProfileResult result;
    result.costs.reserve(R.size() + 1);
    result.costs.push_back(0.0);

    const int n = static_cast<int>(R.size());
    for (int k = 1; k <= n; ++k) {
        LightCandidate chosen;
        bool found = false;

        while (!heap.empty()) {
            LightCandidate candidate = heap.top();
            heap.pop();

            if (!candidate_valid(candidate, alive, prev_idx, next_idx)) continue;

            LightCandidate refreshed = refresh_candidate(candidate, interval_set);
            if (same_priority(candidate, refreshed)) {
                chosen = refreshed;
                found = true;
                break;
            }
            heap.push(refreshed);
        }

        if (!found) {
            throw std::runtime_error("Priority queue became empty before matching completed.");
        }

        auto [swallowed_cost, swallowed_pairs] =
            interval_set.apply_interval(chosen.merged_interval);

        if (std::abs(swallowed_cost - chosen.swallowed_cost) > 1e-9 ||
            swallowed_pairs != chosen.swallowed_pairs) {
            throw std::runtime_error(
                "Chosen candidate summary changed between refresh and application."
            );
        }

        const int left_id = chosen.left_uid;
        const int right_id = chosen.right_uid;
        const int left_neighbor = prev_idx[left_id];
        const int right_neighbor = next_idx[right_id];

        alive[left_id] = 0;
        alive[right_id] = 0;

        if (left_neighbor != -1) next_idx[left_neighbor] = right_neighbor;
        if (right_neighbor != -1) prev_idx[right_neighbor] = left_neighbor;

        prev_idx[left_id] = next_idx[left_id] = -1;
        prev_idx[right_id] = next_idx[right_id] = -1;

        push_candidate(heap, all_points, interval_set, left_neighbor, right_neighbor, query_ds);

        if (interval_set.total_size() != k) {
            throw std::runtime_error("Internal error: total_size mismatch.");
        }

        result.costs.push_back(interval_set.total_cost());
    }

    return result;
}

ProfileWithLifetimesResult compute_profile_squared_with_lifetimes(
    const std::vector<double>& R,
    const std::vector<double>& B) {
    validate_instance(R, B);

    BalancedIntervalSquaredCostDataStructure query_ds(R, B);
    const auto& all_points = query_ds.all_points();
    const int m_points = static_cast<int>(all_points.size());

    std::vector<int> prev_idx(m_points), next_idx(m_points);
    std::vector<unsigned char> alive(m_points, 1);
    for (int i = 0; i < m_points; ++i) {
        prev_idx[i] = i - 1;
        next_idx[i] = (i + 1 < m_points) ? i + 1 : -1;
    }

    DynamicIntervalSetSummary interval_set;
    std::priority_queue<LightCandidate,
                        std::vector<LightCandidate>,
                        CandidateCompare> heap;

    for (int left_id = 0; left_id + 1 < m_points; ++left_id) {
        push_candidate(heap, all_points, interval_set, left_id, left_id + 1, query_ds);
    }

    ProfileWithLifetimesResult result;
    result.costs.reserve(R.size() + 1);
    result.costs.push_back(0.0);
    result.intervals.reserve(R.size());

    const int n = static_cast<int>(R.size());
    for (int k = 1; k <= n; ++k) {
        LightCandidate chosen;
        bool found = false;

        while (!heap.empty()) {
            LightCandidate candidate = heap.top();
            heap.pop();

            if (!candidate_valid(candidate, alive, prev_idx, next_idx)) continue;

            LightCandidate refreshed = refresh_candidate(candidate, interval_set);
            if (same_priority(candidate, refreshed)) {
                chosen = refreshed;
                found = true;
                break;
            }
            heap.push(refreshed);
        }

        if (!found) {
            throw std::runtime_error("Priority queue became empty before matching completed.");
        }

        const int new_interval_id = static_cast<int>(result.intervals.size());
        auto apply_res =
            interval_set.apply_interval_with_ids(chosen.merged_interval, new_interval_id);

        if (std::abs(apply_res.swallowed_cost - chosen.swallowed_cost) > 1e-9 ||
            apply_res.swallowed_pairs != chosen.swallowed_pairs) {
            throw std::runtime_error(
                "Chosen candidate summary changed between refresh and application."
            );
        }

        for (int old_id : apply_res.swallowed_ids) {
            if (old_id >= 0) {
                result.intervals[old_id].dead_k_exclusive = k;
            }
        }

        IntervalLifetime rec;
        rec.red_start = chosen.merged_interval.red_start;
        rec.red_end = chosen.merged_interval.red_end;
        rec.blue_start = chosen.merged_interval.blue_start;
        rec.blue_end = chosen.merged_interval.blue_end;
        rec.born_k = k;
        rec.dead_k_exclusive = n + 1;
        result.intervals.push_back(rec);

        const int left_id = chosen.left_uid;
        const int right_id = chosen.right_uid;
        const int left_neighbor = prev_idx[left_id];
        const int right_neighbor = next_idx[right_id];

        alive[left_id] = 0;
        alive[right_id] = 0;

        if (left_neighbor != -1) next_idx[left_neighbor] = right_neighbor;
        if (right_neighbor != -1) prev_idx[right_neighbor] = left_neighbor;

        prev_idx[left_id] = next_idx[left_id] = -1;
        prev_idx[right_id] = next_idx[right_id] = -1;

        push_candidate(heap, all_points, interval_set, left_neighbor, right_neighbor, query_ds);

        if (interval_set.total_size() != k) {
            throw std::runtime_error("Internal error: total_size mismatch.");
        }

        result.costs.push_back(interval_set.total_cost());
    }

    return result;
}

} // namespace slimfft

#ifndef SLIMFFT_BUILD_PYTHON
int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n;
    if (!(std::cin >> n)) {
        std::cerr << "Expected n on stdin.\n";
        return 1;
    }

    std::vector<double> R(n), B(n);
    for (int i = 0; i < n; ++i) {
        if (!(std::cin >> R[i])) {
            std::cerr << "Expected R[" << i << "] on stdin.\n";
            return 1;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (!(std::cin >> B[i])) {
            std::cerr << "Expected B[" << i << "] on stdin.\n";
            return 1;
        }
    }

    try {
        slimfft::ProfileResult result = slimfft::compute_profile_squared(R, B);
        std::cout << std::setprecision(17);
        for (std::size_t i = 0; i < result.costs.size(); ++i) {
            if (i) std::cout << ' ';
            std::cout << result.costs[i];
        }
        std::cout << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
#endif