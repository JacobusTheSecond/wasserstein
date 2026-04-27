#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <stdexcept>

namespace py = pybind11;

namespace slimfft {

struct IntervalLifetime {
    int red_start;
    int red_end;
    int blue_start;
    int blue_end;
    int born_k;
    int dead_k_exclusive;
};

struct ProfileResult {
    std::vector<double> costs;
    struct AlgorithmStats {
        std::string algorithm_name;
        std::int64_t query_count;
        std::int64_t candidate_count;
        std::int64_t heap_pushes;
        std::int64_t heap_pops;
        double initialization_time_seconds;
        double query_data_structure_time_seconds;
        double candidate_processing_time_seconds;
        double selection_time_seconds;
        double solution_update_time_seconds;
        double total_time_seconds;
    } stats;
};

struct ProfileWithLifetimesResult {
    std::vector<double> costs;
    std::vector<IntervalLifetime> intervals;
};

ProfileResult compute_profile_squared(const std::vector<double>& R,
                                      const std::vector<double>& B);

ProfileWithLifetimesResult compute_profile_squared_with_lifetimes(
    const std::vector<double>& R,
    const std::vector<double>& B
);

} // namespace slimfft

static std::vector<double> profile_squared(
    py::array_t<double, py::array::c_style | py::array::forcecast> r,
    py::array_t<double, py::array::c_style | py::array::forcecast> b
) {
    auto rbuf = r.request();
    auto bbuf = b.request();

    if (rbuf.ndim != 1 || bbuf.ndim != 1) {
        throw std::runtime_error("R and B must be 1D float64 arrays.");
    }
    if (rbuf.shape[0] != bbuf.shape[0]) {
        throw std::runtime_error("R and B must have equal length.");
    }

    const std::size_t n = static_cast<std::size_t>(rbuf.shape[0]);
    const double* rptr = static_cast<const double*>(rbuf.ptr);
    const double* bptr = static_cast<const double*>(bbuf.ptr);

    std::vector<double> R(rptr, rptr + n);
    std::vector<double> B(bptr, bptr + n);

    for (std::size_t i = 1; i < n; ++i) {
        if (R[i - 1] > R[i]) {
            throw std::runtime_error("R must be sorted.");
        }
        if (B[i - 1] > B[i]) {
            throw std::runtime_error("B must be sorted.");
        }
    }

    return slimfft::compute_profile_squared(R, B).costs;
}

static py::tuple profile_squared_with_lifetimes(
    py::array_t<double, py::array::c_style | py::array::forcecast> r,
    py::array_t<double, py::array::c_style | py::array::forcecast> b
) {
    auto rbuf = r.request();
    auto bbuf = b.request();

    if (rbuf.ndim != 1 || bbuf.ndim != 1) {
        throw std::runtime_error("R and B must be 1D float64 arrays.");
    }
    if (rbuf.shape[0] != bbuf.shape[0]) {
        throw std::runtime_error("R and B must have equal length.");
    }

    const std::size_t n = static_cast<std::size_t>(rbuf.shape[0]);
    const double* rptr = static_cast<const double*>(rbuf.ptr);
    const double* bptr = static_cast<const double*>(bbuf.ptr);

    std::vector<double> R(rptr, rptr + n);
    std::vector<double> B(bptr, bptr + n);

    for (std::size_t i = 1; i < n; ++i) {
        if (R[i - 1] > R[i]) {
            throw std::runtime_error("R must be sorted.");
        }
        if (B[i - 1] > B[i]) {
            throw std::runtime_error("B must be sorted.");
        }
    }

    auto result = slimfft::compute_profile_squared_with_lifetimes(R, B);

    py::array_t<double> costs(result.costs.size());
    auto costs_mut = costs.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(result.costs.size()); ++i) {
        costs_mut(i) = result.costs[i];
    }

    py::array_t<long long> intervals(
        {static_cast<py::ssize_t>(result.intervals.size()), static_cast<py::ssize_t>(6)}
    );
    auto ints_mut = intervals.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(result.intervals.size()); ++i) {
        const auto& rec = result.intervals[i];
        ints_mut(i, 0) = rec.red_start;
        ints_mut(i, 1) = rec.red_end;
        ints_mut(i, 2) = rec.blue_start;
        ints_mut(i, 3) = rec.blue_end;
        ints_mut(i, 4) = rec.born_k;
        ints_mut(i, 5) = rec.dead_k_exclusive;
    }

    return py::make_tuple(costs, intervals);
}

PYBIND11_MODULE(_slimfft, m) {
    m.doc() = "C++ slimFFT profile extension";

    m.def("profile_squared",
          &profile_squared,
          py::arg("R"),
          py::arg("B"));

    m.def("profile_squared_with_lifetimes",
          &profile_squared_with_lifetimes,
          py::arg("R"),
          py::arg("B"));
}
