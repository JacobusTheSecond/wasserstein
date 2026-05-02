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
};

struct ProfileWithLifetimesResult {
    std::vector<double> costs;
    std::vector<IntervalLifetime> intervals;
};

ProfileResult compute_profile_p(const std::vector<double>& R,
                                const std::vector<double>& B,
                                int p);

ProfileWithLifetimesResult compute_profile_p_with_lifetimes(
    const std::vector<double>& R,
    const std::vector<double>& B,
    int p
);

ProfileResult compute_profile_squared(const std::vector<double>& R,
                                      const std::vector<double>& B);

ProfileWithLifetimesResult compute_profile_squared_with_lifetimes(
    const std::vector<double>& R,
    const std::vector<double>& B
);

} // namespace slimfft

static std::pair<std::vector<double>, std::vector<double>> parse_inputs(
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

    return {std::move(R), std::move(B)};
}

static py::array_t<double> costs_to_numpy(const std::vector<double>& costs_vec) {
    py::array_t<double> costs(costs_vec.size());
    auto costs_mut = costs.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(costs_vec.size()); ++i) {
        costs_mut(i) = costs_vec[i];
    }
    return costs;
}

static py::array_t<long long> intervals_to_numpy(
    const std::vector<slimfft::IntervalLifetime>& intervals_vec
) {
    py::array_t<long long> intervals(
        {static_cast<py::ssize_t>(intervals_vec.size()), static_cast<py::ssize_t>(6)}
    );
    auto ints_mut = intervals.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(intervals_vec.size()); ++i) {
        const auto& rec = intervals_vec[i];
        ints_mut(i, 0) = rec.red_start;
        ints_mut(i, 1) = rec.red_end;
        ints_mut(i, 2) = rec.blue_start;
        ints_mut(i, 3) = rec.blue_end;
        ints_mut(i, 4) = rec.born_k;
        ints_mut(i, 5) = rec.dead_k_exclusive;
    }

    return intervals;
}

static py::array_t<double> profile_p(
    py::array_t<double, py::array::c_style | py::array::forcecast> r,
    py::array_t<double, py::array::c_style | py::array::forcecast> b,
    int p
) {
    auto [R, B] = parse_inputs(std::move(r), std::move(b));
    auto result = slimfft::compute_profile_p(R, B, p);
    return costs_to_numpy(result.costs);
}

static py::tuple profile_p_with_lifetimes(
    py::array_t<double, py::array::c_style | py::array::forcecast> r,
    py::array_t<double, py::array::c_style | py::array::forcecast> b,
    int p
) {
    auto [R, B] = parse_inputs(std::move(r), std::move(b));
    auto result = slimfft::compute_profile_p_with_lifetimes(R, B, p);
    return py::make_tuple(
        costs_to_numpy(result.costs),
        intervals_to_numpy(result.intervals)
    );
}

static py::array_t<double> profile_squared(
    py::array_t<double, py::array::c_style | py::array::forcecast> r,
    py::array_t<double, py::array::c_style | py::array::forcecast> b
) {
    auto [R, B] = parse_inputs(std::move(r), std::move(b));
    auto result = slimfft::compute_profile_squared(R, B);
    return costs_to_numpy(result.costs);
}

static py::tuple profile_squared_with_lifetimes(
    py::array_t<double, py::array::c_style | py::array::forcecast> r,
    py::array_t<double, py::array::c_style | py::array::forcecast> b
) {
    auto [R, B] = parse_inputs(std::move(r), std::move(b));
    auto result = slimfft::compute_profile_squared_with_lifetimes(R, B);
    return py::make_tuple(
        costs_to_numpy(result.costs),
        intervals_to_numpy(result.intervals)
    );
}

PYBIND11_MODULE(_wasserstein_profile, m) {
    m.doc() = "C++ Wasserstein profile extension";

    m.def("profile_p",
          &profile_p,
          py::arg("R"),
          py::arg("B"),
          py::arg("p"));

    m.def("profile_p_with_lifetimes",
          &profile_p_with_lifetimes,
          py::arg("R"),
          py::arg("B"),
          py::arg("p"));

    m.def("profile_squared",
          &profile_squared,
          py::arg("R"),
          py::arg("B"));

    m.def("profile_squared_with_lifetimes",
          &profile_squared_with_lifetimes,
          py::arg("R"),
          py::arg("B"));
}