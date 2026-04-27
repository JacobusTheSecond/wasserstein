from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from compare import compare_all_algorithms


def _as_xyz(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"Expected a 2D array of shape (n, 3), got shape {points.shape}.")
    if points.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns, got shape {points.shape}.")
    return np.ascontiguousarray(points[:, :3], dtype=np.float64)


def _downsample_points(points: np.ndarray, target_n: int | None, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points of shape (n, 3+), got {points.shape}")
    points = np.ascontiguousarray(points[:, :3], dtype=np.float64)

    if target_n is not None and len(points) > target_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), size=target_n, replace=False)
        points = points[idx]

    return points


def _load_ascii_ply_vertices(path: Path) -> np.ndarray:
    """
    Load only the vertex block from an ASCII PLY file.
    This works for Stanford range-grid files like dragonStandRight_0.ply.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file")

        fmt = None
        vertex_count = None

        # Read header
        for line in f:
            s = line.strip()
            if s.startswith("format "):
                fmt = s
            elif s.startswith("element vertex "):
                vertex_count = int(s.split()[2])
            elif s == "end_header":
                break

        if fmt is None:
            raise ValueError(f"{path}: missing PLY format line")
        if "ascii" not in fmt:
            raise ValueError(
                f"{path}: this fallback loader only supports ASCII PLY files; found format '{fmt}'"
            )
        if vertex_count is None:
            raise ValueError(f"{path}: missing 'element vertex' in header")

        points = np.empty((vertex_count, 3), dtype=np.float64)

        for i in range(vertex_count):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: unexpected EOF while reading vertex block")
            parts = line.strip().split()
            if len(parts) < 3:
                raise ValueError(f"{path}: vertex line {i} has fewer than 3 coordinates")
            points[i, 0] = float(parts[0])
            points[i, 1] = float(parts[1])
            points[i, 2] = float(parts[2])

        return points


def load_3d_points(path: str | Path, *, target_n: int | None = None, seed: int = 0) -> np.ndarray:
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".npy":
        points = np.load(path)
        return _downsample_points(points, target_n, seed)

    if ext == ".npz":
        data = np.load(path)
        if "points" in data:
            points = data["points"]
        else:
            first_key = next(iter(data.keys()))
            points = data[first_key]
        return _downsample_points(points, target_n, seed)

    if ext in {".csv", ".txt", ".xyz"}:
        delimiter = "," if ext == ".csv" else None
        points = np.loadtxt(path, delimiter=delimiter)
        return _downsample_points(points, target_n, seed)

    if ext == ".ply":
        points = _load_ascii_ply_vertices(path)
        return _downsample_points(points, target_n, seed)

    raise ValueError(f"Unsupported file format: {path.suffix}")


def equalize_cardinality(
    red_points: np.ndarray,
    blue_points: np.ndarray,
    *,
    target_n: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make the two point sets have equal size by downsampling both to a common n.

    Your Wasserstein-profile code assumes |R| = |B|.
    """
    red_points = _as_xyz(red_points)
    blue_points = _as_xyz(blue_points)

    common_n = min(len(red_points), len(blue_points))
    if target_n is not None:
        common_n = min(common_n, target_n)

    if common_n <= 0:
        raise ValueError("At least one point set is empty.")

    rng = np.random.default_rng(seed)

    if len(red_points) > common_n:
        idx = rng.choice(len(red_points), size=common_n, replace=False)
        red_points = red_points[idx]

    if len(blue_points) > common_n:
        idx = rng.choice(len(blue_points), size=common_n, replace=False)
        blue_points = blue_points[idx]

    return np.ascontiguousarray(red_points), np.ascontiguousarray(blue_points)


def principal_axis_union(
    red_points: np.ndarray,
    blue_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    PCA on the union of red and blue points.

    Returns:
        mean: shape (3,)
        axis: shape (3,), unit vector for the first principal component
    """
    red_points = _as_xyz(red_points)
    blue_points = _as_xyz(blue_points)

    all_points = np.vstack([red_points, blue_points])
    mean = all_points.mean(axis=0)
    centered = all_points - mean

    cov = centered.T @ centered / max(1, len(all_points) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)

    return mean, axis


def project_onto_axis(
    points: np.ndarray,
    mean: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    points = _as_xyz(points)
    return np.ascontiguousarray((points - mean) @ axis, dtype=np.float64)


def wasserstein_profile_from_3d_points(
    red_points: np.ndarray,
    blue_points: np.ndarray,
    *,
    algo_names: Sequence[str] = ("slimFFT",),
    target_n: int | None = None,
    seed: int = 0,
):
    """
    Full pipeline:
      1. equalize cardinality
      2. PCA on the union
      3. project onto first PCA axis
      4. sort the 1D projections
      5. compute the 1D Wasserstein profile
    """
    red_points, blue_points = equalize_cardinality(
        red_points, blue_points, target_n=target_n, seed=seed
    )

    mean, axis = principal_axis_union(red_points, blue_points)

    red_1d = np.sort(project_onto_axis(red_points, mean, axis))
    blue_1d = np.sort(project_onto_axis(blue_points, mean, axis))

    profiles = compare_all_algorithms(
        red_1d.tolist(),
        blue_1d.tolist(),
        list(algo_names),
    )

    return {
        "mean": mean,
        "axis": axis,
        "red_3d": red_points,
        "blue_3d": blue_points,
        "red_1d": red_1d,
        "blue_1d": blue_1d,
        "profiles": profiles,
    }


def wasserstein_profile_from_3d_files(
    red_path: str | Path,
    blue_path: str | Path,
    *,
    algo_names: Sequence[str] = ("slimFFT",),
    target_n: int | None = 2048,
    seed: int = 0,
):
    red_points = load_3d_points(red_path, target_n=target_n, seed=seed)
    blue_points = load_3d_points(blue_path, target_n=target_n, seed=seed + 1)

    return wasserstein_profile_from_3d_points(
        red_points,
        blue_points,
        algo_names=algo_names,
        target_n=target_n,
        seed=seed,
    )