"""
Create dataset to add multibody correction to 2-body cross.

V_t = M_2(x_t, x_s) F_s + M_all(x_t, x_s, x_a, x_b....x_k) F_s

We're going to learn the correction term: M_all

To sample the neighborhood, we use a Cassini oval approach.
Metric d_kt*d_ks will be minimized, while d_kt < 6.0


Sphere-only.
"""

import time
import numpy as np

from mfs import imp_mfs_mobility_vec
from mfs_utils import build_B


def random_unit_vector() -> np.ndarray:
    """Return a random unit vector in R^3."""
    vec = np.random.randn(3)
    return vec / np.linalg.norm(vec)


def generate_dataset_spheroid_multibody(
    N_data: int,
    dist_min: float,
    dist_max: float,
    no_overlap_thresh: float = 0.05,
    max_neighbors: int = 10,
    num_candidates: int = 20,
):
    """Generate multibody correction samples for spheroids.

    Parameters
    ----------
    N_data : int
        Number of accepted samples to produce.
    dist_min, dist_max : float
        Biased sampling range for the primary source particle.
    no_overlap_thresh : float, optional
        Minimal separation allowed between any two particles (surface-to-surface).
    max_neighbors : int, optional
        Maximum number of additional neighbors (nk) placed around the target.
    num_candidates : int, optional
        Number of candidate locations sampled before probabilistic selection.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Feature vectors (ragged, dtype=object), targets (N x 6), and neighbor counts per sample.
    """

    if max_neighbors < 1:
        raise ValueError("max_neighbors must be at least 1")

    acc = "fine"
    boundary1 = np.loadtxt(
        f"/home/shihab/src/mfs/points/b_sphere_{acc}.txt", dtype=np.float64
    )
    source1 = np.loadtxt(
        f"/home/shihab/src/mfs/points/s_sphere_{acc}.txt", dtype=np.float64
    )
    print(
        f"Target: {boundary1.shape[0]} boundary nodes, {source1.shape[0]} source points"
    )

    B_orig = build_B(boundary1, source1, np.zeros(3))
    t0 = time.time()
    B1_inv = np.linalg.pinv(B_orig)
    print(f"B-matrix built in {time.time() - t0:.2f} seconds.")

    X_features: list[np.ndarray] = []
    y_targets: list[np.ndarray] = []
    neighbor_counts: list[int] = []

    n_accepted = 0
    rejected = 0
    total_trials = 0

    def biased_distance() -> float:
        L = dist_max - dist_min
        u = np.random.uniform(0.0, 1.0)
        x = 2.0 - np.sqrt(4.0 - 3.0 * u)  # inverse CDF for a linear PDF
        return dist_min + x * L

    while n_accepted < N_data:
        total_trials += 1

        dist_s = biased_distance()
        center_s = dist_s * random_unit_vector()

        min_dist_target_source = np.linalg.norm(center_s) - 2.0
        if min_dist_target_source <= no_overlap_thresh:
            rejected += 1
            continue

        scalar = 6 * np.pi * -1.0 * 1.0
        force_on_s = random_unit_vector() * scalar
        torque_on_s = random_unit_vector() * scalar * 1.0

        nk = int(np.random.randint(1, max_neighbors + 1))

        candidate_dirs = np.array([random_unit_vector() for _ in range(num_candidates)])
        min_target_radius = 2.0 + no_overlap_thresh + 0.05
        candidate_dist = np.random.uniform(min_target_radius, dist_max, size=num_candidates)
        candidate_centers = candidate_dirs * candidate_dist[:, None]

        d_kt = np.linalg.norm(candidate_centers, axis=1)
        assert np.allclose(d_kt, candidate_dist)

        d_ks = np.linalg.norm(candidate_centers - center_s.reshape(1, 3), axis=1)

        mask_s = d_ks - 2.0 > no_overlap_thresh
        cur_candidate_count = mask_s.sum()
        if cur_candidate_count < nk:
            rejected += 1
            continue

        d_kt = d_kt[mask_s]
        d_ks = d_ks[mask_s]
        candidate_centers = candidate_centers[mask_s]

        selection_metric = d_kt * d_ks
        weights = 1.0 / (selection_metric + 1e-9)
        weights_sum = weights.sum()
        if not np.isfinite(weights_sum) or weights_sum <= 0.0:
            rejected += 1
            continue
        weights /= weights_sum


        candidate_order = np.random.choice(
            cur_candidate_count, size=cur_candidate_count, replace=False, p=weights
        )

        neighbor_list = np.zeros((nk, 3), dtype=np.float64) 
        neighbor_idx = 0

        # in current order, for nk=3, we really don't have a sample where k
        # is far from both t and s. Wrong: empirically, quite a few samples
        # have k far from both t and s.
        for idx in candidate_order:
            center_k = candidate_centers[idx]

            d_exist = np.linalg.norm(neighbor_list[:neighbor_idx] - center_k.reshape(1, 3), axis=1)
            if np.any(d_exist - 2.0 <= no_overlap_thresh):
                continue

            neighbor_list[neighbor_idx] = center_k
            neighbor_idx += 1
            if neighbor_idx == nk:
                break

        if neighbor_idx < nk:
            print(f"Rejected sample {total_trials}: only {neighbor_idx} neighbors found")
            rejected += 1
            continue

        neighbor_centers = np.array(neighbor_list)

        # Config generation successful, now solve mobility problem.

        boundary_s = boundary1 + center_s.reshape(1, 3)
        source_s = source1 + center_s.reshape(1, 3)
        b_list = [boundary1, boundary_s]
        s_list = [source1, source_s]
        F_ext_list = [np.zeros(3), force_on_s]
        T_ext_list = [np.zeros(3), torque_on_s]
        B_inv_list = [B1_inv, B1_inv]

        for center_k in neighbor_centers:
            boundary_k = boundary1 + center_k.reshape(1, 3)
            source_k = source1 + center_k.reshape(1, 3)

            b_list.append(boundary_k)
            s_list.append(source_k)
            F_ext_list.append(np.zeros(3))
            T_ext_list.append(np.zeros(3))
            B_inv_list.append(B1_inv)

        try:
            V_tilde_list = imp_mfs_mobility_vec(
                b_list, s_list, F_ext_list, T_ext_list, B_inv_list, max_iter=200, tol=1e-7
            )
            M1 = source1.shape[0]
            solution_1 = V_tilde_list[0]
            velocity_1 = solution_1[3 * M1 : 3 * M1 + 3]
            omega_1 = solution_1[3 * M1 + 3 : 3 * M1 + 6]
            velocity_6d = np.concatenate([velocity_1, omega_1])
        except Exception as exc:
            print(f"Error in sample {total_trials}: {exc}")
            rejected += 1
            continue

        feat_s = np.concatenate(
            [
                center_s,
                [np.linalg.norm(center_s), min_dist_target_source],
                force_on_s,
                torque_on_s,
            ]
        )

        feature_vector = np.concatenate([feat_s, neighbor_centers.flatten()])

        X_features.append(feature_vector)
        y_targets.append(velocity_6d)
        neighbor_counts.append(nk)

        n_accepted += 1
        if n_accepted % 5 == 0:
            print(f"{n_accepted} samples generated...", neighbor_counts[-5:])

    X_features_arr = np.array(X_features, dtype=object)
    y_targets_arr = np.array(y_targets)
    neighbor_counts_arr = np.array(neighbor_counts)

    print("Acceptance ratio:", n_accepted / total_trials)
    print("Rejected samples %:", rejected / max(total_trials, 1))

    return X_features_arr, y_targets_arr, neighbor_counts_arr


def main():
    N_samples = 400
    dist_min = 2.05
    dist_max = 8.0

    X, Y, counts = generate_dataset_spheroid_multibody(N_samples, dist_min, dist_max)

    print("Dataset shapes:")
    print(" Features (ragged):", X.shape)
    print(" Targets:", Y.shape)
    print(" Neighbor counts:", counts.shape)

    import random

    tmp = random.randint(0, 1000)
    current_time = time.strftime("%H_%M", time.localtime())
    np.save(f"data/multibody/X_sphere_{current_time}_{tmp}.npy", X, allow_pickle=True)
    np.save(f"data/multibody/Y_sphere_{current_time}_{tmp}.npy", Y)
    np.save(f"data/multibody/neighbors_{current_time}_{tmp}.npy", counts)
    print(f"Saved {N_samples} samples with multibody neighbors.")


if __name__ == "__main__":
    main()