"""Tests for the dataset-v2 training pipeline (selection, labels, cache, trainer, paper harness).

CPU only, seconds.  Run from the repo root: python -m pytest tests -q
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from src import nbody_features as nf

SELF_PATH = "data/models/self_interaction_model.pt"
TWO_BODY_PATH = "data/models/two_body_combined_model.pt"
B1_PATH = "data/models/nbody_pinn_b1.pt"
MODELS_PRESENT = all(os.path.exists(p) for p in (SELF_PATH, TWO_BODY_PATH, B1_PATH))
V2_SHARDS = sorted(glob.glob("data/multibody_v2/*/*/shard_0000.npz"))
needs_models = pytest.mark.skipif(not MODELS_PRESENT, reason="model files missing")
needs_v2 = pytest.mark.skipif(not V2_SHARDS, reason="dataset v2 missing")


# ----------------------------------------------------------------------------- configurations
def _grown(P, delta, seed):
    rng = np.random.default_rng(seed)
    centers = [np.zeros(3)]
    while len(centers) < P:
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        c = centers[rng.integers(len(centers))] + d * (2.0 + delta)
        if all(np.linalg.norm(c - x) >= 2.0 + delta for x in centers):
            centers.append(c)
    return np.array(centers)


def _lattice(P, phi, jitter, seed):
    rng = np.random.default_rng(seed)
    a = ((4 * np.pi / 3) / phi) ** (1 / 3)
    g = np.arange(-4, 5)
    pts = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T * a
    pts = pts[np.argsort(np.linalg.norm(pts, axis=1), kind="stable")[:P]]
    return pts + rng.uniform(-jitter * a, jitter * a, size=pts.shape)


def configs():
    from benchmarks.cluster import uniform_sphere_cluster
    pos_u, _ = uniform_sphere_cluster(0.15, 50, seed=5)
    return {"uniform": pos_u, "grown": _grown(24, 0.2, 3), "lattice": _lattice(32, 0.1, 0.1, 7)}


# ----------------------------------------------------------------------------- 1. selection
@needs_models
@pytest.mark.parametrize("max_neighbors,cutoff,pc", [(10, 6.0, 6.0), (None, 6.0, 6.0), (None, 8.0, 6.0), (None, 8.0, 8.0)])
def test_select_pair_neighbours_matches_operators(max_neighbors, cutoff, pc):
    from src.mob_op_nbody import Mob_Op_Nbody
    base = Mob_Op_Nbody(shape="sphere", self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH, nbody_nn_path=B1_PATH,
                        neighbor_cutoff=cutoff, max_neighbors=10 ** 9 if max_neighbors is None else max_neighbors)
    n_checked = 0
    for name, pos in configs().items():
        t_idx, s_idx, indptr, indices = nf.select_pair_neighbours(pos, pc, cutoff, max_neighbors)
        assert np.all(t_idx < s_idx)
        expected = [(t, s) for t in range(len(pos)) for s in range(t + 1, len(pos))
                    if np.linalg.norm(pos[s] - pos[t]) <= pc]
        assert list(zip(t_idx.tolist(), s_idx.tolist())) == expected, name
        for i, (t, s) in enumerate(expected):
            ref = base._select_neighbor_indices(pos, t, s)
            got = indices[indptr[i]:indptr[i + 1]].tolist()
            assert got == ref, (name, t, s)
            assert got == base._select_neighbor_indices(pos, s, t)  # symmetric in (t, s)
            n_checked += 1
    assert n_checked > 300


@needs_models
def test_moments_pair_rows_match_per_pair_construction(tmp_path):
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
    from src.model_archs import MultiBodyMoments
    wt = tmp_path / "rand.wt"
    torch.manual_seed(0)
    torch.save(MultiBodyMoments(4.69, zero_init_head=False).state_dict(), wt)
    for max_neighbors, cutoff in [(10, 6.0), (None, 8.0)]:
        op = Mob_Op_Nbody_Moments(shape="sphere", self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH,
                                  nbody_nn_path=str(wt), max_neighbors=max_neighbors, neighbor_cutoff=cutoff)
        pos = configs()["uniform"]
        pairs, X_ts, X_st = op._pair_rows(pos)
        assert len(pairs) > 100 and X_ts.shape == (len(pairs), 111)
        for i, (t, s) in enumerate(pairs):
            idx = op._select_neighbor_indices(pos, t, s)
            assert idx, (t, s)
            nbr = (pos[idx] - pos[t])[None]
            X = nf.moment_features((pos[s] - pos[t])[None], nbr, np.ones((1, len(idx))), op.mean_dist_s)[0]
            assert np.allclose(X, X_ts[i], rtol=1e-6, atol=1e-6)
        assert np.allclose(X_st[:, :3], -X_ts[:, :3]) and np.array_equal(X_st[:, 3:], X_ts[:, 3:])


# ----------------------------------------------------------------------------- 2. two-body blocks
@needs_models
def test_two_body_blocks_match_velocity_and_operator():
    from src.mob_op_2b_combined import NNMob
    two_nn = torch.jit.load(TWO_BODY_PATH).eval()
    rng = np.random.default_rng(0)
    n = 200
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    dist = rng.uniform(2.1, 5.99, size=n)
    s_vec = u * dist[:, None]
    F = rng.normal(size=(n, 6))
    blocks = nf.two_body_blocks(two_nn, s_vec, dist)
    v = nf.two_body_velocity(two_nn, s_vec, dist, F, median_2b=nf.MEDIAN_2B_OPERATOR)
    assert np.allclose(np.einsum("nij,nj->ni", blocks, F), v, rtol=1e-5, atol=1e-7)
    # against the operator: force on the source only, target at the origin -> v_target = M2b_ts F_s
    op = NNMob("sphere", SELF_PATH, TWO_BODY_PATH, nn_only=False, rpy_only=False)
    for i in range(5):
        config = np.zeros((2, 7)); config[:, 6] = 1.0
        config[1, :3] = s_vec[i]
        force = np.zeros((2, 6)); force[1] = F[i]
        vel = op.apply(config, force, 1.0)
        assert np.allclose(vel[0], blocks[i] @ F[i], rtol=1e-5, atol=1e-7), i


@needs_models
def test_two_body_blocks_match_operator_in_6_8_shell():
    """pair_cutoff-8 labels: the 2b NN (trained to d=8) is the operator's base out to 8 when switch_dist=8,
    and it stays within ~1e-3 of the RPY base there (measured 2e-4 median) -- the base swap is benign."""
    from src.mob_op_2b_combined import NNMob
    two_nn = torch.jit.load(TWO_BODY_PATH).eval()
    rng = np.random.default_rng(1)
    n = 40
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    dist = rng.uniform(6.01, 7.99, size=n)
    s_vec = u * dist[:, None]
    F = rng.normal(size=(n, 6))
    blocks = nf.two_body_blocks(two_nn, s_vec, dist)
    op8 = NNMob("sphere", SELF_PATH, TWO_BODY_PATH, nn_only=False, rpy_only=False, switch_dist=8.0)
    op6 = NNMob("sphere", SELF_PATH, TWO_BODY_PATH, nn_only=False, rpy_only=False)
    for i in range(6):
        config = np.zeros((2, 7)); config[:, 6] = 1.0
        config[1, :3] = s_vec[i]
        force = np.zeros((2, 6)); force[1] = F[i]
        vel = op8.apply(config, force, 1.0)
        assert np.allclose(vel[0], blocks[i] @ F[i], rtol=1e-5, atol=1e-7), i
        vel6 = op6.apply(config, force, 1.0)  # RPY base beyond 6
        assert np.linalg.norm(vel[0] - vel6[0]) < 2e-3 * np.linalg.norm(vel6[0]) + 1e-9, i


@needs_models
def test_moments_operator_pair_cutoff8(tmp_path):
    """pair_cutoff=8 operator: constructs with switch_dist=8, corrects the 6-8 shell, grand M stays symmetric."""
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
    from src.model_archs import MultiBodyMoments
    wt = tmp_path / "rand.wt"
    torch.save(MultiBodyMoments(4.69, zero_init_head=False).state_dict(), wt)
    op = Mob_Op_Nbody_Moments(shape="sphere", self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH,
                              nbody_nn_path=str(wt), switch_dist=8.0, pair_cutoff=8.0,
                              neighbor_cutoff=8.0, max_neighbors=None)
    pos = configs()["uniform"]
    pairs, X_ts, X_st = op._pair_rows(pos)
    dists = [np.linalg.norm(pos[s] - pos[t]) for t, s in pairs]
    assert max(dists) <= 8.0 and max(dists) > 6.0  # the 6-8 shell is corrected
    expected = {(t, s) for t in range(len(pos)) for s in range(t + 1, len(pos))
                if np.linalg.norm(pos[s] - pos[t]) <= 8.0}
    assert set(pairs) <= expected and len(expected) - len(pairs) < 5  # only zero-neighbour pairs may drop
    n = 12
    Mn = np.zeros((6 * n, 6 * n))
    for j in range(6 * n):
        F = np.zeros((n, 6)); F[j // 6, j % 6] = 1.0
        Mn[:, j] = op.get_nbody_velocity(pos[:n], F, 1.0).reshape(-1)
    assert np.linalg.norm(Mn) > 0
    assert np.linalg.norm(Mn - Mn.T) / np.linalg.norm(Mn) < 1e-5


# ----------------------------------------------------------------------------- 3. baseline rows
@needs_models
def test_baseline_features_torch_and_b1_reciprocity():
    from src.mob_op_nbody import Mob_Op_Nbody
    from src.model_archs import MultiBodyCorrectionB1
    base = Mob_Op_Nbody(shape="sphere", self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH, nbody_nn_path=B1_PATH)
    pos = configs()["uniform"]
    t_idx, s_idx, indptr, indices = nf.select_pair_neighbours(pos, 6.0, 6.0, 10)
    keep = np.diff(indptr) > 0
    nbr, mask = nf.pad_neighbours(pos, t_idx, indptr, indices, K=10)
    s_vec = pos[s_idx] - pos[t_idx]
    X_np = nf.baseline_features(s_vec[keep], nbr[keep], mask[keep], base.mean_dist_s)
    X_t = nf.baseline_features_torch(*[torch.as_tensor(a, dtype=torch.float64) for a in (s_vec[keep], nbr[keep], mask[keep])],
                                     base.mean_dist_s).numpy()
    assert np.allclose(X_np, X_t, rtol=1e-5, atol=1e-5)
    # the operator's own (t,s) and (s,t) rows: MLP columns 33: equal, s_vec negated
    torch.manual_seed(1)
    model = MultiBodyCorrectionB1(base.mean_dist_s).eval()
    for i in np.nonzero(keep)[0][:40]:
        t, s = int(t_idx[i]), int(s_idx[i])
        idx = base._select_neighbor_indices(pos, t, s)
        row_ts = base._build_pair_feature_vector(pos[t], pos[s], [pos[k] - pos[t] for k in idx])
        row_st = base._build_pair_feature_vector(pos[s], pos[t], [pos[k] - pos[s] for k in idx])
        assert np.allclose(row_ts, X_np[list(np.nonzero(keep)[0]).index(i)], rtol=1e-5, atol=1e-5)
        assert np.allclose(row_ts[33:], row_st[33:], rtol=1e-5, atol=1e-5)
        assert np.allclose(row_ts[:3], -row_st[:3])
        with torch.no_grad():
            K_ts = model.predict_mobility(torch.as_tensor(row_ts)[None])[0]
            K_st = model.predict_mobility(torch.as_tensor(row_st)[None])[0]
        assert torch.allclose(K_st, K_ts.T, rtol=1e-4, atol=1e-6)


# ----------------------------------------------------------------------------- 5. split
@needs_v2
def test_v2_split_is_configuration_level():
    seeds = np.concatenate([np.load(f)["seed"] for f in V2_SHARDS])
    assert len(np.unique(seeds)) == len(seeds)
    val = nf.v2_is_val(seeds)
    assert 0.08 < val.mean() < 0.12
    assert np.array_equal(val, nf.v2_is_val(seeds))  # deterministic


# ----------------------------------------------------------------------------- 4. cache parity + oracle
def _build_smoke_cache(out, max_shards=2, max_configs=6, pair_cutoff=6.0):
    cmd = [sys.executable, "experiments/build_nbody_v2_cache.py", "--out", str(out), "--max-shards", str(max_shards),
           "--max-configs", str(max_configs), "--workers", "1", "--pair-cutoff", str(pair_cutoff)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


@needs_models
@needs_v2
@pytest.mark.parametrize("pc", [6.0, 8.0])
def test_cache_rows_match_operators_and_labels(tmp_path, pc):
    import experiments.train_nbody_v2 as tr
    from src.mob_op_nbody import Mob_Op_Nbody
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
    from src.model_archs import MultiBodyMoments
    out = _build_smoke_cache(tmp_path / "cache", pair_cutoff=pc)
    cfg = np.load(out / "configs.npz")
    two_nn = torch.jit.load(TWO_BODY_PATH).eval()
    wt = tmp_path / "rand.wt"
    torch.save(MultiBodyMoments(4.69, zero_init_head=False).state_dict(), wt)
    common = dict(shape="sphere", self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH)
    base = Mob_Op_Nbody(nbody_nn_path=B1_PATH, **common)
    for variant in ["k10_rc6", "kinf_rc8"]:
        cache = tr.V2Cache(out, variant, "cpu", "cpu")
        K, rc = nf.SELECTION_VARIANTS[variant]
        mom = Mob_Op_Nbody_Moments(nbody_nn_path=str(wt), max_neighbors=K, neighbor_cutoff=rc,
                                   pair_cutoff=pc, switch_dist=max(6.0, pc), **common)
        for c in [0, int(cfg["P"].shape[0]) - 1]:
            P = int(cfg["P"][c]); pos = cfg["positions"][c, :P]
            rows = np.nonzero(cache.pair_cfg.numpy() == c)[0]
            pairs, X_ts, _ = mom._pair_rows(pos)
            assert [(int(t), int(s)) for t, s in zip(cache.pair_t[rows], cache.pair_s[rows])] == pairs
            X = cache.features(rows, "moments").numpy()
            assert np.allclose(X, X_ts, rtol=1e-5, atol=1e-5)
            if variant == "k10_rc6":
                Xb = cache.features(rows, "baseline").numpy()
                for r, (t, s) in zip(range(len(pairs)), pairs):
                    idx = base._select_neighbor_indices(pos, t, s)
                    row = base._build_pair_feature_vector(pos[t], pos[s], [pos[k] - pos[t] for k in idx])
                    assert np.allclose(Xb[r], row, rtol=1e-5, atol=1e-5)
            # labels: R = 0.5 (M_ts + M_st^T) - M2b_ts with the operator's two-body block
            shard = np.load(sorted(glob.glob("data/multibody_v2/*/*/shard_*.npz"))[int(cfg["shard"][c])])
            M = shard["M"][int(cfg["index"][c])].astype(np.float64).reshape(P, 6, P, 6)
            t = cache.pair_t[rows].numpy(); s = cache.pair_s[rows].numpy()
            Mts = 0.5 * (M[t, :, s, :] + np.transpose(M[s, :, t, :], (0, 2, 1)))
            s_vec = pos[s] - pos[t]
            M2b = nf.two_body_blocks(two_nn, s_vec, np.linalg.norm(s_vec, axis=1))
            assert np.allclose(cache.labels(rows).numpy(), (Mts - M2b), rtol=1e-4, atol=1e-6)


@needs_models
@needs_v2
def test_oracle_residual_reproduces_grand_M(tmp_path):
    """Error decomposition of the 2-body operator on real v2 configurations: adding the cached near-pair residual
    blocks, then the diagonal residual, then the far-pair (d > 6, RPY) residuals reproduces M F to the label
    noise -- a wrong sign / orientation convention in the labels fails this by construction."""
    import experiments.train_nbody_v2 as tr
    from src.mob_op_2b_combined import NNMob
    out = _build_smoke_cache(tmp_path / "cache", max_shards=4, max_configs=3)
    cfg = np.load(out / "configs.npz")
    cache = tr.V2Cache(out, "kinf_rc8", "cpu", "cpu")
    Mtt_res = np.load(out / "Mtt_res.npy")
    op = NNMob("sphere", SELF_PATH, TWO_BODY_PATH, nn_only=False, rpy_only=False)
    shards = sorted(glob.glob("data/multibody_v2/*/*/shard_*.npz"))
    rng = np.random.default_rng(0)
    errs = []
    for c in range(int(cfg["P"].shape[0])):
        P = int(cfg["P"][c]); pos = cfg["positions"][c, :P]
        M = np.load(shards[int(cfg["shard"][c])])["M"][int(cfg["index"][c])].astype(np.float64)
        Mb = M.reshape(P, 6, P, 6)
        F = rng.normal(size=(P, 6))
        v_true = (M @ F.reshape(-1)).reshape(P, 6)
        config = np.zeros((P, 7)); config[:, :3] = pos; config[:, 6] = 1.0
        v_2b = op.apply(config, F, 1.0)
        rows = np.nonzero(cache.pair_cfg.numpy() == c)[0]
        R = cache.labels(rows).numpy().astype(np.float64)
        t = cache.pair_t[rows].numpy(); s = cache.pair_s[rows].numpy()
        v_near = v_2b.copy()
        np.add.at(v_near, t, np.einsum("nij,nj->ni", R, F[s]))
        np.add.at(v_near, s, np.einsum("nji,nj->ni", R, F[t]))
        v_diag = v_near + np.einsum("nij,nj->ni", Mtt_res[c, :P].astype(np.float64).reshape(P, 6, 6), F)
        v_far = v_diag.copy()
        for a in range(P):
            for b in range(P):
                if a != b and np.linalg.norm(pos[b] - pos[a]) > 6.0:
                    v_far[a] += (Mb[a, :, b, :] - op.compute_rpy_mobility(pos[b] - pos[a])) @ F[b]
        e = lambda v: np.linalg.norm(v - v_true) / np.linalg.norm(v_true)
        errs.append((e(v_2b), e(v_near), e(v_diag), e(v_far)))
        assert e(v_far) < 3e-3, (c, errs[-1])          # label symmetrisation + fine-MFS noise only
    E = np.array(errs)
    assert np.median(E[:, 1] / E[:, 0]) < 0.5 and np.all(E[:, 2] <= E[:, 1] + 1e-3) and np.median(E[:, 3] / E[:, 0]) < 0.05


# ----------------------------------------------------------------------------- 6. trainer smoke
@needs_models
@needs_v2
@pytest.mark.parametrize("model", ["moments", "baseline"])
def test_trainer_smoke_and_operator_symmetry(tmp_path, model):
    from benchmarks.cluster import uniform_sphere_cluster
    out = _build_smoke_cache(tmp_path / "cache", max_shards=3, max_configs=12)
    run = tmp_path / "run"
    cmd = [sys.executable, "experiments/train_nbody_v2.py", "--cache", str(out), "--model", model, "--variant", "k10_rc6",
           "--epochs", "2", "--max-steps", "6", "--batch", "64", "--fit-rows", "500", "--eval-rows", "300", "--eval-every", "1",
           "--device", "cpu", "--data-on", "cpu", "--out", str(run), "--publish", "--publish-name", f"test_{model}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (run / "model.pt").exists() and (run / "metrics.json").exists()
    m = json.load(open(run / "metrics.json"))
    assert m["variant"] == "k10_rc6" and "by_family" in m and m["twobody_only"]["prmse_lin"] > 0
    side = json.load(open(f"data/models/test_{model}.json"))
    assert side["max_neighbors"] == 10 and side["neighbor_cutoff"] == 6.0
    for p in [f"data/models/test_{model}.pt", f"data/models/test_{model}.json", f"experiments/test_{model}.wt"]:
        os.remove(p)
    # the exported model runs in the operator and gives a symmetric n-body grand M
    if model == "moments":
        from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
        op = Mob_Op_Nbody_Moments(shape="sphere", self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH, nbody_nn_path=str(run / "model.pt"))
    else:
        from src.mob_op_nbody import Mob_Op_Nbody
        op = Mob_Op_Nbody(shape="sphere", self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH, nbody_nn_path=str(run / "model.pt"))
    centers, _ = uniform_sphere_cluster(0.2, 12, seed=0)
    n = centers.shape[0]
    Mn = np.zeros((6 * n, 6 * n))
    for j in range(6 * n):
        F = np.zeros((n, 6)); F[j // 6, j % 6] = 1.0
        Mn[:, j] = op.get_nbody_velocity(centers, F, 1.0).reshape(-1)
    assert np.linalg.norm(Mn) > 0
    assert np.linalg.norm(Mn - Mn.T) / np.linalg.norm(Mn) < 1e-5


# ----------------------------------------------------------------------------- 7. harness
def test_harness_cases_and_metric():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pav2", "benchmarks/paper_accuracy_v2.py")
    pav2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(pav2)
    c3, c4 = pav2.cases("fig3"), pav2.cases("fig4")
    assert len(c3) == 160 and len(c4) == 1120
    assert [c for c in c4 if c["phi"] == 0.075 and c["N"] == 70][3]["seed"] == 123 + 2 * 1000 + 5 * 100 + 3
    assert sorted({c["seed"] for c in c3}) == list(range(123, 133))
    assert len(pav2.cases("fig4", phis=[0.1])) == 140 and len(pav2.cases("cluster")) == 6
    if os.path.exists("tmp/reference_sphere_1.0.csv"):
        config, forces, velocity = pav2.load_case({"exp": "cluster", "N": 10, "phi": 1.0, "seed": 0})
        assert config.shape == (10, 7) and velocity.shape == (10, 6)
        assert pav2.nearfield_interactions(config) > 0
    warp = pytest.importorskip("warp")  # noqa: F841  (the paper harness imports it at module level)
    from benchmarks.accuracy_grand_M import _compute_error_stats
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(50, 6)), rng.normal(size=(50, 6))
    ref = _compute_error_stats(a, b); got = pav2.compute_error_stats(a, b)
    for k in ref:
        assert np.isclose(ref[k], got[k])


def test_harness_selection_for_pc8(tmp_path):
    """The sidecar-vs-registry cross-check covers pair_cutoff; absent sidecar keys default to the pc6 convention."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("pav2_sel", "benchmarks/paper_accuracy_v2.py")
    pav2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(pav2)
    mdl = tmp_path / "m.pt"; mdl.touch()
    pav2.MODELS["mom_v2_kinf_rc8_pc8"] = str(mdl)
    assert pav2._selection_for("mom_v2_kinf_rc8_pc8") == (None, 8.0, 8.0)  # no sidecar: registry wins
    (tmp_path / "m.json").write_text(json.dumps({"max_neighbors": None, "neighbor_cutoff": 8.0, "pair_cutoff": 8.0}))
    assert pav2._selection_for("mom_v2_kinf_rc8_pc8") == (None, 8.0, 8.0)
    (tmp_path / "m.json").write_text(json.dumps({"max_neighbors": None, "neighbor_cutoff": 8.0, "pair_cutoff": 6.0}))
    with pytest.raises(AssertionError):
        pav2._selection_for("mom_v2_kinf_rc8_pc8")  # a pc6-trained model must not run at pc8
    pav2.MODELS["mom_v2_kinf_rc6"] = str(mdl)
    (tmp_path / "m.json").write_text(json.dumps({"max_neighbors": None, "neighbor_cutoff": 6.0}))
    assert pav2._selection_for("mom_v2_kinf_rc6") == (None, 6.0, 6.0)  # old sidecar without pair_cutoff
