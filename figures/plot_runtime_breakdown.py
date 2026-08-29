
"""Runtime breakdown data extracted from figures/runtime_breakdown/*.txt.

Includes raw per-run timings (ms) and derived averages over the last 3 runs.
"""

# SUPERSEDED, kept for provenance: this is the published Figure 11, whose data
# is the hand-transcribed RAW_DATA below. The reproducible path is now
# benchmarks/figure11_breakdown.py -> data/fig11_breakdown_h200.csv ->
# figures/grand_M_perf.py:runtime_breakdown(). That script can re-derive this
# file's numbers from the original stdout captures with --from-logs.

RAW_DATA = {
    "10k": {
        "FMM_2body_RPY": {
            "far_field_fmm_gpu_ms": [
                13.692,
                7.555,
                7.293,
                7.225,
                7.219,
                7.222,
                7.220,
                7.318,
                7.314,
                7.303,
                7.313,
                7.316,
            ],
            "warp_hashgrid_ms": [
                22.280,
                0.730,
                0.560,
                0.470,
                0.500,
                0.460,
                0.460,
                0.530,
                0.530,
                0.570,
                0.530,
                0.540,
            ],
            "base_velocity_compute_ms": [],
            "post_base_path_ms": [],
            "near_field_construction_ms": [
                22.363,
                0.806,
                0.610,
                0.516,
                0.543,
                0.506,
                0.509,
                0.582,
                0.584,
                0.621,
                0.583,
                0.594,
            ],
            "overall_nearfield_ms": [
                3820.764,
                239.382,
                1.472,
                1.224,
                1.225,
                1.182,
                1.195,
                1.471,
                1.481,
                1.517,
                1.478,
                1.496,
            ],
        },
        "FMM_2body_NN": {
            "far_field_fmm_gpu_ms": [
                7.373,
                7.491,
                7.295,
                7.222,
                7.222,
                7.219,
                7.229,
                7.333,
                7.320,
                7.330,
                7.313,
                7.313,
            ],
            "warp_hashgrid_ms": [
                0.550,
                0.680,
                0.590,
                0.460,
                0.490,
                0.470,
                0.460,
                0.530,
                0.540,
                0.560,
                0.540,
                0.530,
            ],
            "base_velocity_compute_ms": [],
            "post_base_path_ms": [],
            "near_field_construction_ms": [
                0.604,
                0.743,
                0.643,
                0.505,
                0.532,
                0.521,
                0.509,
                0.576,
                0.593,
                0.614,
                0.590,
                0.587,
            ],
            "overall_nearfield_ms": [
                3270.948,
                290.942,
                1.712,
                1.331,
                1.369,
                1.347,
                1.330,
                1.643,
                1.660,
                1.683,
                1.657,
                1.656,
            ],
        },
        "FMM_Nbody_NN": {
            "far_field_fmm_gpu_ms": [
                7.389,
                7.531,
                7.307,
                7.259,
                7.254,
                7.244,
                7.250,
                7.332,
                7.336,
                7.333,
                7.339,
                7.340,
            ],
            "warp_hashgrid_ms": [
                0.550,
                0.680,
                0.580,
                0.470,
                0.500,
                0.480,
                0.470,
                0.530,
                0.570,
                0.530,
                0.530,
                0.540,
            ],
            "base_velocity_compute_ms": [
                2445.944,
                310.011,
                0.920,
                0.758,
                0.748,
                0.745,
                0.744,
                0.976,
                0.971,
                0.967,
                0.969,
                0.970,
            ],
            "post_base_path_ms": [
                6486.846,
                648.132,
                1.969,
                1.845,
                1.829,
                1.838,
                1.828,
                1.933,
                1.930,
                1.936,
                1.935,
                1.944,
            ],
            "near_field_construction_ms": [
                0.601,
                0.746,
                0.636,
                0.514,
                0.552,
                0.525,
                0.514,
                0.582,
                0.620,
                0.584,
                0.582,
                0.592,
            ],
            "overall_nearfield_ms": [
                8933.596,
                959.114,
                3.714,
                3.257,
                3.271,
                3.247,
                3.224,
                3.662,
                3.694,
                3.655,
                3.659,
                3.679,
            ],
        },
    },
    "50k": {
        "FMM_2body_RPY": {
            "far_field_fmm_gpu_ms": [
                20.366,
                10.923,
                10.765,
                10.648,
                10.682,
                10.650,
                10.648,
                10.744,
                10.812,
                10.747,
                10.796,
                10.720,
            ],
            "warp_hashgrid_ms": [
                26.450,
                0.750,
                0.660,
                0.530,
                0.540,
                0.510,
                0.510,
                0.580,
                0.580,
                0.620,
                0.570,
                0.580,
            ],
            "base_velocity_compute_ms": [],
            "post_base_path_ms": [],
            "near_field_construction_ms": [
                26.537,
                0.820,
                0.719,
                0.573,
                0.588,
                0.562,
                0.559,
                0.634,
                0.635,
                0.671,
                0.629,
                0.634,
            ],
            "overall_nearfield_ms": [
                3548.506,
                211.846,
                2.832,
                2.030,
                2.026,
                1.991,
                2.012,
                2.283,
                2.287,
                2.312,
                2.278,
                2.281,
            ],
        },
        "FMM_2body_NN": {
            "far_field_fmm_gpu_ms": [
                10.795,
                10.809,
                10.757,
                10.630,
                10.638,
                10.631,
                10.626,
                10.720,
                10.736,
                10.726,
                10.746,
                10.705,
            ],
            "warp_hashgrid_ms": [
                0.600,
                0.660,
                0.660,
                0.520,
                0.550,
                0.510,
                0.520,
                0.580,
                0.590,
                0.610,
                0.600,
                0.580,
            ],
            "base_velocity_compute_ms": [],
            "post_base_path_ms": [],
            "near_field_construction_ms": [
                0.662,
                0.724,
                0.712,
                0.568,
                0.599,
                0.563,
                0.567,
                0.634,
                0.646,
                0.659,
                0.651,
                0.632,
            ],
            "overall_nearfield_ms": [
                3390.697,
                271.461,
                3.432,
                2.503,
                2.530,
                2.486,
                2.483,
                2.792,
                2.798,
                2.806,
                2.815,
                2.779,
            ],
        },
        "FMM_Nbody_NN": {
            "far_field_fmm_gpu_ms": [
                11.152,
                11.142,
                10.873,
                10.793,
                10.804,
                10.801,
                10.798,
                10.977,
                10.995,
                10.983,
                10.995,
                10.989,
            ],
            "warp_hashgrid_ms": [
                0.650,
                0.720,
                0.670,
                0.530,
                0.560,
                0.530,
                0.530,
                0.590,
                0.630,
                0.580,
                0.590,
                0.580,
            ],
            "base_velocity_compute_ms": [
                2379.545,
                293.777,
                2.212,
                1.846,
                1.861,
                1.861,
                1.845,
                2.018,
                2.033,
                2.025,
                2.022,
                2.021,
            ],
            "post_base_path_ms": [
                6180.108,
                659.960,
                7.061,
                6.961,
                6.969,
                6.959,
                6.961,
                7.022,
                6.996,
                7.021,
                7.015,
                7.027,
            ],
            "near_field_construction_ms": [
                0.710,
                0.795,
                0.730,
                0.589,
                0.620,
                0.588,
                0.583,
                0.646,
                0.694,
                0.640,
                0.645,
                0.638,
            ],
            "overall_nearfield_ms": [
                8560.535,
                954.778,
                10.197,
                9.548,
                9.605,
                9.571,
                9.537,
                9.869,
                9.906,
                9.869,
                9.862,
                9.870,
            ],
        },
    },
    "100k": {
        "FMM_2body_RPY": {
            "far_field_fmm_gpu_ms": [
                25.940,
                18.522,
                18.238,
                18.317,
                18.595,
                18.604,
                18.608,
                18.625,
                18.819,
                18.850,
                18.367,
                18.771,
            ],
            "warp_hashgrid_ms": [
                24.650,
                0.830,
                0.710,
                0.590,
                0.620,
                0.590,
                0.590,
                0.640,
                0.640,
                0.680,
                0.640,
                0.640,
            ],
            "base_velocity_compute_ms": [],
            "post_base_path_ms": [],
            "near_field_construction_ms": [
                24.730,
                0.909,
                0.777,
                0.645,
                0.677,
                0.639,
                0.638,
                0.696,
                0.692,
                0.742,
                0.692,
                0.695,
            ],
            "overall_nearfield_ms": [
                3766.649,
                223.743,
                3.849,
                3.061,
                3.080,
                3.011,
                3.038,
                3.261,
                3.300,
                3.317,
                3.262,
                3.259,
            ],
        },
        "FMM_2body_NN": {
            "far_field_fmm_gpu_ms": [
                18.255,
                18.192,
                18.504,
                18.256,
                17.909,
                17.870,
                18.072,
                18.424,
                18.354,
                17.952,
                18.431,
                18.143,
            ],
            "warp_hashgrid_ms": [
                0.660,
                0.730,
                0.750,
                0.580,
                0.610,
                0.590,
                0.580,
                0.630,
                0.630,
                0.670,
                0.630,
                0.630,
            ],
            "base_velocity_compute_ms": [],
            "post_base_path_ms": [],
            "near_field_construction_ms": [
                0.720,
                0.784,
                0.811,
                0.639,
                0.661,
                0.638,
                0.633,
                0.694,
                0.687,
                0.725,
                0.688,
                0.682,
            ],
            "overall_nearfield_ms": [
                3341.833,
                250.060,
                5.122,
                3.968,
                3.969,
                3.938,
                3.932,
                4.194,
                4.187,
                4.216,
                4.196,
                4.176,
            ],
        },
        "FMM_Nbody_NN": {
            "far_field_fmm_gpu_ms": [
                18.669,
                18.546,
                18.510,
                18.474,
                18.057,
                18.290,
                18.434,
                18.233,
                18.599,
                18.658,
                18.210,
                18.670,
            ],
            "warp_hashgrid_ms": [
                0.690,
                0.790,
                0.690,
                0.600,
                0.630,
                0.600,
                0.600,
                0.640,
                0.680,
                0.640,
                0.630,
                0.630,
            ],
            "base_velocity_compute_ms": [
                2569.210,
                311.032,
                3.560,
                3.243,
                3.213,
                3.213,
                3.217,
                3.349,
                3.351,
                3.350,
                3.345,
                3.349,
            ],
            "post_base_path_ms": [
                6953.919,
                694.510,
                13.411,
                13.363,
                13.347,
                13.361,
                13.341,
                13.409,
                13.410,
                13.401,
                13.393,
                13.393,
            ],
            "near_field_construction_ms": [
                0.753,
                0.866,
                0.747,
                0.653,
                0.691,
                0.656,
                0.659,
                0.700,
                0.736,
                0.700,
                0.691,
                0.686,
            ],
            "overall_nearfield_ms": [
                9524.074,
                1006.640,
                17.922,
                17.433,
                17.421,
                17.397,
                17.379,
                17.644,
                17.683,
                17.640,
                17.620,
                17.667,
            ],
        },
    },
    "200k": {
        "FMM_2body_RPY": {
            "far_field_fmm_gpu_ms": [
                57.330,
                47.496,
                47.123,
                47.594,
                47.098,
                47.014,
                47.690,
                47.532,
                48.299,
                47.955,
                47.409,
                46.999,
            ],
            "warp_hashgrid_ms": [
                39.030,
                1.310,
                1.140,
                0.850,
                0.890,
                0.840,
                0.850,
                0.880,
                0.880,
                1.110,
                0.880,
                0.870,
            ],
            "base_velocity_compute_ms": [],
            "post_base_path_ms": [],
            "near_field_construction_ms": [
                39.137,
                1.387,
                1.211,
                0.915,
                0.959,
                0.902,
                0.911,
                0.943,
                0.950,
                1.198,
                0.939,
                0.939,
            ],
            "overall_nearfield_ms": [
                3731.727,
                203.529,
                5.980,
                5.385,
                5.423,
                5.359,
                5.367,
                5.442,
                5.468,
                5.877,
                5.430,
                5.436,
            ],
        },
        "FMM_2body_NN": {
            "far_field_fmm_gpu_ms": [
                47.434,
                47.954,
                46.922,
                46.593,
                47.392,
                47.672,
                46.995,
                47.421,
                47.896,
                47.128,
                47.893,
                47.114,
            ],
            "warp_hashgrid_ms": [
                1.220,
                1.070,
                1.150,
                0.860,
                0.880,
                0.840,
                0.840,
                0.870,
                0.880,
                0.910,
                0.880,
                0.860,
            ],
            "base_velocity_compute_ms": [],
            "post_base_path_ms": [],
            "near_field_construction_ms": [
                1.322,
                1.150,
                1.221,
                0.922,
                0.946,
                0.900,
                0.901,
                0.937,
                0.943,
                0.978,
                0.949,
                0.929,
            ],
            "overall_nearfield_ms": [
                3434.724,
                271.955,
                7.911,
                7.064,
                7.041,
                7.008,
                7.023,
                7.155,
                7.159,
                7.177,
                7.142,
                7.129,
            ],
        },
        "FMM_Nbody_NN": {
            "far_field_fmm_gpu_ms": [
                46.810,
                48.352,
                47.135,
                47.608,
                47.028,
                47.033,
                47.516,
                46.824,
                47.493,
                46.591,
                47.339,
                47.808,
            ],
            "warp_hashgrid_ms": [
                0.930,
                1.060,
                1.220,
                0.880,
                0.920,
                0.870,
                0.910,
                0.890,
                0.920,
                0.910,
                0.860,
                0.860,
            ],
            "base_velocity_compute_ms": [
                2395.344,
                315.136,
                6.365,
                6.069,
                6.063,
                6.062,
                6.133,
                6.071,
                6.064,
                6.091,
                6.058,
                6.040,
            ],
            "post_base_path_ms": [
                6554.825,
                687.895,
                26.649,
                26.563,
                26.525,
                26.536,
                26.668,
                26.497,
                26.561,
                26.555,
                26.531,
                26.526,
            ],
            "near_field_construction_ms": [
                0.994,
                1.145,
                1.293,
                0.953,
                0.989,
                0.939,
                0.975,
                0.966,
                0.990,
                0.983,
                0.930,
                0.932,
            ],
            "overall_nearfield_ms": [
                8951.354,
                1004.458,
                34.559,
                33.805,
                33.781,
                33.741,
                34.001,
                33.742,
                33.834,
                33.837,
                33.723,
                33.768,
            ],
        },
    },
    "1000k": {
        "FMM_2body_RPY": {
            "far_field_fmm_gpu_ms": [
                231.861,
                225.226,
                224.585,
                224.105,
                223.975,
                224.999,
                223.202,
                223.923,
                223.411,
                224.069,
                224.205,
                224.110,
            ],
            "warp_hashgrid_ms": [
                25.070,
                2.820,
                2.570,
                2.620,
                2.580,
                2.530,
                2.580,
                2.520,
                2.540,
                2.600,
                2.530,
                2.530,
            ],
            "base_velocity_compute_ms": [],
            "post_base_path_ms": [],
            "near_field_construction_ms": [
                25.195,
                2.922,
                2.650,
                2.710,
                2.657,
                2.608,
                2.663,
                2.595,
                2.616,
                2.678,
                2.606,
                2.611,
            ],
            "overall_nearfield_ms": [
                3541.766,
                238.054,
                26.519,
                26.468,
                26.309,
                26.252,
                26.342,
                26.238,
                26.256,
                26.319,
                26.257,
                26.247,
            ],
        },
        "FMM_2body_NN": {
            "far_field_fmm_gpu_ms": [
                226.004,
                224.631,
                225.597,
                225.058,
                224.477,
                225.503,
                223.585,
                224.445,
                225.100,
                223.702,
                225.432,
                224.100,
            ],
            "warp_hashgrid_ms": [
                2.730,
                2.590,
                2.570,
                2.560,
                2.580,
                2.530,
                2.510,
                2.750,
                2.540,
                2.600,
                2.520,
                2.520,
            ],
            "base_velocity_compute_ms": [],
            "post_base_path_ms": [],
            "near_field_construction_ms": [
                2.813,
                2.676,
                2.646,
                2.646,
                2.659,
                2.609,
                2.593,
                2.915,
                2.618,
                2.683,
                2.600,
                2.604,
            ],
            "overall_nearfield_ms": [
                3281.465,
                260.780,
                31.257,
                31.191,
                31.178,
                31.128,
                31.094,
                31.563,
                31.123,
                31.207,
                31.078,
                31.110,
            ],
        },
        "FMM_Nbody_NN": {
            "far_field_fmm_gpu_ms": [
                224.974,
                224.345,
                224.593,
                224.137,
                225.492,
                224.632,
                223.587,
                225.758,
                225.052,
                224.417,
                225.269,
                224.530,
            ],
            "warp_hashgrid_ms": [
                2.920,
                2.680,
                2.730,
                2.540,
                2.590,
                2.520,
                2.520,
                2.590,
                2.580,
                2.530,
                2.510,
                2.780,
            ],
            "base_velocity_compute_ms": [
                2417.852,
                310.525,
                28.374,
                28.386,
                28.315,
                28.331,
                28.405,
                28.499,
                28.396,
                28.344,
                28.356,
                28.419,
            ],
            "post_base_path_ms": [
                6936.679,
                776.468,
                131.851,
                131.634,
                131.725,
                131.802,
                132.128,
                131.718,
                131.724,
                131.856,
                131.676,
                131.818,
            ],
            "near_field_construction_ms": [
                3.034,
                2.773,
                2.870,
                2.617,
                2.672,
                2.597,
                2.606,
                2.672,
                2.664,
                2.604,
                2.594,
                2.886,
            ],
            "overall_nearfield_ms": [
                9357.834,
                1090.050,
                163.427,
                162.929,
                162.994,
                163.001,
                163.468,
                163.191,
                163.065,
                163.075,
                162.892,
                163.451,
            ],
        },
    },
}


def _add_near_field_setup(raw):
    for size_data in raw.values():
        for op_data in size_data.values():
            hash_ms = op_data["warp_hashgrid_ms"]
            near_ms = op_data["near_field_construction_ms"]
            if hash_ms and near_ms:
                assert len(hash_ms) == len(near_ms)
                op_data["near_field_setup_ms"] = [
                    a + b for a, b in zip(hash_ms, near_ms)
                ]
            else:
                op_data["near_field_setup_ms"] = []


def _avg_last3(raw):
    avg = {}
    for size, size_data in raw.items():
        avg[size] = {}
        for op, op_data in size_data.items():
            op_avg = {}
            for key, vals in op_data.items():
                if not vals:
                    op_avg[key] = None
                else:
                    tail = vals[-3:]
                    op_avg[key] = sum(tail) / len(tail)
            avg[size][op] = op_avg
    return avg


def _fill_base_velocity_from_overall(raw):
    for size_data in raw.values():
        for op_data in size_data.values():
            if op_data["base_velocity_compute_ms"]:
                continue
            overall = op_data.get("overall_nearfield_ms", [])
            setup = op_data.get("near_field_setup_ms", [])
            if overall and setup:
                assert len(overall) == len(setup)
                op_data["base_velocity_compute_ms"] = [
                    o - s for o, s in zip(overall, setup)
                ]


_add_near_field_setup(RAW_DATA)
_fill_base_velocity_from_overall(RAW_DATA)
AVG_LAST3 = _avg_last3(RAW_DATA)


def _component_values(size, operator, keys):
    return [AVG_LAST3[size][operator].get(key, 0.0) or 0.0 for key in keys]


def _size_label(size):
    if size.endswith("k"):
        value = int(size[:-1])
        if value >= 1000 and value % 1000 == 0:
            return f"{value // 1000}M"
    return size


def plot_runtime_summary(save_path=None, show=True, sizes=None):
    """Plot total operator time and full-NeMO component breakdown.

    This is intended as a cleaner replacement for the single stacked grouped
    bar chart in plot_avg_last3.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if sizes is None:
        sizes = ["10k", "50k", "100k", "200k", "1000k"]
    missing_sizes = [size for size in sizes if size not in AVG_LAST3]
    assert not missing_sizes, f"Unknown problem sizes: {missing_sizes}"

    operators = ["FMM_2body_RPY", "FMM_2body_NN", "FMM_Nbody_NN"]
    operator_labels = {
        "FMM_2body_RPY": "RPY",
        "FMM_2body_NN": "NeMO (excl. nbody)",
        "FMM_Nbody_NN": "NeMO",
    }
    operator_colors = {
        "FMM_2body_RPY": "#4C78A8",
        "FMM_2body_NN": "#F58518",
        "FMM_Nbody_NN": "#54A24B",
    }
    operator_markers = {
        "FMM_2body_RPY": "o",
        "FMM_2body_NN": "s",
        "FMM_Nbody_NN": "^",
    }

    component_keys = [
        "far_field_fmm_gpu_ms",
        "base_velocity_compute_ms",
        "post_base_path_ms",
        "near_field_setup_ms",
    ]
    component_labels = [
        "far-field",
        "self + 2-body",
        "nbody correction",
        "neighbor search",
    ]
    component_colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    x = np.arange(len(sizes))
    tick_labels = [_size_label(size) for size in sizes]

    fig, (ax_total, ax_breakdown) = plt.subplots(
        1,
        2,
        figsize=(11, 4.2),
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )
    fig.patch.set_facecolor("white")

    for operator in operators:
        totals = [
            sum(_component_values(size, operator, component_keys))
            for size in sizes
        ]
        ax_total.plot(
            x,
            totals,
            color=operator_colors[operator],
            marker=operator_markers[operator],
            linewidth=2.0,
            markersize=5.5,
            label=operator_labels[operator],
        )

    ax_total.set_title("(a) Total wall time", loc="left")
    ax_total.set_xlabel("Problem size")
    ax_total.set_ylabel("Runtime (ms)")
    ax_total.set_xticks(x)
    ax_total.set_xticklabels(tick_labels)
    ax_total.legend(frameon=False, loc="upper left")

    full_nemo_components = np.array(
        [_component_values(size, "FMM_Nbody_NN", component_keys) for size in sizes]
    )

    bottom = np.zeros(len(sizes))
    for component_idx, (label, color) in enumerate(
        zip(component_labels, component_colors)
    ):
        values = full_nemo_components[:, component_idx]
        ax_breakdown.bar(
            x,
            values,
            width=0.68,
            bottom=bottom,
            color=color,
            edgecolor="none",
            linewidth=0,
            label=label,
        )
        bottom += values

    ax_breakdown.set_title("(b) Full NeMO breakdown", loc="left")
    ax_breakdown.set_xlabel("Problem size")
    ax_breakdown.set_ylabel("Runtime (ms)")
    ax_breakdown.set_xticks(x)
    ax_breakdown.set_xticklabels(tick_labels)
    ax_breakdown.legend(frameon=False, loc="upper left")

    for ax in (ax_total, ax_breakdown):
        ax.set_facecolor("white")
        ax.set_ylim(bottom=0)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=1000, bbox_inches="tight")
    if show:
        plt.show()
    return fig, (ax_total, ax_breakdown)


def plot_avg_last3(save_path=None, show=True):
    import matplotlib.pyplot as plt
    import numpy as np

    sizes = ["50k", "100k", "200k"]

    operators = ["FMM_2body_RPY", "FMM_2body_NN", "FMM_Nbody_NN"]
    #operators = ["FMM_Nbody_NN"]

    step_keys = [
        "far_field_fmm_gpu_ms",
        "base_velocity_compute_ms",
        "post_base_path_ms",
        "near_field_setup_ms",
    ]
    step_labels = ["far-field", "Self+2body", "Nbody", "Neighbor Search"]

    data = {
        size: {
            op: [
                AVG_LAST3[size][op].get(k, 0.0) or 0.0 for k in step_keys
            ]
            for op in operators
        }
        for size in sizes
    }

    x = np.arange(len(sizes))
    bar_w = 0.28
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    hatches = ["///", "\\\\\\", "xxx"]

    for i, op in enumerate(operators):
        offset = (i - 1) * bar_w
        bottom = np.zeros(len(sizes))
        for step_idx, (label, color) in enumerate(zip(step_labels, colors)):
            vals = np.array([data[size][op][step_idx] for size in sizes])
            ax.bar(
                x + offset,
                vals,
                bar_w,
                bottom=bottom,
                label=label if i == 0 else None,
                color=color,
                hatch=hatches[i],
                edgecolor="black",
                linewidth=0.4,
            )
            bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("Runtime (ms)")
    ax.set_xlabel("Problem size")
    ax.set_title("Runtime breakdown (AVG_LAST3)")

    op_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="white",
            edgecolor="black",
            hatch=hatches[i],
            label=op,
        )
        for i, op in enumerate(operators)
    ]
    legend1 = ax.legend(handles=op_handles, title="Operator", loc="upper left")
    ax.add_artist(legend1)
    ax.legend(title="Step", loc="upper right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    plot_runtime_summary(
        save_path="/mnt/home/khanmd/pinn-stokes/figures/runtime_summary_two_panel.pdf",
        show=False,
    )
