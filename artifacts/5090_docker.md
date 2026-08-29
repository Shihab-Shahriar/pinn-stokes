[MobFMM] far-field FMM GPU time: 45.390 ms
Building spatial grid for neighbor search...
Positions shape: (750000,)
Counting neighbors per particle...
Total near-field pairs found: 15649214
Collecting neighbor edge indexes...
Warp::HashGrid took 2.69 ms
no of NN interactions: 15649214
[Mob_Nbody] Base velocity compute time: 44.474 ms
[Mob_Nbody] Post-base path time: 116.343 ms
[MobFMM] peak GPU memory: allocated 2442.74 MB, reserved 3986.00 MB
[MobFMM] near-field construction: 2.902 ms
[MobFMM] near-field operator time: 161.000 ms
[MobFMM] overall Nearfield time: 164.197 ms
[MobFMM] near-field particles updates per sec: 4,567,682.49
[MobFMM] total GPU time: 210.705 ms
[MobFMM] total particles updates per sec: 3,559,472.42


[MobFMM] far-field FMM GPU time: 45.383 ms
Building spatial grid for neighbor search...
Positions shape: (750000,)
Counting neighbors per particle...
Total near-field pairs found: 15649214
Collecting neighbor edge indexes...
Warp::HashGrid took 2.74 ms
no of NN interactions: 15649214
[Mob_Nbody] Base velocity compute time: 44.492 ms
[Mob_Nbody] Post-base path time: 116.410 ms
[MobFMM] peak GPU memory: allocated 2442.74 MB, reserved 3986.00 MB
[MobFMM] near-field construction: 2.951 ms
[MobFMM] near-field operator time: 161.085 ms
[MobFMM] overall Nearfield time: 164.337 ms
[MobFMM] near-field particles updates per sec: 4,563,793.81
[MobFMM] total GPU time: 211.060 ms
[MobFMM] total particles updates per sec: 3,553,498.88


@@ N=750,000  total=211.12 ms (+-0.17)  far=45.38 ms  3.552 M updates/s  VRAM 2.39 GB

wrote 5 rows -> /persistent/results/fig12_5090.csv

 N          total ms    far ms   M updates/s   VRAM GB
    50,000     32.88     14.29        1.520      1.01
   100,000     44.80     16.99        2.232      1.94
   200,000     70.79     23.55        2.825      2.20
   500,000    148.03     35.82        3.378      2.30
   750,000    211.12     45.38        3.552      2.39
root@600cd41cf048:~/pinn-stokes# 




  Expected far-field error at that point (relative L2 of the treecode's far-field velocity vs. an exact fp64
  RPY sum at 2048 sampled targets, benchmarks/mac_calibration.py):

  ┌─────────────────────────────────────────┬─────────┬──────────────┬───────────────────────────────────┐
  │                  cloud                  │ loading │  rel_far,    │         rel_far, fp32 L3          │
  │                                         │         │     fp64     │                                   │
  ├─────────────────────────────────────────┼─────────┼──────────────┼───────────────────────────────────┤
  │ uniform φ=0.1, N=100k (the Fig-12 type, │ random  │ 3.85e-4      │ — (not measured on uniform; see   │
  │  H200)                                  │         │              │ below)                            │
  ├─────────────────────────────────────────┼─────────┼──────────────┼───────────────────────────────────┤
  │ uniform, N=100k                         │ gravity │ 8.7e-7       │ —                                 │
  ├─────────────────────────────────────────┼─────────┼──────────────┼───────────────────────────────────┤
  │ two-drop t=0, N=1.05M                   │ random  │ 3.09e-4      │ 3.09e-4                           │
  ├─────────────────────────────────────────┼─────────┼──────────────┼───────────────────────────────────┤
  │ two-drop t=0, N=1.05M                   │ gravity │ 1.13e-6      │ 1.14e-6                           │
  ├─────────────────────────────────────────┼─────────┼──────────────┼───────────────────────────────────┤
  │ 3k drop                                 │ random  │ 4.08e-4      │ —                                 │
  └─────────────────────────────────────────┴─────────┴──────────────┴───────────────────────────────────┘
