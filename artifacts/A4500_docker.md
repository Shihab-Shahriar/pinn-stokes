Velocity computed in 1.119 seconds.
Step t=0.48
[MobFMM] far-field FMM GPU time: 253.039 ms
Building spatial grid for neighbor search...
Positions shape: (1047968,)
Counting neighbors per particle...
Total near-field pairs found: 20218290
Collecting neighbor edge indexes...
Warp::HashGrid took 8.64 ms
no of NN interactions: 20218290
[Mob_Nbody] Base velocity compute time: 261.199 ms
[Mob_Nbody] Post-base path time: 626.890 ms
[MobFMM] peak GPU memory: allocated 2579.99 MB, reserved 3544.00 MB
[MobFMM] near-field construction: 8.897 ms
[MobFMM] near-field operator time: 888.566 ms
[MobFMM] overall Nearfield time: 898.152 ms
[MobFMM] near-field particles updates per sec: 1,166,804.12
[MobFMM] total GPU time: 1153.859 ms
[MobFMM] total particles updates per sec: 908,229.20


Velocity computed in 1.155 seconds.
Step t=0.49
[MobFMM] far-field FMM GPU time: 227.594 ms
Building spatial grid for neighbor search...
Positions shape: (1047968,)
Counting neighbors per particle...
Total near-field pairs found: 20220758
Collecting neighbor edge indexes...
Warp::HashGrid took 9.13 ms
no of NN interactions: 20220758
[Mob_Nbody] Base velocity compute time: 262.221 ms
[Mob_Nbody] Post-base path time: 627.079 ms
[MobFMM] peak GPU memory: allocated 2579.99 MB, reserved 3544.00 MB
[MobFMM] near-field construction: 9.435 ms
[MobFMM] near-field operator time: 889.843 ms
[MobFMM] overall Nearfield time: 899.837 ms
[MobFMM] near-field particles updates per sec: 1,164,619.93
[MobFMM] total GPU time: 1129.722 ms
[MobFMM] total particles updates per sec: 927,633.69


Velocity computed in 1.130 seconds.
Simulation complete.
Total simulation time: 55.90 seconds.
Wrote 50 per-step rows -> /persistent/results/a4500_two_drop_fp32.csv
@@ steps=50  total_sim_time=55.90s  (1117.9 ms/step wall incl. Euler update)
@@ wall_s      mean      1.12  median      1.12  min      1.07  max      1.17
@@ far_ms      mean    214.79  median    215.82  min    198.91  max    253.04
@@ near_ms     mean    900.75  median    895.90  min    856.40  max    961.04
@@ nsearch_ms  mean      8.53  median      8.44  min      7.46  max      9.88
@@ self2b_ms   mean    263.21  median    261.14  min    249.08  max    282.46
@@ nbody_ms    mean    628.13  median    625.78  min    598.34  max    668.57
@@ total_gpu   mean   1117.28  median   1118.74  min   1074.47  max   1164.77
@@ peak_mem_mb  max 2580  (torch allocated)
root@dabb3ad1fdc7:~/pinn-stokes# 

