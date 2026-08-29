## Jan 14

+ Optimize two_suspens_1M.py. Baseline: for t=0.5 i.e. 50 timesteps
    + t=.5 => 1.05/1.04. t=.25=>1.004/1.002. Total: 75.22s/75.23
    + After making `positions` tensors contiguousL 72.75s. t=.25=>.96