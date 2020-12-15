## ToDO

* using 7 seeds each

-[] A quantity test [FIX network:acyclic, kernel:tanh, time_steps:5, solver:Tsit5, ]
    - network_size [5, 50, 10, 100, 20] x num_samples [64, 128, 256, 1024] x 7 seeds
    - 5 x 4 x 7 = 140
        => add one round of learning rates? <HOLD after B1>
        => pick some sweet spots => ETA per run

-[] B1 for sweet spots [FIX network_size(sweet) and num_samples(sweet)]
    - B1a no. of ODE time points [1, 5, 10, 40] +. 1!
    - B2b network types (cyclic, acyclic, bifurication, multi-furication, oscillation)
        [5x or 4x if osc==cyc]
        <TODO>
    - B3c solvers Tsit5, RK4, Euler [3x]
    - 4 x 5 x 3 = 60

-[] B2 Quantity test [FIX network_size(sweet) + num_samples(sweet) + ode_time_steps(sweet) + ode_solver(sweet))
    - B2a with Gaussian noise [0, 1e-4, 1e-3, 1e-2, 1e-1, 1e+0, 1e+1] (7 x 7 = 56)
    - B2b with dropout noise [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8] (8 x 7 = 56)

-[] C Kernel choice [FIX network_size(sweet) + num_samples(almost max) + ode_time_steps(sweet) + ode_solver(sweet)]
    - cross-train with sufficient data points
    - kernels: relu, tanh, polynomial (3 x 3 x 7 = 63)
    <TODO>

-[] D Network stability
    - from A, B1 and B2
    - AUROC for C


## Training Log

### Task A and B1a
* **[2020-12-14] Start running Task A and Task B1 on O2**
  * An issued was identified later that d\mu is supposed to be constantly zero.
* **[2020-12-15] Fix the bugs and synchronize everything onto github**
* **[2020-12-15] Note that the L2_LAMBDA was taken to be 3e-8, but should be 1e-4**
```
# Task A - Part I - 48 * 20 min
sbatch run_large.sh final_train5_o2_TaskA/net5_50.jl
# Task A - Part II - 48 * 20 min
sbatch run_large.sh final_train5_o2_TaskA/net10_20.jl
# Task A - Part III - 24 * 40 min
sbatch run_large.sh final_train5_o2_TaskA/net100.jl

# Task B - Part I - 96 * 10 min
sbatch run_large.sh final_train5_o2_TaskB/ts1_40.jl
# Task B - Part II - 96 * 10 min
sbatch run_large.sh final_train5_o2_TaskB/ts5_10.jl

```
