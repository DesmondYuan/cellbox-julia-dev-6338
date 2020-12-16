## ToDO

* using 7 seeds each (1 - 25 mins/seed => 10min - 3.5hr per run)

1. **A** `ongoing` quantity test [FIX network:acyclic, kernel:tanh, time_steps:5, solver:Tsit5, ]
    * network_size [5, 50, 10, 100, 20] x num_samples [64, 128, 256, 1024] x 7 seeds
        * 5 x 4 = 20 (Part I-III: 8 + 8 + 4)
        * => add one round of learning rates? <HOLD after B1>
        * => pick some sweet spots => ETA per run

2. **B1** for sweet spots [FIX network_size(sweet) and num_samples(sweet)]
    * **B1a** `ongoing` no. of ODE time points [1, 5, 10, 40] +. 1!
        * => picked 2m x 2n x 4nT = 16 (Part I-II: 8 + 8)

    * **B1b** network types (cyclic, acyclic, bifurication, multi-furication, oscillation)
        * [5x or 4x if osc==cyc]
            * `<TODO>` it is indeed `</TODO>`
        * => do 1m x 3n x nT[1, 10?/40?] x [cyclic, acyclic, and bifurication] x [Tsit5, RK4, Euler]
            * = 1 x 3 x 2 x 3 x 3 = 54
    * **B1c** solvers Tsit5, RK4, Euler [3x]

3. **B2** Quantity test [FIX network_size(sweet) + num_samples(sweet) + ode_time_steps(sweet) + ode_solver(sweet))
    * **B2a** with Gaussian noise [0, 1e-4, 1e-3, 1e-2, 1e-1, 1e+0, 1e+1]
    * **B2b** with dropout noise [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

4. **C** Kernel choice [FIX network_size(sweet) + num_samples(almost max) + ode_time_steps(sweet) + ode_solver(sweet)]
    * cross-train with sufficient data points
    * kernels: relu, tanh, polynomial (3 x 3 = 9)
        * `<TODO>`

5. **D** Network stability
    * from A, B1 and B2
    * AUROC for C


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
* **[2020-12-15 midnight] Task A Round1: [64, 128, 256, 1024] nearly finished and all look great, so running with Round2: [8, 16, 32] + m=200 (note L2_LAMBDA is not fixed on purpose for consistency)**
* **[2020-12-15 midnight] Task B1 nearly finished**
```
sbatch run_large.sh final_train5_o2_TaskA/net5_50.jl
sbatch run_large.sh final_train5_o2_TaskA/net10_20.jl
sbatch run_large.sh final_train5_o2_TaskA/net100.jl
sbatch run_large.sh final_train5_o2_TaskA/net200.jl
```
* **[2020-12-16] Task B1 finished**
* **[2020-12-16] Task A Round1: n[64, 128, 256, 1024] + m[20, 50] but nT was not working, so running with Round 2: nT = [5, 10, 20] n[8, 32, 64, 128, 256, 1024] + m[10, 20] (note m is reduced for faster training). And the round2 are reformatted for different seeds.**

```
# 5min x (36+24) x 7seeds = 420
# sbatch run_large.sh final_train5_o2_TaskB/ts1_40.jl  # 36 in total
# sbatch run_large.sh final_train5_o2_TaskB/ts5_10.jl  # 24 in total
# changed to separated by seed
sbatch run_large.sh final_train5_o2_TaskB/ts_all_seed01.jl  # 120 in total
sbatch run_large.sh final_train5_o2_TaskB/ts_all_seed23.jl  # 120 in total
sbatch run_large.sh final_train5_o2_TaskB/ts_all_seed45.jl  # 120 in total

```
