## ToDO

* using 7 seeds each (1 - 25 mins/seed => 10min - 3.5hr per run)

1. **A** `finished` `analyzed` quantity test [FIX network:acyclic, kernel:tanh, time_steps:5, solver:Tsit5, ]
    * network_size [5, 50, 10, 100, 20] x num_samples [64, 128, 256, 1024] x 7 seeds
        * 5 x 4 = 20 (Part I-III: 8 + 8 + 4)
        * => add one round of learning rates? <HOLD after B1>
        * => pick some sweet spots => ETA per run (Round 1)
    * Round 1 looks all good. Round 2: adding [8, 16, 32]

2. **B1** `finished` for sweet spots [FIX network_size(sweet) and num_samples(sweet)]
    * `finished` no. of ODE time points [1, 5, 10, 40] +. 1!
        * Round 1 => picked 2m x 2n x 4nT = 16 (Part I-II: 8 + 8), Round 1 has a critical bug and is `deprecated`.
        * Round 2 `nearly finished` `analysis in progress`

3. **B2** `finished` Quantity test [FIX network_size(sweet) + num_samples(sweet) + ode_time_steps(sweet) + ode_solver(sweet))
    * **B2a** with Gaussian noise [0, 1e-4, 1e-3, 1e-2, 1e-1, 1e+0, 1e+1]
    * **B2b** with dropout noise [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

4. **B3a** `finished` network types (cyclic, acyclic, bifurication, multi-furication, oscillation)  **x B3b** solvers Tsit5, RK4, Euler [3x]
    * [5x or 4x if osc==cyc]
        * `<TODO>` it is indeed with preliminary testing `<\TODO>` `finished`
        * `<TODO>` bifurication? `<\TODO>` `finished`
    * ODE solvers efficiency
        ```
        ts = 0:0.2:10
        @elapsed [solve(ode_golds[1], Tsit5(), saveat=ts) for _ in 1:10]  # 1.47s
        @elapsed [solve(ode_golds[1], Euler(), saveat=ts, tstops=tstops=0:0.01:10) for _ in 1:10]  # 0.16s
        @elapsed [solve(ode_golds[1], Midpoint(), saveat=ts) for _ in 1:10]  # 1.42s
        @elapsed [solve(ode_golds[1], Heun(), saveat=ts) for _ in 1:10]  # 1.34s
        @elapsed [solve(ode_golds[1], RK4(), saveat=ts) for _ in 1:10]  # 2.80s
        @elapsed [solve(ode_golds[1], DP5(), saveat=ts) for _ in 1:10]  # 1.12s
        ```
    * => do 1m[10] x 2n[32, 256] x nT[1, 10, 30] x [cyclic, acyclic, bifurication] x [Tsit5, Euler, DP5, RK4]
        * = 1 x 2 x 3 x 3 x 4 = 72

5. **C** Kernel choice [FIX network_size(sweet) + num_samples(almost max) + ode_time_steps(sweet) + ode_solver(sweet)]
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
* **[2020-12-16] Task B2 started**
```
sbatch run_mini.sh final_train5_o2_TaskB2/noise_dropout.jl  # 120 in total
sbatch run_mini.sh final_train5_o2_TaskB2/noise_gaussian.jl  # 120 in total
```
* **[2020-12-16] Task B3 started**
```
sbatch run_mini.sh "final_train5_o2_TaskB3/solvers_euler&RK4.jl"  # 216 in total
sbatch run_mini.sh "final_train5_o2_TaskB3/solvers_Tsit5&DP5.jl"  # 216 in total
```
* **[2020-12-16 night] Task A round 2 finished**
* **[2020-12-16 night] Task B2 finished**
* **[2020-12-16] Task B3 takes too long because of the cyclic data**
```
sbatch run_mini.sh "final_train5_o2_TaskB3/solvers_euler&RK4.jl"  # 144 in total
sbatch run_mini.sh "final_train5_o2_TaskB3/solvers_Tsit5&DP5.jl"  # 144 in total
sbatch run_large.sh "final_train5_o2_TaskB3/solvers_Tsit5&DP5_cyclic1.jl"  # 24 in total
sbatch run_large.sh "final_train5_o2_TaskB3/solvers_Tsit5&DP5_cyclic2.jl"  # 24 in total
sbatch run_large.sh "final_train5_o2_TaskB3/solvers_Tsit5&DP5_cyclic3.jl"  # 24 in total
sbatch run_large.sh "final_train5_o2_TaskB3/solvers_Tsit5&DP5_cyclic4.jl"  # 24 in total
sbatch run_large.sh "final_train5_o2_TaskB3/solvers_Tsit5&DP5_cyclic5.jl"  # 24 in total
sbatch run_large.sh "final_train5_o2_TaskB3/solvers_Tsit5&DP5_cyclic6.jl"  # 24 in total
```
* **[2020-12-17] Actually the long training maybe due to too many tstops, changed and reruning**
* **[2020-12-17] It indeed improved. Waiting for the first two jobs to finish and then cyclic**
```
sbatch run_mini.sh "final_train5_o2_TaskB3/solvers_euler&RK4.jl"  # 144 in total
sbatch run_mini.sh "final_train5_o2_TaskB3/solvers_Tsit5&DP5.jl"  # 144 in total
sbatch run_large.sh "final_train5_o2_TaskB3/solvers_euler&RK4_cyclic1.jl"  # 24 in total
sbatch run_large.sh "final_train5_o2_TaskB3/solvers_euler&RK4_cyclic2.jl"  # 24 in total
```
* **[2020-12-17] All tasks A-B finished!**
