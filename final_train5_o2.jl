# Setup
PATH_DIR = "/n/data1/hms/cellbio/sander/bo/julia/final"
GRID_NAME = "network_data_size"
PATH_DIR = "$(PATH_DIR)/$(GRID_NAME)"
# EXPR_NAME = "Baseline_DAG20"
# NUM_NODES = 20
# NUM_CONDITIONS = 64
NUM_CONDITIONS_TEST = 32

BATCH_SIZE = 8
NUM_ITERATIONS = 1000
NUM_ITERATIONS_BURNIN = 200
NUM_ITERATIONS_WINDOW = 50
LISTENING_FREQ = 20
TOLERANCE = 1e-4

LEARNING_RATE = 0.1
LEARNING_RATE_DECAY = 2e-3
L2_LAMBDA = 1e-4


# Main
for n in [64, 128, 256, 1024]
    global NUM_CONDITIONS
    NUM_CONDITIONS = n

    for m in [50, 5, 10, 100, 20]
        global EXPR_NAME, NUM_NODES
        EXPR_NAME = "Scalability_DAG$(m)_n$(n)"
        NUM_NODES = m

        for i in 0:6
            t = @elapsed begin
                global SEED, PATH, FILEHEADER, NUM_NODES, NUM_CONDITIONS
                global w_gold, ts, conditions, sol_golds, ode_golds,
                            iters, w, counter, lr, tracked_total_loss, opt
                SEED = i
                PATH = "$(PATH_DIR)/$(EXPR_NAME)"
                FILEHEADER = "$(EXPR_NAME)_seed_$(SEED)"
                [mkpath(p) for p in ["$(PATH)/figures", "$(PATH)/results"]]
                cd(PATH)

                # Generate ground truth
                Random.seed!(SEED)
                w_gold = gen_network(NUM_NODES, (0, 1.), 0.8, true)
                CSV.write("results/$(FILEHEADER)_params_ground_truth.csv", DataFrame(w_gold))

                dt = 2.
                ts = 0.:dt:10
                conditions, sol_golds, ode_golds = get_data(w_gold, NUM_CONDITIONS+NUM_CONDITIONS_TEST)

                # Training
                iters = Iterators.repeated((0.), NUM_ITERATIONS)
                w = gen_network(NUM_NODES, (0, 1e-2), 0., false)
                plot_ode(w, conditions[2, :], f, ode_golds[2], "$(EXPR_NAME): û (t) before training", ts)
                savefig("figures/$(FILEHEADER)_iter_before_training.png")
                lr = copy(LEARNING_RATE)  # to refresh lr decay
                tracked_total_loss = []
                counter = -1 ; cb!()
                opt = ADAM(lr)
                Flux.train!(tmp->ode_loss(ts)[3], Flux.params([w]), iters, opt, cb=cb!)
            end

            CSV.write("results/$(FILEHEADER)_params_final_estimation.csv", DataFrame(w))
            train_loss = ode_loss(ts, 1:NUM_CONDITIONS)
            test_loss = ode_loss(ts, 1+NUM_CONDITIONS:NUM_CONDITIONS+NUM_CONDITIONS_TEST)

            open("../$(GRID_NAME)_summary.csv", "a") do io
               writedlm(io, [GRID_NAME EXPR_NAME NUM_NODES NUM_CONDITIONS SEED train_loss[1] train_loss[3] test_loss[1] test_loss[3] counter t], ",")
            end
        end
    end
end
