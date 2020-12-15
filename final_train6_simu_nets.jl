# Setup
PATH_DIR = "/Users/byuan/Dropbox (HMS)/study/2020b.Fall.MIT.6.338.ParallelComputation_ScientificModeling/julia_scripts/final"
GRID_NAME = "network_style"
PATH_DIR = "$(PATH_DIR)/$(GRID_NAME)"
EXPR_NAME = "DAG20"
NUM_NODES = 20
NUM_CONDITIONS = 16
NUM_CONDITIONS_TEST = 0
SEED = 0

NUM_ITERATIONS = 5
NUM_ITERATIONS_BURNIN = 20
NUM_ITERATIONS_WINDOW = 10
LISTENING_FREQ = 1
TOLERANCE = 1e-4

LEARNING_RATE = 0.1
LEARNING_RATE_DECAY = 1e-2
L2_LAMBDA = 1e-1

# Main
PATH = "$(PATH_DIR)/$(EXPR_NAME)"
FILEHEADER = "$(EXPR_NAME)_seed_$(SEED)"
# [mkpath(p) for p in ["$(PATH)/figures", "$(PATH)/results"]]
# cd(PATH)

# Generate ground truth
begin
    NUM_NODES = 20
    SEED = 4; Random.seed!(SEED)
    dt = .5
    ts = 0.:dt:50
    w_gold = gen_network(NUM_NODES, (0, 1e3), 0.8, false, false); heatmap(w_gold)
    conditions, sol_golds, ode_golds = get_data(w_gold, NUM_CONDITIONS+NUM_CONDITIONS_TEST)
    plot_ode(w_gold, conditions[2, :], f, ode_golds[2], "$(EXPR_NAME): û (t) before training", ts)
end
# Training
w = gen_network(NUM_NODES, (0, 1e3), 0.2, false, false); heatmap(w)
plot_ode(w, conditions[2, :], f, ode_golds[2], "$(EXPR_NAME): û (t) before training", ts)

mean(abs2, w)/mean(abs, w)

eigen(w_gold[:,2:end]).values
eigen(zeros(20,20)).values
w = gen_network(NUM_NODES, (0, 2.), 0.8, false, true); heatmap(w)
