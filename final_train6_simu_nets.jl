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
    w_gold = gen_network_dev(NUM_NODES, (0, 1e3), 0.8, false, false); heatmap(w_gold)
    conditions, sol_golds, ode_golds = get_data(w_gold, NUM_CONDITIONS+NUM_CONDITIONS_TEST)
    plot_ode(w_gold, conditions[2, :], f, ode_golds[2], "$(EXPR_NAME): û (t) before training", ts)
end
# Training
w = gen_network_dev(NUM_NODES, (0, 1e-1), 0.2, false, false); heatmap(w)
plot_ode(w, conditions[2, :], f, ode_golds[2], "$(EXPR_NAME): û (t) before training", ts)

eigen(w_gold[:,2:end]).values
eigen(zeros(20,20)).values
w = gen_network_dev(NUM_NODES, (0, 2.), 0.8, false, true); heatmap(w)



function gen_network_dev(m, weight_params=(0.,1.), sparsity=0.,
                    acyclic=false, symmetric=false)
    w = rand(Normal(weight_params[1], weight_params[2]), (m, m))
    items = [0, 1]
    p = [sparsity, 1-sparsity]
    mask = sample(items, weights(p), (m,m), replace=true)
    w .*= mask
    @assert acyclic * symmetric != true
    if acyclic
        for i in 1:m
            w[i, i:end] .= 0
        end
    end
    if symmetric
        for i in 1:m
            w[i, i:end] .= w[i:end, i]
        end
    end

    α = abs.(rand(m) * 0.2 .+ 0.9)
    # α = ones(m)
    # w .-= diag(abs.(α))
    # w .+= diagm(abs.(α))
    # w .+= diagm(-ones(m))
    return hcat(α, w)
end

function f(du, u, p, t, envelope=tanh)
    # u = x + μ
    x = view(u, 1:NUM_NODES)
    μ = view(u, (NUM_NODES+1):2*NUM_NODES)
    α = view(p, :, 1)
    w = view(p, :, 2:NUM_NODES+1)
    # du .= vcat(envelope.(w*x-μ) - α.*x, μ)
    du[1:NUM_NODES] .= α.*envelope.(w*x-μ) - x
end

for i in [1,5,10,20,50]
    # sol_u = sol_golds[1][end]
    sol_u = sol_golds[1](i)
    x = sol_u[1:20]
    μ = sol_u[21:40]
    α = w_gold[:, 1]
    w = w_gold[:, 2:NUM_NODES+1]

    (tanh.(w * x - μ) - x./α)
    du = zeros(40)
    f(du, sol_u, w_gold, nothing)
    println(du, "\n")
end
sol_u
