
@time using Random, CSV, DataFrames, Distributions, StatsBase, Plots, DelimitedFiles, Distributed  # 240s -> 60s
@time using DifferentialEquations, ForwardDiff, Flux, DiffEqFlux  # 800s -> 240s
using Distributed; nworkers()  # use 8 for 12 core O2
ENV["GKSwstype"] = "nul" # original 100


function gen_network(m, weight_params=(0.,1.), sparsity=0., acyclic=false, )
    w = rand(Normal(weight_params[1], weight_params[2]), (m, m))
    items = [0, 1]
    p = [sparsity, 1-sparsity]
    mask = sample(items, weights(p), (m,m), replace=true)
    w .*= mask
    if acyclic
        for i in 1:m
            w[i, i:end] .= 0
        end
    end
    α = abs.(rand(Normal(weight_params[1], weight_params[2]), (m)))
    return hcat(α, w)
end

function f(du, u, p, t, envelope=tanh)
    # u = x + μ
    x = view(u, 1:NUM_NODES)
    μ = view(u, (NUM_NODES+1):2*NUM_NODES)
    α = view(p, :, 1)
    w = view(p, :, 2:NUM_NODES+1)
    du[1:NUM_NODES] .= envelope.(w*x-μ) - α.*x
end

function plot_ode(params, u0=u0, f=f, ode_gold=nothing, label="û (t)", plot_ts=0.:10)
    # variables = xμ + w
    u_prob = ODEProblem((du,u,w,t) -> f(du,u,w,t,tanh), u0, (0., ts[end]), params)
    u_sol = solve(u_prob, Tsit5())
    sol_gold = solve(ode_gold, Tsit5())
    plot(u_sol.t, [x[3] for x in u_sol.u], label="x1_hat", color=RGB(0.4, 0.5, 0.8))
    plot!(u_sol.t, [x[2] for x in u_sol.u], label="x2_hat", color=RGB(0.4, 0.7, 0.5))
    plot!(u_sol.t, [x[4] for x in u_sol.u], label="x3_hat", color=RGB(0.9, 0.6, 0.6))
    plot!(sol_gold.t, [sol_gold(t)[3] for t in sol_gold.t], label="x1", legend=true, color="blue1")
    plot!(sol_gold.t, [sol_gold(t)[2] for t in sol_gold.t], label="x2", legend=true, color="green")
    plot!(sol_gold.t, [sol_gold(t)[4] for t in sol_gold.t], label="x3", legend=true, color="red")
    scatter!(plot_ts, [sol_gold(t)[3] for t in plot_ts], label="x1 data points", color="lightblue")
    scatter!(plot_ts, [sol_gold(t)[2] for t in plot_ts], label="x2 data points", color="lightgreen")
    scatter!(plot_ts, [sol_gold(t)[4] for t in plot_ts], label="x3 data points", color="pink")
    xlabel!("ODE steps")
    title!(label)
end


function l2_loss(preds, golds)
    y = convert(Array, VectorOfArray(golds))
    x = convert(Array, VectorOfArray(preds))
    1/2 * mean(abs2, y-x)
end


# Define ground truth
function get_data(w, n)
    m = size(w)[1]
    conditions = randn((m, n))
    mask = sample([0, 1], weights([0.8, 0.2]), size(conditions), replace=true)
    conditions .*= mask
    u0 = zeros(m)
    sol_golds = Array{Any}(undef, n)
    u0s = zeros(n, 2*m)
    probs = Array{Any}(undef, n)
    for i in 1:n
        u0s[i, m+1:end] = conditions[:, i]
        probs[i] = ODEProblem((du,u,w,t) -> f(du,u,w,t,tanh), u0s[i, :], (ts[1], ts[end]), w_gold)
        sol_golds[i] = solve(probs[i], Tsit5(), saveat=ts)
    end
    return u0s, sol_golds, probs
end


# Training
function cb!()
    global counter, lr, tracked_total_loss
    counter += 1
    lr *= 1 - LEARNING_RATE_DECAY
    train_loss = ode_loss(ts, 1:NUM_CONDITIONS)
    test_loss = ode_loss(ts, 1+NUM_CONDITIONS:NUM_CONDITIONS+NUM_CONDITIONS_TEST)

    println("Training $FILEHEADER (iter=", counter, "): ", train_loss, test_loss)

    open("results/$(FILEHEADER).csv", "a") do io
       writedlm(io, hcat(counter, train_loss, test_loss), ",")
    end

    if counter%LISTENING_FREQ == 0
        plot_ode(w, conditions[2, :], f, ode_golds[2], "$(EXPR_NAME): û (t) for iteration: $(counter)", ts)
        savefig("figures/$(FILEHEADER)_train_iter_$(counter).png")
        plot_ode(w, conditions[NUM_CONDITIONS+2, :], f, ode_golds[NUM_CONDITIONS+2], "$(EXPR_NAME): û (t) for iteration: $(counter)", ts)
        savefig("figures/$(FILEHEADER)_test_iter_$(counter).png")
        # save params
        CSV.write("results/$(FILEHEADER)_params_iter_$(counter).csv", DataFrame(w))
    end
    append!(tracked_total_loss, train_loss[3])
    if counter > NUM_ITERATIONS_BURNIN
        if (mean(tracked_total_loss[end-NUM_ITERATIONS_WINDOW*2:end-NUM_ITERATIONS_WINDOW]) -
                mean(tracked_total_loss[end-NUM_ITERATIONS_WINDOW:end])) < TOLERANCE
            println(tracked_total_loss)
            Flux.stop()
    end end
    return 0
end

function ode_loss_cond(i, tp)
    _prob = remake(ode_golds[i], p=w)
    _sol = solve(_prob, Tsit5(), saveat=tp)
    l2_loss(_sol.u, sol_golds[i].u)
end

function ode_loss(tp=ts, idx=BATCH_SIZE)
    if length(idx) == 1
        idx = rand(1:NUM_CONDITIONS, BATCH_SIZE)
    end
    # estimation_loss = [ode_loss_cond(i, tp) for i in 1:NUM_CONDITIONS]
    estimation_loss = mean(ode_loss_cond(i, tp) for i in idx)
    regularization_loss = L2_LAMBDA*mean(abs2, w)
    total_loss = estimation_loss + regularization_loss
    return [estimation_loss regularization_loss total_loss]
end
