include("../src/ResourcePOMDP.jl")
using .ResourcePOMDP

using POMDPs
using POMDPSimulators
using BasicPOMCP
using POMCPOW

using DataFrames
using CSV
using JLD
using Random
using Distributed
using Statistics

#GLOBAL PARAMS
N_WORKERS = 8
START_ID = 1

#Common Environment Params
max_steps = 10
n_initial = 2
rng = MersenneTwister(1984)

config = CSV.File("./data/gp_config.csv", skipto=START_ID+1)
results_string = "./data/gp_results.csv"
for row in config
    n_runs = row.Trials
    grid_size = row.Grid
    delta = row.delta
    cl = row.l
    queries = row.Queries

    pomdp = GaussianPOMDP(grid_size, max_steps, delta, n_initial, cl)
    up = GaussianProcessUpdater(grid_size, Float64, 0.0, cl)
    ro = RolloutSimulator(rng, 50)

    dat = "./data/gp_$grid_size.jld"
    if dat in readdir("./data/")
        s0_array = load(dat, "S")
    else
        ds0 = POMDPs.initialstate_distribution(pomdp)
        s0_array = []
        for i=1:n_runs;
            s0 = Base.rand(ds0)
            push!(s0_array, s0)
        end
        save(dat, "S", s0_array)
    end
    pool = Distributed.WorkerPool([1:N_WORKERS;])

    solver_string = row.Solver
    if solver_string == "BG-POMCPOW"
        action_selector = ActionSelector{CartesianIndex{2}}(action_values, [0:0.01:1.0;])
        solver = AEPOMCPOWSolver(tree_queries=queries, check_repeat_obs=true, max_depth=25,
        check_repeat_act=true, next_action=action_selector, belief_updater=up)
        planner = solve(solver, pomdp)
    elseif solver_string == "POMCPOW"
        solver = POMCPOWSolver(tree_queries=queries, max_depth=25,
        check_repeat_obs=true, check_repeat_act=true)
        planner = solve(solver, pomdp)
    elseif solver_string == "POMCP"
        solver = POMCPSolver(tree_queries=queries, max_depth=25)
        planner = solve(solver, pomdp)
    elseif solver_string == "Greedy"
        planner = GreedyPolicy(pomdp)
    else
        println("SOLVER TYPE NOT RECOGNIZED!")
    end
    queue = []
    for i=[1:n_runs;]
        s0 = s0_array[i]
        b0 = POMDPs.initialize_belief(up, s0)
        sim = POMDPSimulators.POMDPSim(ro, pomdp, planner, up, b0, s0, (test=nothing,))
        push!(queue, sim)
    end
    println("Starting Runs for $solver_string, grid size $grid_size, $queries queries")
    time = @elapsed rewards = run_parallel(POMDPSimulators.default_process, queue, pool, show_progress=true).reward
    println("$solver_string, grid size $grid_size, $queries queries Complete!")
    mu = mean(rewards)
    sig = std(rewards)
    df = DataFrame(id=row.ID, solver=solver_string, queries=queries, grid=grid_size, mu=mu, sig=sig, t=time)
    print("DATA FRAME")
    CSV.write(results_string, df, append=true)
    println("Results Saved!")
end
