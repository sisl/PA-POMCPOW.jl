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
avg_fuel = 18.0
burn_thresh = 15.0
grid_dim = 40
jumpsteps = 5
area_sizes = [5,5,5,5]
containment_dim = 1

jumpsteps = 5
max_steps = 7000
rng = MersenneTwister(1984)

config = CSV.File("./data/wildfire_config.csv", skipto=START_ID+1)
results_string = "./data/wildfire_results.csv"
for row in config
    n_runs = row.Trials
    grid_size = row.Grid
    delta = row.delta
    cl = row.l
    queries = row.Queries

    pomdp = WildfirePOMDP(avg_fuel, grid_dim, burn_thresh, jumpsteps, area_sizes, containment_dim, false)
    up = WildfireBeliefUpdater(grid_dim)
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
        action_selector = ActionSelector{CartesianIndex{2}}(action_values, [0.5:0.1:1.5;])
        solver = AEPOMCPOWSolver(tree_queries=queries, check_repeat_obs=true, max_depth=20,
        check_repeat_act=true, next_action=action_selector, belief_updater=up)
        planner = solve(solver, pomdp)
    elseif solver_string == "POMCPOW"
        solver = POMCPOWSolver(tree_queries=queries, max_depth=20,
        check_repeat_obs=true, check_repeat_act=true)
        planner = solve(solver, pomdp)
    elseif solver_string == "POMCP"
        solver = POMCPSolver(tree_queries=queries, max_depth=20)
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
