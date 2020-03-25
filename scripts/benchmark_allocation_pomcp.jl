#------------------------------------------------------------------------#
#   by     Anil Yildiz                                                   #
#  for     Stanford Intelligent Systems Laboratory (SISL)                #
#   at     Stanford University                                           #
#   on     Jan 2019                                 yildiz@stanford.edu  #
#------------------------------------------------------------------------#

"""
Reads ground truth from csv file, and solves with a custom POMCP solver that is more greedy during branching/rollouts.
"""

using Revise

includet("../src/ResourcePOMDP.jl")
using .ResourcePOMDP
using POMDPs
using POMDPModelTools
using POMDPSimulators
using BasicPOMCP

using DelimitedFiles
using Discretizers
using Random
using D3Trees
using Plots #REMOVE?
using Random

GRD = 100
no_of_scenarios = 1
CreateGroundTruthSmooth(no_of_scenarios, amount_of_scatters=50, grid_size=(GRD,GRD), limit_range=0:100)
global all_scenarios_ground_truth = ReadGroundTruth(no_of_scenarios)

for n_initial in [2, 4, 6, 8, 10]
    println("### Scenario with $n_initial sensors ###")
    scenario_ground_truth = all_scenarios_ground_truth[:,:,1]

    scenario_ground_truth = ReadGroundTruth(1)
    # savefig_recursive(heatmap(scenario_ground_truth'), "groundtruth", true)

    grid_size = size(scenario_ground_truth,1)

    max_steps = 15
    delta = 5
    # n_initial = 2
    
    true_cl = log(grid_size/10)
    belief_cl = log(grid_size/10)
    tree_queries = 10

    
    m = AllocationPOMDP(grid_size, max_steps, delta, n_initial, true_cl, scenario_ground_truth, tree_queries)
    ds0 = POMDPs.initialstate_distribution(m)
    
    Random.seed!(n_initial*grid_size)
    initial_sensor_locations = Base.rand(CartesianIndices((1:size(scenario_ground_truth,1),1:size(scenario_ground_truth,2))), n_initial)
    # initial_sensor_locations = CartesianIndex{2}[CartesianIndex(49, 24), CartesianIndex(5, 24), CartesianIndex(10, 8), CartesianIndex(18, 25), CartesianIndex(13, 26)]
    
    s0 = ResourcePOMDP.rand(ds0, scenario_ground_truth, initial_sensor_locations)
    up = GaussianProcessUpdater(grid_size, Float64, 0.0, belief_cl)
    b0 = initialize_belief(up, s0)

    solver = POMCPSolver(tree_queries=tree_queries)
    planner = solve(solver, m)

    # mu_sigma = GaussianProcessSolve(b0)
    # plt_obj = heatmap(reshape(mu_sigma[1], grid_size, grid_size), title="Initial Belief Mean", fill=true, clims=(0,100), xlims = (0,grid_size), ylims = (0,grid_size))
    # savefig_recursive(plt_obj, "belief_mean", false)

    # plt_obj = heatmap(reshape(mu_sigma[2], grid_size, grid_size), title="Initial Belief Mean", fill=true, xlims = (0,grid_size), ylims = (0,grid_size))
    # savefig_recursive(plt_obj, "belief_var", false)


    up_rollout = RolloutUpdater(grid_size, Float64, 0.0, belief_cl, 0)
    pol = RolloutPolicy(m, rng=solver.rng, updater=up_rollout)
    solved_estimator = BasicPOMCP.SolvedPORollout(pol, up_rollout, solver.rng)
    planner = POMCPPlanner(solver, m, solved_estimator, solver.rng, Int[], nothing)

    # simulate
    # covs = []
    rewards = []
    println("### Simulations Started ###")
    @show initial_sensor_locations
    for (s, a, o, r, t, b, bp) in stepthrough(m, planner, up, b0, s0, "s,a,o,r,t,b,bp", max_steps=max_steps)
        println()
        # @show s
        # @show b
        mu_sigma = GaussianProcessSolve(bp, cov_matrix=false)  # bp is the updated belief after taking action a.
        # push!(covs, mu_sigma[2])
        push!(rewards, r)
        # plt_obj_mean = heatmap(reshape(mu_sigma[1], grid_size, grid_size),title="Belief Means Step $t", fill=true, clims=(0,100), xlims = (0,grid_size), ylims = (0,grid_size))
        # plt_obj_var = heatmap(reshape(mu_sigma[2], grid_size, grid_size),title="Belief Means Step $t", fill=true, xlims = (0,grid_size), ylims = (0,grid_size))
        # plt_obj = plot(plt_obj_mean, plt_obj_var)
        # savefig_recursive(plt_obj, "belief_var", false)
        @show a
        @show o
        @show r
        println()
    end

    # writedlm("covs_scenario$itr.txt", covs)
    io = open("./ScenarioOutputs/rewards_scenario_$grid_size-$n_initial.txt", "a") 
    writedlm(io,["GRD: $grid_size"])
    writedlm(io,CartInd_to_Array(initial_sensor_locations))
    writedlm(io, rewards)
    rewards = sum(rewards)
    writedlm(io, rewards)
    close(io)
    println("Total Rewards: $rewards")
end
