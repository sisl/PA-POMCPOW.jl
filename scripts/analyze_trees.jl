using Revise #REMOVE

includet("../src/ResourcePOMDP.jl")
using .ResourcePOMDP
using POMDPs
using POMDPSimulators

using D3Trees
using Plots #REMOVE?
using Statistics

using POMCPOW
using BasicPOMCP

N_TRIALS = 100
N_QUERIES = 500

grid_size = 50
max_steps = 10
delta = 3
n_initial = 2
true_cl = 4.0*(grid_size/50.)
belief_cl = 4.0*(grid_size/50.)
m = GaussianPOMDP(grid_size, max_steps, delta, n_initial, true_cl)
ds0 = POMDPs.initialstate_distribution(m)
up = GaussianProcessUpdater(grid_size, Float64, 0.0, belief_cl)

C = [0:0.1:2.0;]
action_selector = ActionSelector{CartesianIndex{2}}(action_values, C) #, ResourcePOMDP.gp_prune)

solver = AEPOMCPOWSolver(tree_queries=N_QUERIES, check_repeat_obs=true, k_observation=5,
check_repeat_act=true, next_action=action_selector, belief_updater=up, max_depth=25)
aepomcpow_planner = ResourcePOMDP.solve(solver, m)

solver = POMCPOWSolver(tree_queries=N_QUERIES, check_repeat_obs=true, k_observation=5,
check_repeat_act=true, max_depth=25)
pomcpow_planner = POMDPs.solve(solver, m)

solver = POMCPSolver(tree_queries=N_QUERIES)
pomcp_planner = POMDPs.solve(solver, m)

function max_height(tree::Array; d::Int64=1, offset::Int64=0)
    children = tree[d]
    max_depth = 0
    for child in children
        d = child - offset
        depth = max_height(tree, d=d, offset=offset)
        if depth > max_depth
            max_depth = depth
        end
    end
    max_depth = max_depth + 1
    return max_depth
end

aepomcpow_results = []
pomcpow_results = []
pomcp_results = []

for i=1:N_TRIALS
    t1 = time_ns()
    s0 = ResourcePOMDP.rand(ds0)
    b0 = initialize_belief(up, s0)

    ##### PA-POMCPOW #####
    a, info = ResourcePOMDP.action_info(aepomcpow_planner, b0, tree_in_info=true)
    d_tree = D3Tree(info[:tree], init_expand=1)
    offset = d_tree.children[1][1] - 2

    height = max_height(d_tree.children, offset=offset)
    push!(aepomcpow_results, height)
    ##### POMCPOW #####
    a, info = POMCPOW.action_info(pomcpow_planner, b0, tree_in_info=true)
    d_tree = D3Tree(info[:tree], init_expand=1)
    offset = d_tree.children[1][1] - 2

    height = max_height(d_tree.children, offset=offset)
    push!(pomcpow_results, height)
    # ##### POMCP #####
    # a, info = BasicPOMCP.action_info(pomcp_planner, b0, tree_in_info=true)
    # d_tree = D3Tree(info[:tree], init_expand=1)
    # offset = d_tree.children[1][1] - 2
    #
    # height = max_height(d_tree.children, offset=offset)
    # push!(pomcp_results, height)
    t2 = time_ns()
    elapsed = (t2 - t1)*1e-9
    println("Trial $i done after $elapsed sec")
end

mean_aepomcpow = mean(aepomcpow_results)
std_aepomcpow = std(aepomcpow_results)
se_aepomcpow = std_aepomcpow/sqrt(N_TRIALS)

mean_pomcpow = mean(pomcpow_results)
std_pomcpow = std(pomcpow_results)
se_pomcpow = std_pomcpow/sqrt(N_TRIALS)

# mean_pomcp = mean(pomcp_results)
# std_pomcp = std(pomcp_results)
# se_pomcp = std_pomcp/sqrt(N_TRIALS)
