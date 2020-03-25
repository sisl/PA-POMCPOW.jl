using Revise #REMOVE

includet("../src/ResourcePOMDP.jl")
using .ResourcePOMDP
using POMDPs
using POMDPSimulators

using D3Trees
using Plots #REMOVE?

grid_size = 50
max_steps = 10
delta = 3
n_initial = 2
true_cl = 4.0*(grid_size/50.)
belief_cl = 4.0*(grid_size/50.)
m = GaussianPOMDP(grid_size, max_steps, delta, n_initial, true_cl)
ds0 = POMDPs.initialstate_distribution(m)
s0 = ResourcePOMDP.rand(ds0)
up = GaussianProcessUpdater(grid_size, Float64, 0.0, belief_cl)
b0 = initialize_belief(up, s0)


C = [0:0.1:2.0;]

action_selector = ActionSelector{CartesianIndex{2}}(action_values, C) #, ResourcePOMDP.gp_prune)

solver = AEPOMCPOWSolver(tree_queries=1000, check_repeat_obs=true,k_observation=5,
check_repeat_act=true, next_action=action_selector, belief_updater=up, max_depth=25)
planner = ResourcePOMDP.solve(solver, m)

# mu_sigma = GaussianProcessSolve(b0)
# mu = reshape_gp_samples(mu_sigma[1], CartesianIndices((1:grid_size, 1:grid_size)), grid_size)
# fig = heatmap(mu, title="Initial Belief Mean", fill=true, clims=(-3,3), xlims = (0,grid_size), ylims = (0,grid_size))
# display(fig)
# #
# sig = reshape_gp_samples(mu_sigma[2], CartesianIndices((1:grid_size, 1:grid_size)), grid_size)
# fig = heatmap(sig, title="Initial Belief Variance", fill=true, clims=(0,1), xlims = (0,grid_size), ylims = (0,grid_size))
# display(fig)

function plot_actions(acts)
    true_acts = zeros(Float64, 2, 0)
    false_acts = zeros(Float64, 2, 0)
    for act in acts
        act_array = Float64[act[2][1]; act[2][2]]
        if act[1]
            true_acts = [true_acts  act_array]
        else
            false_acts = [false_acts act_array]
        end
    end
    return (true_acts, false_acts)
end

# b_out = nothing
# s_out = nothing
# r_total = 0
# for (s, a, r, bp, t) in stepthrough(m, planner, up, b0, s0, "s,a,r,bp,t", max_steps=100)
#     @show a
#     @show r
#     @show t
#     mu_sigma = GaussianProcessSolve(bp)
#     global grid_size
#     mu = reshape_gp_samples(mu_sigma[1], CartesianIndices((1:grid_size, 1:grid_size)), grid_size)
#     heatmap(mu,title="Belief Means Step $t", fill=true, clims=(-3,3), xlims = (0,grid_size), ylims = (0,grid_size))
#     action_vals = action_values(m, bp)
#     # acts = ResourcePOMDP.embed_actions(action_vals, C, 50, false)
#     acts = ResourcePOMDP.embed_actions(action_vals, 1.0, 20, true)
#
#     true_acts, false_acts = plot_actions(acts)
#     fig = scatter!(true_acts[2,:], true_acts[1,:], legend=false, markershape=:hexagon)
#     fig = scatter!(false_acts[2,:], false_acts[1,:], legend=false, markershape=:square)
#     display(fig)
#     global b_out
#     b_out = bp
#     global s_out
#     s_out = s
#     global r_total
#     r_total += r*discount(m)^(t-1)
# end
# println("Total Reward: $r_total")
# heatmap(s0.full,title="Sensor Placement Field", fill=true, clims=(-3,3), xlims = (0,grid_size), ylims = (0,grid_size), border=:none)
# fig = scatter!(b_out.x_obs[2,:], b_out.x_obs[1,:], label="Observation", markershape=:circle)
# fig = scatter!(b_out.x_act[2,:], b_out.x_act[1,:], label="Action", markershape=:square)
# savefig(fig, "./data/truth.png")
# display(fig)
a, info = ResourcePOMDP.action_info(planner, b0, tree_in_info=true)
# inbrowser(D3Tree(info[:tree], init_expand=1), "firefox")

tree = info[:tree]
o_nodes = tree.o_child_lookup

d_tree = D3Tree(tree, init_expand=1)
inbrowser(d_tree, "firefox")
# for (k,v) in o_nodes
#     println(k)
# end
