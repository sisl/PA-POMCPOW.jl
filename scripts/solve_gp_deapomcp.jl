using Revise #REMOVE

includet("../src/ResourcePOMDP.jl")
using .ResourcePOMDP
using POMDPs
using POMDPModelTools
using POMDPSimulators

using D3Trees
using Plots #REMOVE?

grid_size = 50
max_steps = 10
delta = 2
n_initial = 5
true_cl = 4.0*(grid_size/50.)
belief_cl = 4.0*(grid_size/50.)
m = GaussianPOMDP(grid_size, max_steps, delta, n_initial, true_cl)
ds0 = POMDPs.initialstate_distribution(m)
s0 = ResourcePOMDP.rand(ds0)
up = GaussianProcessUpdater(grid_size, Float64, 0.0, belief_cl)
b0 = initialize_belief(up, s0)

solver = DAEPOMCPSolver(belief_updater=up, tree_queries=1000, n_embeddings=60,
c=0.25, sample_acts=true)
planner = solve(solver, m)

# mu_sigma = GaussianProcessSolve(b0)
# heatmap(reshape(mu_sigma[1], grid_size, grid_size), title="Initial Belief Mean", fill=true, clims=(-3,3), xlims = (0,grid_size), ylims = (0,grid_size))
#
# action_vals = action_values(m, b0)
# acts = ResourcePOMDP.embed_actions(action_vals, planner.C, planner.solver.n_embeddings, planner.solver.sample_acts)
# acts = ResourcePOMDP.CartInd_to_Array(acts)
#
# fig = scatter!(acts[1,:], acts[2,:], legend=false)
# savefig(fig, "./data/mean_0.png")
# display(fig)
#
# heatmap(reshape(mu_sigma[2], grid_size, grid_size), title="Initial Belief Variance", fill=true, clims=(0,1), xlims = (0,grid_size), ylims = (0,grid_size))
# fig = scatter!(acts[1,:], acts[2,:], legend=false)
# savefig(fig, "./data/var_0.png")
# display(fig)
#
# b_out = nothing
# r_total = 0
# for (s, a, r, bp, t) in stepthrough(m, planner, up, b0, s0, "s,a,r,bp,t", max_steps=max_steps)
#     @show a
#     @show r
#     @show t
#     mu_sigma = GaussianProcessSolve(bp)
#     global grid_size
#     heatmap(reshape(mu_sigma[1], grid_size, grid_size),title="Belief Means Step $t", fill=true, clims=(-3,3), xlims = (0,grid_size), ylims = (0,grid_size))
#     action_vals = action_values(m, bp)
#     acts = ResourcePOMDP.embed_actions(action_vals, planner.C, planner.solver.n_embeddings, planner.solver.sample_acts)
#     acts = ResourcePOMDP.CartInd_to_Array(acts)
#
#     fig = scatter!(acts[1,:], acts[2,:], legend=false)
#     savefig(fig, "./data/belief_$t.png")
#     display(fig)
#     global b_out
#     b_out = bp
#     global r_total
#     r_total += r*discount(m)^(t-1)
# end
# println("Total Reward: $r_total")
# heatmap(s0.full,title="Ground Truth Realization", fill=true, clims=(-3,3), xlims = (0,grid_size), ylims = (0,grid_size))
# fig = scatter!(b_out.x_obs[1,:], b_out.x_obs[2,:], legend=false)
# savefig(fig, "./data/truth.png")
# display(fig)
a, info = ResourcePOMDP.action_info(planner, b0, tree_in_info=true)
inbrowser(D3Tree(info[:tree], init_expand=1), "firefox")
