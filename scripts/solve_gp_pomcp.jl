using Revise #REMOVE

includet("../src/ResourcePOMDP.jl")
using .ResourcePOMDP
using POMDPs
using POMDPModelTools
using POMDPSimulators
using BasicPOMCP

using D3Trees
using Plots #REMOVE?

grid_size = 50
max_steps = 10
delta = 2
n_initial = 5
true_cl = 5.0*(grid_size/50.)
belief_cl = 5.0*(grid_size/50.)
m = GaussianPOMDP(grid_size, max_steps, delta, n_initial, true_cl)
ds0 = POMDPs.initialstate_distribution(m)
s0 = ResourcePOMDP.rand(ds0)
up = GaussianProcessUpdater(grid_size, Float64, 0.0, belief_cl)
b0 = initialize_belief(up, s0)

solver = POMCPSolver(tree_queries=1000)
planner = solve(solver, m)

mu_sigma = GaussianProcessSolve(b0)
display(heatmap(reshape(mu_sigma[1], grid_size, grid_size), title="Initial Belief Mean", fill=true, clims=(-3,3), xlims = (0,grid_size), ylims = (0,grid_size)))
s_out = nothing
for (s, a, r, b, t) in stepthrough(m, planner, up, b0, s0, "s,a,r,b,t", max_steps=max_steps)
    @show s
    @show a
    @show r
    @show t
    mu_sigma = GaussianProcessSolve(b)
    global grid_size
    display(heatmap(reshape(mu_sigma[1], grid_size, grid_size),title="Belief Means Step $t", fill=true, clims=(-3,3), xlims = (0,grid_size), ylims = (0,grid_size)))
    global s_out
    s_out = s
end
up = GaussianProcessUpdater(grid_size, Float64, 0.0, true_cl)
b_out = initialize_belief(up, s_out)
truth = GaussianProcessSample(b_out)
heatmap(reshape(truth, grid_size, grid_size),title="Ground Truth Realization", fill=true, clims=(-3,3), xlims = (0,grid_size), ylims = (0,grid_size))
fig = scatter!(b_out.x_obs[1,:], b_out.x_obs[2,:], legend=false)
# test = POMDPs.actions(m, b0)
# a, info = BasicPOMCP.action_info(planner, b0, tree_in_info=true)
# inbrowser(D3Tree(info[:tree], init_expand=1), "firefox")
