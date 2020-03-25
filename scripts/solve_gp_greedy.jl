using Revise #REMOVE

includet("../src/ResourcePOMDP.jl")
using .ResourcePOMDP
using POMDPs
using POMDPSimulators
using Plots #REMOVE?

grid_size = 50
max_steps = 10
delta = 2
n_initial = 2
true_cl = 5.0*(grid_size/50.)
belief_cl = 5.0*(grid_size/50.)

m = GaussianPOMDP(grid_size, max_steps, delta, n_initial, true_cl)
ds0 = POMDPs.initialstate_distribution(m)
s0 = ResourcePOMDP.rand(ds0)
up = GaussianProcessUpdater(grid_size, Float64, 0.0, belief_cl)
b = initialize_belief(up, s0)
policy = GreedyPolicy(m)

mu_sigma = GaussianProcessSolve(b)
heatmap(reshape(mu_sigma[1], grid_size, grid_size), title="Initial Belief Mean", fill=true, clims=(-3,3), xlims = (0,grid_size), ylims = (0,grid_size), border=:none)
display(scatter!(b.x_obs[1,:], b.x_obs[2,:], legend=false))
heatmap(s0.full,title="Ground Truth Realization", fill=true, clims=(-3,3), xlims = (0,grid_size), ylims = (0,grid_size))
fig = scatter!(b.x_obs[1,:], b.x_obs[2,:], legend=false)
display(fig)
b_out = nothing
r_total = 0
for (s, a, r, bp, t) in stepthrough(m, policy, up, b, s0, "s,a,r,bp,t", max_steps=max_steps)
    @show a
    @show r
    @show t
    global grid_size
    mu_sigma = GaussianProcessSolve(bp)
    mu = reshape_gp_samples(mu_sigma[1], CartesianIndices((1:grid_size, 1:grid_size)), grid_size)
    heatmap(mu,title="Belief Means Step $t", fill=true, clims=(-3,3), xlims = (0,grid_size), ylims = (0,grid_size))
    display(scatter!(bp.x_obs[2,:], bp.x_obs[1,:], legend=false))
    # display(heatmap(reshape(mu_sigma[2], grid_size, grid_size),title="Belief Vars Step $t", fill=true, clims=(0,1), xlims = (0,grid_size), ylims = (0,grid_size)))
    max_var = maximum(mu_sigma[2])
    println("Max Var Step $t: $max_var")
    global b_out
    b_out = bp
    global r_total
    r_total += r*discount(m)^(t-1)
end
println("Total Reward: $r_total")
heatmap(s0.full,title="Ground Truth Realization", fill=true, clims=(-3,3), xlims = (0,grid_size), ylims = (0,grid_size))
fig = scatter!(b_out.x_obs[2,:], b_out.x_obs[1,:], legend=false)
display(fig)
# # savefig(fig, "./data/test.png")
