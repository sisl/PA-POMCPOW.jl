using Revise #REMOVE

includet("../src/ResourcePOMDP.jl")
using .ResourcePOMDP
using POMDPs
# using POMDPModelTools
using POMDPSimulators
using POMCPOW

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

function POMCPOW.obs_weight(p::GaussianPOMDP, s::GaussianProcessState, a::CartesianIndex{2}, sp::GaussianProcessState, o::Float64)
    a_array = Float64[a[1], a[2]]
    obs_s = find_obs(sp, a_array)
    if obs_s == o
        return 1.
    else
        return 0.
    end
end

function find_obs(s::GaussianProcessState, x::Array{Float64})
    idx = nothing
    for i=1:size(s.x)[2]
        x_obs = s.x[:,i]
        if x_obs == x
            idx = i
            break
        end
    end
    return s.y[idx]
end

solver = POMCPOWSolver(tree_queries=100, check_repeat_obs=true, check_repeat_act=true)
planner = solve(solver, m)

# mu_sigma = GaussianProcessSolve(b0)
# fig = heatmap(reshape(mu_sigma[1], grid_size, grid_size), title="Initial Belief Mean", fill=true, clims=(-3,3), xlims = (0,grid_size), ylims = (0,grid_size))
# display(fig)
#
# fig = heatmap(reshape(mu_sigma[2], grid_size, grid_size), title="Initial Belief Variance", fill=true, clims=(0,1), xlims = (0,grid_size), ylims = (0,grid_size))
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
#     fig = heatmap(reshape(mu_sigma[1], grid_size, grid_size),title="Belief Means Step $t", fill=true, clims=(-3,3), xlims = (0,grid_size), ylims = (0,grid_size))
#     display(fig)
#     global b_out
#     b_out = bp
#     global r_total
#     r_total += r*discount(m)^(t-1)
# end
# println("Total Reward: $r_total")
# heatmap(s0.full,title="Ground Truth Realization", fill=true, clims=(-3,3), xlims = (0,grid_size), ylims = (0,grid_size))
# fig = scatter!(b_out.x_obs[1,:], b_out.x_obs[2,:], legend=false)
# # savefig(fig, "./data/truth.png")
# display(fig)
a, info = POMCPOW.action_info(planner, b0, tree_in_info=true)
inbrowser(D3Tree(info[:tree], init_expand=1), "firefox")
