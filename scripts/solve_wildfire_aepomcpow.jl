using Revise
using Random
using Plots
includet("../src/ResourcePOMDP.jl")
using .ResourcePOMDP

using POMDPs
using POMDPModelTools
using POMDPSimulators
using BasicPOMCP


########## PARAMS ##########
AVG_FUEL = 18.0
BURN_THRESH = 15.0
GRID_DIM = 40
area_sizes = [5,5,5,5]
containment_dim = 1
cl = 1.0

jumpsteps = 5
max_steps = 7000   # large number runs until terminal state is reached. Note that terminal state might never be reached (sometimes, the solver manages to "block" the fire to enter the last corner); therefore, it is a good idea to terminate after a certain timestep.
tree_queries = 100
############################

m = WildfirePOMDP(AVG_FUEL, GRID_DIM, BURN_THRESH, jumpsteps, area_sizes, containment_dim, false)
ds0 = POMDPs.initialstate_distribution(m)
s0 = rand(ds0)  # this will be the ground truth.
up = WildfireBeliefUpdater(GRID_DIM)
b0 = POMDPs.initialize_belief(up, s0)


C = [0.5:0.05:1.5;]

action_selector = ActionSelector{CartesianIndex{2}}(action_values, C)

solver = AEPOMCPOWSolver(tree_queries=500, check_repeat_obs=true,
check_repeat_act=true, next_action=action_selector, belief_updater=up, max_depth=20)
planner = ResourcePOMDP.solve(solver, m)

function plot_actions(acts)
    true_acts = zeros(Float64, 2, 0)
    false_acts = zeros(Float64, 2, 0)
    for act in acts
        act_array = Float64[act[1]; act[2]]
        if act[1][1] >= 0
            true_acts = [true_acts  act_array]
        else
            false_acts = [false_acts abs.(act_array)]
        end
    end
    return (true_acts, false_acts)
end

global gb, gs, s0
println("STARTING SIMULATIONS")
for (s, a, r, o,b, t) in stepthrough(m, planner, up, b0, s0, "s,a,r,o,b,t", max_steps=max_steps)
    # @show s
    # @show a
    @show s.ac
    @show r
    @show s.w
    @show b.wind_mean
    global gs=s
    global gb=b

    # display(plot_state(s)[3])
    # mu,sigma = GaussianProcessSolve(b)
    # p0 = heatmap(reshape(reshape(mu,b.dim,b.dim)', GRID_DIM, GRID_DIM),xlims=(0,GRID_DIM), ylims=(0,GRID_DIM), clims=(0,40), c=:algae_r)
    # p1 = plot_state(s)[1]
    # plot!(p0, gb.x_obs[2,:], b.x_obs[1,:], title="Belief of Fuel Map", seriestype=:scatter, legend=false)
    # plot!(p1, gb.x_obs[2,:], b.x_obs[1,:], title="Burn Map", seriestype=:scatter, legend=false)
    # p2 = plot_state(s)[2]
    # display(plot(p0,p1,p2))
    plots = plot_state(s)
    action_vals = action_values(m, b)
    # acts = ResourcePOMDP.embed_actions(action_vals, 1., 20, true)
    acts = ResourcePOMDP.embed_actions(action_vals, C, 21, false)
    true_acts, false_acts = plot_actions(acts)

    p1 = plots[1]
    p2 = scatter!(plots[2], true_acts[2,:], true_acts[1,:], legend=false, markershape=:hexagon)
    p2 = scatter!(plots[2], false_acts[2,:], false_acts[1,:], legend=false, markershape=:square)
    fig = plot(p1, p2, layout=(1,2), legend=false)
    display(fig)
    # savefig_recursive(p3, "wildfire", false)
    # savefig_recursive(heatmap(s.bm), "wildfire_state", false)
end
