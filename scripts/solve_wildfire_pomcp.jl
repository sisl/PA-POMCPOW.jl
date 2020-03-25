using Revise
using Random

includet("../src/ResourcePOMDP.jl")
using .ResourcePOMDP

using POMDPs
using POMDPModelTools
using POMDPSimulators
using BasicPOMCP


########## PARAMS ##########
AVG_FUEL = 18.0
BURN_THRESH = 15.0
GRID_DIM = 50
init_counters = [1000,1000,1000,1000]
area_sizes = [5,5,5,5]
containment_dim = 1

jumpsteps = 1
max_steps = 7000   # large number runs until terminal state is reached. Note that terminal state might never be reached (sometimes, the solver manages to "block" the fire to enter the last corner); therefore, it is a good idea to terminate after a certain timestep.
tree_queries = 200
############################

m = WildfirePOMDP(AVG_FUEL, GRID_DIM, BURN_THRESH, jumpsteps, area_sizes, containment_dim, false)
ds0 = POMDPs.initialstate_distribution(m)
s0 = rand(ds0)  # this will be the ground truth.
up = WildfireBeliefUpdater(GRID_DIM, Float64, AVG_FUEL, belief_cl, max_steps)
b0 = initialize_belief(up, s0)
#
solver = POMCPSolver(tree_queries=tree_queries)
planner = solve(solver, m)
#
global gb, gs, s0
for (s, a, r, o,b, t) in stepthrough(m, planner, up, b0, s0, "s,a,r,o,b,t", max_steps=max_steps)
    # @show s
    # @show a
    # @show r
    # @show t
    global gs=s
    global gb=b
    obs = o[1]
    println("At timestep $t, observed $obs after action $a is taken")
    if r <-1 @show r end  # print r if entered a keepout zone

    # display(plot_state(s)[3])
    mu,sigma = GaussianProcessSolve(b)
    p0 = heatmap(reshape(reshape(mu,b.dim,b.dim)', GRID_DIM, GRID_DIM),xlims=(0,GRID_DIM), ylims=(0,GRID_DIM), clims=(0,40), c=:algae_r)
    p1 = plot_state(s)[1]
    plot!(p0, gb.x_obs[2,:], b.x_obs[1,:], title="Belief of Fuel Map", seriestype=:scatter, legend=false)
    plot!(p1, gb.x_obs[2,:], b.x_obs[1,:], title="Burn Map", seriestype=:scatter, legend=false)
    p2 = plot_state(s)[2]
    display(plot(p0,p1,p2))

    # savefig_recursive(p3, "wildfire", false)
    # savefig_recursive(heatmap(s.bm), "wildfire_state", false)
end
