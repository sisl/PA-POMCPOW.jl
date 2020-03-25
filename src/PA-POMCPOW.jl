module PA-POMCPOW

using POMDPs
using POMDPModelTools
using BasicPOMCP
using POMCPOW
using POMDPPolicies
using POMDPSimulators

using GaussianProcesses
using Random
using Distributions
using StatsBase
using Parameters
using CPUTime
using Plots

import Base: abs

include("./utils/misc.jl")
export
        average,
        savefig_recursive,
        cartind_to_array,
        reshape_gp_samples,
        euclidean_dist

include("./utils/linexp_kernel.jl")
export
        LEIso

include("beliefstates.jl")
export
        GaussianProcessBelief,
        GaussianProcessUpdater,
        GaussianProcessSolve,
        GaussianProcessSample,
        GroundTruthUpdater,
        GaussianProcessState,
        RolloutPolicy,
        RolloutUpdater,
        WildfireBelief,
        WildfireBeliefUpdater

include("./pomdps/gaussianpomdp.jl")
export
        GaussianPOMDP,
        action_values,
        obs_weight

include("wildfirepomdp.jl")
export
        WildfirePOMDP,
        action_values,
        obs_weight,
        plot_state,
        action_values

include("./baselines/greedysolvers.jl")
export
        GreedyPolicy,
        action,
        GreedyUpdater

include("./baselines/pa_pomcp.jl")
export
        DAEPOMCPSolver,
        DAEPolicy,
        AEPolicy,
        DAEPOMCPPlanner

include("pa_pomcpow.jl")
export
        PAPOMCPOWSolver,
        PAPOMCPOWPlanner,
        solve

include("action_select.jl")
export
        ActionSelector,
        average,
        sigmoid,
        expo_activator,
        normalize!,
        stretch!,
        Idx_to_CartInd,
        ScenarioSimulator,
        savefig_recursive
end # module
