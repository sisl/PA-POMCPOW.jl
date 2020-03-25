
struct GaussianPOMDP <: POMDP{GaussianProcessState, Tuple{Bool,CartesianIndex}, Float64}
    dim::Int
    timesteps::Int
    delta::Int # Minimum distance between actions (in x and y)
    n_initial::Int # Number of initial sensor positions
    l::Float64 # Length Scale for generative Gaussian Process
    function GaussianPOMDP(gridDim::Int, timesteps::Int, delta::Int, n_initial::Int, l::Float64)
        # May use in future to enforce constraints on dimensions vs no initial samples
        new(gridDim, timesteps, delta, n_initial, l)
    end
end

function POMDPs.gen(m::GaussianPOMDP, s, a, rng)
    # transition model
    x_obs = s.x_obs
    x_act = s.x_act
    y_obs = s.y
    full = s.full
    act = a[1]
    loc = a[2]
    if isnothing(full)
        gp = GPE(x_obs, y_obs, MeanZero(), LEIso(m.l, 0.0), -3.)
        a = reshape(Float64[loc[1] loc[2]], 2, 1)
        o = Base.rand(gp, a)[1]
    else
        o = full[loc]
        a = reshape(Float64[loc[1] loc[2]], 2, 1)
    end
    o = clamp(o, -3, 3)
    o = round(o/0.1, digits=0)*0.1
    if act
        r = o
        xp_act = [x_act a]
    else
        xp_act = x_act
        r = 0
    end
    xp_obs = [x_obs a]
    yp = [y_obs; o]
    sp = GaussianProcessState(xp_obs, xp_act, yp, full)
    return (sp=sp, o=o, r=r)
end

struct GaussianInitialStateDistribution
    dim::Int
    n_initial::Int
    l::Float64
    gp::GPE
    function GaussianInitialStateDistribution(dim::Int, n_initial::Int, l::Float64)
        x = Array{Float64}(undef, 2, 0)
        y = Array{Float64}(undef, 0)
        gp = GPE(x, y, MeanZero(), LEIso(l, 0.0), -3.)
        new(dim, n_initial, l, gp)
    end
end

POMDPs.initialstate_distribution(m::GaussianPOMDP) = GaussianInitialStateDistribution(m.dim, m.n_initial, m.l)

function Base.rand(d::GaussianInitialStateDistribution)
    n = d.n_initial
    idxs = CartesianIndices((1:d.dim, 1:d.dim))
    idx_array = cartind_to_array(idxs)
    full = Base.rand(d.gp, idx_array)
    full = clamp.(full, -3, 3)
    full = round.(full./0.1, digits=0)*0.1
    full = reshape_gp_samples(full, idxs, d.dim)
    x_idxs = rand(idxs[:], d.n_initial)
    x = cartind_to_array(x_idxs)
    y = full[x_idxs]
    x_act = zeros(Float64, 2, 0)
    s = GaussianProcessState(x, x_act, y, full)
end

function Base.rand(rng::AbstractRNG, d::GaussianInitialStateDistribution)
    n = d.n_initial
    idxs = CartesianIndices((1:d.dim, 1:d.dim))
    idx_array = cartind_to_array(idxs)
    full = Base.rand(d.gp, idx_array)
    full = clamp.(full, -3, 3)
    full = round.(full./0.1, digits=0)*0.1
    full = reshape_gp_samples(full, idxs, d.dim)
    x_idxs = rand(idxs[:], d.n_initial)
    x = cartind_to_array(x_idxs)
    y = full[x_idxs]
    x_act = zeros(Float64, 2, 0)
    s = GaussianProcessState(x, x_act, y, full)
end

function POMDPs.actions(p::GaussianPOMDP)
    locs = CartesianIndices((1:p.dim, 1:p.dim))[:]
    acts = []
    for i=1:length(locs)
        loc = locs[i]
        push!(acts, (true, loc))
        push!(acts, (false, loc))
    end
    return acts
end

function POMDPs.actions(p::GaussianPOMDP, s::GaussianProcessState)
    actions = Set(POMDPs.actions(p))
    for i = 1:size(s.x_act)[2]
        loc = convert(Array{Int}, s.x_act[:, i])
        idxs = CartesianIndices((loc[1]-p.delta:loc[1]+p.delta, loc[2]-p.delta:loc[2]+p.delta))[:]
        true_set = Set(collect(((true, item) for item in idxs)))
        false_set = Set(collect(((false, item) for item in idxs)))
        diff_set = union(true_set, false_set)
        setdiff!(actions, diff_set)
    end
    collect(actions)
end

function POMDPs.actions(p::GaussianPOMDP, b::GaussianProcessBelief)
    actions = Set(POMDPs.actions(p))
    for i = 1:size(b.x_act)[2]
        loc = convert(Array{Int}, b.x_act[:, i])
        idxs = CartesianIndices((loc[1]-p.delta:loc[1]+p.delta, loc[2]-p.delta:loc[2]+p.delta))[:]
        true_set = Set(collect(((true, item) for item in idxs)))
        false_set = Set(collect(((false, item) for item in idxs)))
        diff_set = union(true_set, false_set)
        setdiff!(actions, diff_set)
    end
    collect(actions)
end
"""
Solver specific requirements
"""

POMDPs.discount(::GaussianPOMDP) = 0.9
POMDPs.isterminal(p::GaussianPOMDP, s::GaussianProcessState) = (size(s.x_act)[2] >= p.timesteps || size(s.x_obs)[2] >= 50)

function action_values(p::GaussianPOMDP, b::GaussianProcessBelief)
    actions = POMDPs.actions(p, b)
    locations = []
    for item in actions
        if item[1]
            push!(locations, item[2])
        end
    end
    action_array = cartind_to_array(locations)
    mu_sigma = GaussianProcessSolve(b, action_array)
    mu = mu_sigma[1]
    sig = sqrt.(mu_sigma[2])
    delta_true = sig.*POMDPs.discount(p)
    delta_false = -maximum(mu)*(1 - POMDPs.discount(p))
    mus = Float64[]
    sigs = Float64[]
    actions = []
    for i=1:length(mu)
        ##### Actions #####
        push!(mus, mu[i] - delta_true[i])
        push!(sigs, sig[i])
        push!(actions, (true, locations[i]))
        ##### Observations #####
        push!(mus, delta_false)
        push!(sigs, sig[i])
        push!(actions, (false, locations[i]))
    end
    return (mu=mus, sig=sigs, act=actions)
end

function POMDPModelTools.obs_weight(p::GaussianPOMDP, s::GaussianProcessState, a::Tuple{Bool,CartesianIndex{2}}, sp::GaussianProcessState, o::Float64)
    act, loc = a
    a_array = Float64[loc[1], loc[2]]
    obs_s = find_obs(sp, a_array)
    if obs_s == o
        return 1.
    else
        return 0.
    end
end

function find_obs(s::GaussianProcessState, x::Array{Float64})
    idx = nothing
    for i=1:size(s.x_obs)[2]
        x_obs = s.x_obs[:,i]
        if x_obs == x
            idx = i
            break
        end
    end
    return s.y[idx]
end

function gp_prune(acts::Array)
    d_threshold = 1.
    acts_out = []
    acts_array = zeros(Float64, 2, 0)
    for act in acts
        act_array = Float64[act[1], act[2]]
        if length(acts_array) != 0
            d_array = sqrt.((acts_array .- act_array).^2)
            min_d = minimum(d_array)
        else
            min_d = Inf
        end
        if min_d >= d_threshold
            push!(acts_out, act)
            acts_array = [acts_array act_array]
        end
    end
    return acts_out
end
