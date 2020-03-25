
struct NoisePOMDP <: POMDP{NamedTuple, CartesianIndex, Float64}
    dim::Int
    timesteps::Int
    delta::Int # Minimum distance between actions (in x and y)
    n_initial::Int # Number of initial sensor positions
    d::Normal{Float64}
    function NoisePOMDP(gridDim::Int, timesteps::Int, delta::Int, n_initial::Int)
        # May use in future to enforce constraints on dimensions vs no initial samples
        d = Normal()
        new(gridDim, timesteps, delta, n_initial, d)
    end
end

function POMDPs.gen(m::NoisePOMDP, s, a, rng)
    # transition model
    x_obs = s.x
    y_obs = s.y
    a = reshape(Float64[a[1] a[2]], 2, 1)
    o = Base.rand(m.d)
    o = clamp(o, -3, 3)
    o = round(o, digits=1)
    r = o
    xp = [x_obs a]
    yp = [y_obs; o]
    sp = GaussianProcessState(xp, yp)
    # create and return a NamedTuple
    return (sp=sp, o=o, r=r)
end

struct NoiseInitialStateDistribution
    dim::Int
    n_initial::Int
end

POMDPs.initialstate_distribution(m::NoisePOMDP) = NoiseInitialStateDistribution(m.dim, m.n_initial)

function rand(d::NoiseInitialStateDistribution)
    n = d.n_initial
    x1 = sample(Float64[1:d.dim;], n, replace=false)
    x2 = sample(Float64[1:d.dim;], n, replace=true)
    x = [reshape(x1, 1, n); reshape(x2, 1, n)]
    y = Base.rand(Normal(), d.n_initial)
    s = GaussianProcessState(x, y)
end

function Base.rand(rng::AbstractRNG, d::NoiseInitialStateDistribution)
    n = d.n_initial
    x1 = sample(Float64[1:d.dim;], n, replace=false)
    x2 = sample(Float64[1:d.dim;], n, replace=true)
    x = [reshape(x1, 1, n); reshape(x2, 1, n)]
    y = Base.rand(Normal(), d.n_initial)
    s = GaussianProcessState(x, y)
end

function POMDPs.actions(p::NoisePOMDP)
    collect(reshape(CartesianIndices((1:p.dim, 1:p.dim)), p.dim^2))
end

function POMDPs.actions(p::NoisePOMDP, s::GaussianProcessState)
    locations = Set(POMDPs.actions(p))
    for i = 1:size(s.x)[2]
        loc = convert(Array{Int}, s.x[:, i])
        setdiff!(locations, Set(CartesianIndices((loc[1]-p.delta:loc[1]+p.delta, loc[2]-p.delta:loc[2]+p.delta))))
    end
    collect(locations)
end

"""
Solver specific requirements
"""

POMDPs.discount(::NoisePOMDP) = 0.8
POMDPs.isterminal(p::NoisePOMDP, s::GaussianProcessState) = length(s.y) >= p.timesteps + p.n_initial

function action_values(p::NoisePOMDP, b::GaussianProcessBelief)
    actions = POMDPs.actions(p, b)
    action_array = CartInd_to_Array(actions)
    mu_sigma = GaussianProcessSolve(b, action_array)
    return (mu=mu_sigma[1], sig=mu_sigma[2], act=actions)
end
