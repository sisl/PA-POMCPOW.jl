
struct GaussianProcessState
    x_obs::Array{Float64, 2}
    x_act::Array{Float64, 2}
    y::Array{Float64, 1}
    full::Union{Array{Float64}, Nothing}
end

struct GaussianProcessBelief
    dim::Int #Grid size (e.g. 25 = gridworld 25x25)
    x_obs::Array
    x_act::Union{Nothing, Array}
    y::Array
    m::AbstractFloat
    l::AbstractFloat
    gp::GPE
    function GaussianProcessBelief(dim::Int, x_obs::Array, x_act::Union{Nothing, Array}, y::Array, m::Real=0, l::AbstractFloat=0.)
        gp = GPE(x_obs, y, MeanConst(m), LEIso(l,0.), -3.0)
        new(dim, x_obs, x_act, y, m, l, gp)
    end
end

function GaussianProcessBelief(dim::Int, dtype::Type=Float64, m::Float64=0.0, l::AbstractFloat=0.0)
    x_obs = Array{dtype}(undef, 2, 0)
    y = Array{dtype}(undef, 0)
    GaussianProcessBelief(dim, x_obs, nothing, y, m, l)
end

struct GaussianProcessUpdater <: POMDPs.Updater
    dim::Int
    dtype::Type
    m::AbstractFloat
    l::AbstractFloat
end

function POMDPs.update(b::GaussianProcessUpdater, old_b, action, obs)
    act = action[1]
    location = action[2]
    act_array = reshape(Float64[location[1] location[2]], 2, 1)
    x_obs = [old_b.x_obs act_array] # This is specific to our problem formulation where action:=location
    y = [old_b.y; obs]
    if act
        x_act = [old_b.x_act act_array]
    else
        x_act = old_b.x_act
    end
    GaussianProcessBelief(old_b.dim, x_obs, x_act, y, old_b.m, old_b.l)
end

function POMDPs.initialize_belief(up::GaussianProcessUpdater, s::GaussianProcessState)
    GaussianProcessBelief(up.dim, s.x_obs, s.x_act, s.y, up.m, up.l)
end

function POMDPs.initialize_belief(up::GaussianProcessUpdater, b::GaussianProcessBelief)
    GaussianProcessBelief(up.dim, b.x_obs, b.x_act, b.y, up.m, up.l)
end

function GaussianProcessSolve(b::GaussianProcessBelief; cov_matrix::Bool=false)
    x_out = Array{Float64}(undef, b.dim, b.dim)
    x_idxs = CartesianIndices((1:b.dim, 1:b.dim))
    x_grid = cartind_to_array(x_idxs)
    predict_y(b.gp, x_grid, full_cov=cov_matrix)
end

function GaussianProcessSolve(b::GaussianProcessBelief, x_grid::Array; cov_matrix::Bool=false)
    predict_y(b.gp, x_grid, full_cov=cov_matrix)
end

function GaussianProcessSample(b::GaussianProcessBelief)
    x_out = Array{Float64}(undef, b.dim, b.dim)
    x_idxs = CartesianIndices((1:b.dim, 1:b.dim))
    x_grid = cartind_to_array(x_idxs)
    GaussianProcesses.rand(b.gp, x_grid)
end

function POMDPs.rand(rng::AbstractRNG, b::GaussianProcessBelief)
    GaussianProcessState(b.x_obs, b.x_act, b.y, nothing)
end

###################################################
## Below, are functions for the Wildfire problem ##
###################################################

struct WildfireBelief
    dim::Int64 #Grid size (e.g. 25 = gridworld 25x25)
    burn_map::Array{Int8}  # full-observability of the burn map
    fuel_map::Array{Float64}  # full-observability of the fuel map
    wind_mean::Array{Float64}
    wind_cov_diag::Array{Float64}
    area_counters::Array{Int64}  # 4-element Array, each element is a counter for the area in the corner.
end

struct WildfireBeliefUpdater <: POMDPs.Updater
    dim::Int64
end

function POMDPs.update(b::WildfireBeliefUpdater, old_b::WildfireBelief, action::CartesianIndex{2}, obs::NamedTuple)
    ow, df, fuel_map, burn_map, ac = obs
    wm = old_b.wind_mean
    wc = old_b.wind_cov_diag
    S = wc .+ (0.1 + clamp((20.0-df)/20., 0, 2.)) #Assumes obs noise from normal with 0.1 diag cov
    K = wc./S
    wmp = wm + K.*(ow - wm)
    wcp = (1 .- K).*wc
    WildfireBelief(old_b.dim, burn_map, fuel_map, wmp, wcp, ac)
end

function POMDPs.initialize_belief(up::WildfireBeliefUpdater, s::NamedTuple)
    WildfireBelief(up.dim, s.bm, s.fm, [0.0 0.0], [2.0 2.0], s.ac)
end

function POMDPs.rand(rng::AbstractRNG, b::WildfireBelief)
    wx = rand(Normal(b.wind_mean[1], b.wind_cov_diag[1]))
    wy = rand(Normal(b.wind_mean[2], b.wind_cov_diag[2]))
    wind = clamp.([wx wy], -1., 1.)
    return (fm=b.fuel_map, bm=b.burn_map, w=wind, ac=b.area_counters)
end
