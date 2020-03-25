
struct KeepOutAreas
    # Fields are quadrants in the coordinate system. Values are the CartesianIndices of the positions covered by the corner areas.
    i::CartesianIndices
    ii::CartesianIndices
    iii::CartesianIndices
    iv::CartesianIndices
    mids::AbstractArray

    function KeepOutAreas(i_dim::Int64, ii_dim::Int64, iii_dim::Int64, iv_dim::Int64, total_dim::Int64)
        mids = [CartesianIndex(total_dim-Int(round(i_dim/2)), Int(round(i_dim/2))),
            CartesianIndex(Int(round(ii_dim/2)), Int(round(ii_dim/2))),
            CartesianIndex(Int(round(iii_dim/2)), total_dim-Int(round(iii_dim/2))),
            CartesianIndex(total_dim - Int(round(iv_dim/2)), total_dim - Int(round(iv_dim/2)))]
        new(CartesianIndices((total_dim-i_dim+1:total_dim, 1:i_dim)),                           # quadrant I
            CartesianIndices((1:ii_dim, 1:ii_dim)),                                             # quadrant II
            CartesianIndices((1:iii_dim, total_dim-iii_dim+1:total_dim)),                       # quadrant III
            CartesianIndices((total_dim-iv_dim+1:total_dim, total_dim-iv_dim+1:total_dim)),
            mids
            )     # quadrant IV
    end
end

mutable struct WildfirePOMDP <: POMDP{NamedTuple, CartesianIndex{2}, NamedTuple}
    # To keep things simple, the action type is still CartesianIndex. Negative index means just observe and don't burn. I.e., a1 = CartesianIndex(4,7) will burn, a2 = CartesianIndex(-4,-7) will just observe. Note that -a2 is a legal syntax, and is equal to a1.
    avg_fuel::Float64
    min_fuel::Float64
    max_fuel::Float64
    grid_dim::Int64
    burn_threshold::Float64
    burn_rate::Float64
    arc_shape::Bool
    area_indices::KeepOutAreas
    area_sizes::Array{Int64}
    containment_dim::Int64
    jumpsteps::Int64
    rng::AbstractRNG
    function WildfirePOMDP(avg_fuel::Float64, grid_dim::Int64, burn_threshold::Float64, jumpsteps::Int64, area_sizes::Array{Int64}, containment_dim::Int64, arc_shape::Bool, rng::AbstractRNG=Random.GLOBAL_RNG)
        min_fuel = avg_fuel - 3.0
        max_fuel = avg_fuel + 3.0
        A1,A2,A3,A4 = area_sizes
        new(avg_fuel, min_fuel, max_fuel, grid_dim, burn_threshold, 1., arc_shape, KeepOutAreas(A1,A2,A3,A4,grid_dim), area_sizes, containment_dim, jumpsteps, rng)  # For now, initialize all keepout areas 10x10 having counter 1000.
    end
end

abs(x::CartesianIndex) = CartesianIndex(abs(x[1]), abs(x[2]))

# State: NamedTuple (fm=Fuel Map, bm=Burn Map, w=Wind, ac=Area Counters)
# Action: CartesianIndex
# Obs: AbstractArray [Float64, bmp]
function POMDPs.gen(m::WildfirePOMDP, s::NamedTuple, a::CartesianIndex{2}, rng::AbstractRNG)

    # Jump through multiple timesteps.
    s_dyn = wildfire_dynamics(s, m)
    fmp = s_dyn.fm #fuel map
    bmp = s_dyn.bm #burn map
    wp = s_dyn.w #wind
    acp = s_dyn.ac #counters

    r_spread = 0.0
    for (idx,fname) in enumerate(fieldnames(typeof(m.area_indices)))  # loop through i,ii,iii,iv
        if idx <= 4
            KO_bmp = bmp[getfield(m.area_indices, fname)]
            if sum(KO_bmp)>0  # if there is a single point greater than 1, then fire has breached that zone.
                r_spread -= 10.0*acp[idx]
                acp[idx] = 0
            end
        end
    end

    r = r_spread
    fmp = clear_fuel(fmp, a, m.area_indices, m.containment_dim)

    # Append the entire fire map and the corner area counters to the observation.
    df = dist_to_fire(m, s, a)
    ow = wp + randn(1, 2).*(0.1 + clamp((20.0-df)/20.0, 0, 2.0))
    ow = clamp.(ow, -1, 1)
    ow = round.(ow, digits=2)
    o = (ow=ow, df=df, fmp=fmp, bmp=bmp, acp=acp)

    sp = (fm=fmp, bm=bmp, w=wp, ac=acp)

    return (sp=sp,o=o,r=r)
end

function wildfire_dynamics(s::NamedTuple,m::WildfirePOMDP)
    for i in 1:m.jumpsteps
        fm = s.fm
        bm = s.bm
        w = s.w
        ac = s.ac

        # Wind Changes
        wx = w[1] + rand()*0.05 - 0.025
        wy = w[2] + rand()*0.05 - 0.025
        wp = [wx wy]
        # Fuel map changes.
        fmp = max.(fm .- bm.*m.burn_rate, 0.0)

        # Burn map dynamics.
        burn_weights = zeros(Float64, m.grid_dim, m.grid_dim)
        for ii=-2:2
            for jj=-2:2
                if ii != 0 | jj != 0
                    shifted_map = shift_map(bm, ii, jj)
                    num = max(0.0, 1.0 - sign(ii)*w[1] - sign(jj)*w[2])
                    den = (ii^2 + jj^2)
                    increment_map = shifted_map.*num/den
                    burn_weights += increment_map
                end
            end
        end

        # Decrement all counters by 1. Clamp so that counters do not go below zero.
        acp = clamp.(ac - [1,1,1,1], 0, 100) #changed Inf for type stability

        # Burn map changes.
        bmp = burn_weights .> rand(m.grid_dim, m.grid_dim).*m.burn_threshold
        bmp = bmp + bm .> 0
        bmp = bmp .* fmp .> 0
        bmp = convert(Array{Int8}, bmp)
        s = (fm=fmp, bm=bmp, w=wp, ac=acp)
    end
    return s
end

function shift_map(map::Array, ii::Int64, jj::Int64)
    T = eltype(map)
    r, c = size(map)
    shifted_map = zeros(T, r, c)
    row_src_idxs = ii >= 0 ? (1+abs(ii):r) : (1:r-abs(ii))
    row_trg_idxs = ii >= 0 ? (1:r-abs(ii)) : (1+abs(ii):r)
    col_src_idxs = jj >= 0 ? (1+abs(jj):c) : (1:c-abs(jj))
    col_trg_idxs = jj >= 0 ? (1:c-abs(jj)) : (1+abs(jj):c)
    shifted_map[row_trg_idxs, col_trg_idxs] = map[row_src_idxs, col_src_idxs]
    return shifted_map
end

function plot_state(s::NamedTuple)
    p1 = heatmap(s.bm,title="Burn Map", fill=true, clims=(0,1), legend=false, border=:none, c=:heat) #c=:grays_r)
    p2 = heatmap(s.fm,title="Fuel Map", fill=true, clims=(0,40), legend=true, border=:none, c=:algae_r) # c=:grays)
    p3 = plot(p1,p2,layout=(1,2),legend=false)
    return p1, p2, p3
end

function clear_fuel(fmp::Array{Float64}, a::CartesianIndex, area_indices::KeepOutAreas, containment_dim::Int64)
    fmp_cleared = fmp
    x,y = a.I
    areas_to_clear = vec(collect(CartesianIndices((x-containment_dim:x+containment_dim, y-containment_dim:y+containment_dim))))

    # Shall not clear a fuel location if it is in the keepout zone. During plotting, if a cleared area has a wierd shape, notice that it is in the vicinity of a keepout zone.
    for fname in fieldnames(typeof(area_indices))  # loop through i,ii,iii,iv
        filter!(e-> e∉getfield(area_indices, fname) ,areas_to_clear)
    end

    # Clearing fuel here.
    for item in areas_to_clear
        try
            fmp_cleared[item] = 0
        catch  # catches BoundsError, no action needed, just continue.
            continue
        end
    end
    return fmp_cleared
end

struct WildfireStateInitializer
    dim::Int64
    bm::Array{Int8}
    init_counters::Array{Int64}
    mean::Float64
    std::Float64
    function WildfireStateInitializer(m::WildfirePOMDP)
        bm0 = zeros(Int8, m.grid_dim, m.grid_dim)
        low_border = Int8(m.grid_dim/2-1)
        hi_border = Int8(m.grid_dim/2+2)
        bm0[low_border:hi_border, low_border:hi_border] .= 1
        init_counters = [100,100,100,100]
        mean = m.avg_fuel
        std = (m.max_fuel - m.min_fuel)/4.
        new(m.grid_dim, bm0, init_counters, mean, std)
    end
end


POMDPs.initialstate_distribution(m::WildfirePOMDP) = WildfireStateInitializer(m)

function Base.rand(d::WildfireStateInitializer)
    fm = Base.rand(Normal(d.mean, d.std), d.dim^2)
    fm = clamp.(fm, 0, 25)
    fm = round.(fm, digits=0)
    fm = reshape(fm, d.dim, d.dim)
    w = rand(1, 2).*2 .- 1
    (fm=fm, bm=d.bm, w=w, ac=d.init_counters)
end

function Base.rand(rng::AbstractRNG, d::WildfireStateInitializer)
    fm = Base.rand(rng, Normal(d.mean, d.std), d.dim^2)
    fm = clamp.(fm, 0, 25)
    fm = round.(fm, digits=0)
    fm = reshape(fm, d.dim, d.dim)
    w = rand(rng, 1, 2).*2 .- 1
    (fm=fm, bm=d.bm, w=w, ac=d.init_counters) #Maked named tuple parametric type def {(:a, :b),Tuple{Int64,String}}
end

function POMDPs.actions(p::WildfirePOMDP)
    burning_actions = vec(collect(CartesianIndices((1:p.grid_dim, 1:p.grid_dim))))
end

function POMDPs.actions(p::WildfirePOMDP, b::WildfireBelief)
    all_actions = POMDPs.actions(p)
    all_actions = [item for item in all_actions if b.burn_map[abs(item)]==0]  # removes locations already burnt by fire.
    all_keepout_areas = [item for fname in fieldnames(typeof(p.area_indices)) for item in getfield(p.area_indices,fname)]
    [item for item in all_actions if abs(item)∉all_keepout_areas]
end

POMDPs.discount(::WildfirePOMDP) = 0.99
POMDPs.isterminal(p::WildfirePOMDP, s::NamedTuple) = sum(s.ac) == 0  # terminate when all counters are zero.

function dist_to_keepouts(p::WildfirePOMDP, actions::Array{CartesianIndex{2}}, counters::Array{Int64})
    n_actions = size(actions)[1]
    d_k = zeros(Float64, n_actions)
    for (i, a) in enumerate(actions)
        d_k_a = Inf
        for (j, b) in enumerate(p.area_indices.mids)
            d = euclidean_dist(a,b)
            count = counters[j]
            if (d < d_k_a && count != 0)
                d_k_a = d
            end
        end
        d_k[i] = d_k_a
    end
    return d_k
end

function dist_to_fire(p::WildfirePOMDP, b::WildfireBelief, actions::Array{CartesianIndex{2}})  #d_f
    n_actions = size(actions)[1]
    d_f = zeros(Float64, n_actions)
    for (i, a) in enumerate(actions)
        d_f_a = Inf
        for f in findall(x->isequal(x,1), b.burn_map)
            d = euclidean_dist(a,f)
            if d < d_f_a
                d_f_a = d
            end
        end
        d_f[i] = d_f_a[1]
    end
    return d_f
end

function dist_to_fire(p::WildfirePOMDP, s::NamedTuple, action::CartesianIndex{2})
    d_f = Inf
    for f in findall(x->isequal(x,1), s.bm)
        d = euclidean_dist(action,f)
        if d < d_f
            d_f = d
        end
    end
    return d_f
end

function action_values(p::WildfirePOMDP, b::WildfireBelief)
    actions = POMDPs.actions(p, b)
    mu = -dist_to_keepouts(p, actions, b.area_counters)
    sig = -dist_to_fire(p,b,actions)
    fuel = b.fuel_map[actions]
    fuel = fuel .<= 0
    mu -= fuel.*p.grid_dim
    sig -= fuel.*p.grid_dim
    return (mu=mu, sig=sig, act=actions)
end

function POMDPModelTools.obs_weight(p::WildfirePOMDP, s::NamedTuple, a::CartesianIndex{2}, sp::NamedTuple, o)
    mean = s.w
    prob = pdf(Normal(mean[1], 0.1), o.ow[1])
    prob *= pdf(Normal(mean[2], 0.1), o.ow[2])
end
