
mutable struct ActionSelector{A}
    k::Function
    C::Array{Float64}
    b0::Any
    D::Dict
    d::Union{Nothing, Function}
    function ActionSelector{A}(k::Function, C::Array{Float64}, d::Union{Nothing, Function}=nothing) where {A} #, sz::Int)
        b0 = nothing
        D = Dict{POWTreeObsNode, Array{A}}()
        new(k, C, b0, D, d)
    end
end

"""
    embed_actions(action_vals::NamedTuple, p::DAEPOMCPPlanner)
A helper function that embeds the action space into the DAE action space.

Returns: Array of actions defined by the POMDP action space
"""
function embed_actions(action_vals::NamedTuple, C::Union{Array{Float64}, Float64}, n_embeddings::Integer, sample_acts::Bool)
    if sample_acts
        acts = sample_actions(C, action_vals.mu, action_vals.sig, action_vals.act, n_embeddings)
    else
        mu = deepcopy(action_vals.mu)
        sig = deepcopy(action_vals.sig)
        act_array = deepcopy(action_vals.act)
        acts = []
        for c=C
            act_idx,  act = max_action(c, mu, sig, act_array)
            push!(acts, act)
            deleteat!(mu, act_idx)
            deleteat!(sig, act_idx)
            deleteat!(act_array, act_idx)
        end
    end
    return acts
end

"""
    max_action(c::Float64, mu::Array{Float64}, sig::Array{Float64}, act_array::Array)
A helper function that selects the action that maximizes the linear combination of the exploitation
score "mu" and exploration score "sig" weighted by "c" as below

score = (1. - c)*mu + c*sigma

Returns: An action of the type defined by the POMDP
"""
function max_action(c::Float64, mu::Array{Float64}, sig::Array{Float64}, act_array::Array)
    val = mu + c*sig
    act_idx = argmax(val)
    act = act_array[act_idx]
    return act_idx, act
end

function sample_actions(c::Float64, mu::Array{Float64}, sig::Array{Float64}, act_array::Array, n_embeddings::Integer)
    val = mu + c*sig
    scale = float(n_embeddings)
    val = exp.(val)
    val = val ./ sum(val)
    n_acts = size(act_array)
    n_vals = size(val)
    acts = sample(act_array, ProbabilityWeights(val), n_embeddings, replace=false)
    return acts
end

function POMCPOW.next_action(o::ActionSelector, pomdp::POMDP, b::Any, h::Any)
    b0 = h.tree.root_belief
    if o.b0 == nothing || o.b0 != b0
        o.b0 = b0
        o.D = Dict{POWTreeObsNode, Any}()
    end
    if !(h in keys(o.D))
        action_vals = o.k(pomdp, b)
        acts = embed_actions(action_vals, o.C, size(o.C)[1], false)
        if !(isnothing(o.d))
            acts = o.d(acts)
        end
        push!(o.D, h => acts)
    else
        acts = o.D[h]
    end
    act = rand(acts)
    if length(acts) > 1 #necessary for check_repeat_act to work correctly
        filter!(e->e!=act, acts)
    end
    return act
end
