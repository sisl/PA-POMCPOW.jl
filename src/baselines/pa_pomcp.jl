
"""
    DAEPOMCPSolver
A solver type for the Dynamic Action Embedding POMCP Algorithm. DAE parmater "n_embeddings"
defines number of action-value embeddings per-branch step. For standard POMCP
parameter definition, see the BasicPOMCP.jl documentation.
"""
@with_kw mutable struct DAEPOMCPSolver <: AbstractPOMCPSolver
    belief_updater::POMDPs.Updater
    max_depth::Int          = 20
    c::Float64              = 1.0
    sample_acts::Bool       = false
    tree_queries::Int       = 1000
    max_time::Float64       = Inf
    tree_in_info::Bool      = false
    default_action::Any     = ExceptionRethrow()
    rng::AbstractRNG        = Random.GLOBAL_RNG
    n_embeddings::Int       = 100
end

"""
    DAEPolicy
A policy which randomly samples from the embedded action space. The actions are re-embedded
each step according to the C-values provided.
"""
mutable struct DAEPolicy <: Policy
    C::Array{Float64}
    pomdp::POMDP
end

"""
    AEPolicy
A policy which randomly samples from the embedded action space. The a static action embedding
is used for every policy call
"""
struct AEPolicy <: Policy
    acts::Array
end

"""
    DAEPOMCPPlanner{P::POMDP, RNG::AbstractRNG}
    DAEPOMCPPlanner(solver::DAEPOMCPSolver, pomdp::POMDP)
    solve(solver::DAEPOMCPSolver, pomdp::POMDP)
A planner type for the Dynamic Action Embedding POMCP Algorithm. For standard
POMCP parameter definition, see the BasicPOMCP.jl documentation.
NOTE: Currently DAEPOMCPPlanner is defaulted to use the AE policy for computational efficiency reasons
"""
mutable struct DAEPOMCPPlanner{P, RNG} <: Policy
    solver::DAEPOMCPSolver
    C::Union{Array{Float64}, Float64}
    problem::P
    rng::RNG
    _best_node_mem::Vector{Int}
    _tree::Union{Nothing, Any}
end

function DAEPOMCPPlanner(solver::DAEPOMCPSolver, pomdp::POMDP)
    if solver.sample_acts
        C = solver.c
    else
        C = range(0., stop=solver.c, length=solver.n_embeddings) |> collect
        C = exp.(C) .- 1.0
    end
    return DAEPOMCPPlanner(solver, C, pomdp, solver.rng, Int[], nothing)
end

POMDPs.solve(solver::DAEPOMCPSolver, pomdp::POMDP) = DAEPOMCPPlanner(solver, pomdp)

"""
    action_values(p::POMDP, b)
Given current belief "b", returns the exploitation value and the exploration value of each action
Must be implemeted for each POMDP and belief combination to be used

Returns: Named Tuple (mu=Array{Float64}, sig=Array{Float64}, act=Array)
    where mu is the exploitation score, sig is the exploration score, and act is the action
    corresponding to each score.
"""

function action_values end

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
    val = val.*scale
    val = exp.(val)
    val = val ./ sum(val)
    acts = sample(act_array, ProbabilityWeights(val), n_embeddings, replace=false)
    return acts
end

"""
    embed_actions(action_vals::NamedTuple, p::DAEPOMCPPlanner)
A helper function that embeds the action space into the DAE action space.

Returns: Array of actions defined by the POMDP action space
"""
function embed_actions(action_vals::NamedTuple, C::Union{Array{Float64}, Float64}, n_embeddings::Integer, sample_acts::Bool)
    mu = deepcopy(action_vals.mu)
    sig = deepcopy(action_vals.sig)
    act_array = deepcopy(action_vals.act)
    if sample_acts
        acts = sample_actions(C, mu, sig, act_array, n_embeddings)
    else
        acts = CartesianIndex{2}[]
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
    POMCPTree(pomdp::POMDP, b, p::DAEPOMCPPlanner, sz::Int=1000)
A constuctor function for the POMCPTree type that creates branches based on the
dynamic belief embedding.
"""
function POMCPTree(pomdp::POMDP, b, p::DAEPOMCPPlanner, sz::Int=1000)
    action_vals = action_values(pomdp, b)
    acts = embed_actions(action_vals, p.C, p.solver.n_embeddings, p.solver.sample_acts)
    A = actiontype(pomdp)
    O = obstype(pomdp)
    sz = min(100_000, sz)
    tree = BasicPOMCP.POMCPTree{A,O}(sizehint!(Int[0], sz),
                          sizehint!(Vector{Int}[collect(1:length(acts))], sz),
                          sizehint!(Array{O}(undef, 1), sz),

                          sizehint!(Dict{Tuple{Int,O},Int}(), sz),

                          sizehint!(zeros(Int, length(acts)), sz),
                          sizehint!(zeros(Float64, length(acts)), sz),
                          sizehint!(acts, sz)
                         )
    return tree
end

"""
    dae_insert_obs_node!(t::BasicPOMCP.POMCPTree, pomdp::POMDP, p::DAEPOMCPPlanner, ha::Int, o, b)
A helper function that adds an observation node and children action nodes to the POMCPTree

Returns: The updated "HAO", the used action space, and action-value map
"""
function dae_insert_obs_node!(t::BasicPOMCP.POMCPTree, pomdp::POMDP, p::DAEPOMCPPlanner, ha::Int, o, b)
    action_vals = action_values(pomdp, b)
    acts = embed_actions(action_vals, p.C, p.solver.n_embeddings, p.solver.sample_acts)
    push!(t.total_n, 0)
    push!(t.children, sizehint!(Int[], length(acts)))
    push!(t.o_labels, o)
    hao = length(t.total_n)
    t.o_lookup[(ha, o)] = hao
    for a in acts
        n = BasicPOMCP.insert_action_node!(t, hao, a)
        push!(t.children[hao], n)
    end
    return hao, acts
end

"""
    action_info(p::DAEPOMCPPlanner, b; tree_in_info=false)
The action_info method required by POMDPs.jl simulation standard.

Returns: the selected action and the info vector which may contain the search tree
"""
function action_info(p::DAEPOMCPPlanner, b; tree_in_info=false)
    local a::actiontype(p.problem)
    info = Dict{Symbol, Any}()
    try
        tree = POMCPTree(p.problem, b, p, p.solver.tree_queries)
        a = search(p, b, tree, info)
        p._tree = tree
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end
    catch ex
        # Note: this might not be type stable, but it shouldn't matter too much here
        a = convert(actiontype(p.problem), default_action(p.solver.default_action, p.problem, b, ex))
        info[:exception] = ex
    end
    return a, info
end

"""
    search(p::DAEPOMCPPlanner, b, t::BasicPOMCP.POMCPTree, info::Dict)
Executes the SEARCH subroutine as defined by the POMCP algorithm

Returns: The highest value action node from the current belief node
"""
function search(p::DAEPOMCPPlanner, b, t::BasicPOMCP.POMCPTree, info::Dict)
    all_terminal = true
    nquery = 0
    start_us = CPUtime_us()
    for i in 1:p.solver.tree_queries
        nquery += 1
        if CPUtime_us() - start_us >= 1e6*p.solver.max_time
            break
        end
        s = Base.rand(p.rng, b)
        if !POMDPs.isterminal(p.problem, s)
            simulate(p, s, b, BasicPOMCP.POMCPObsNode(t, 1), p.solver.max_depth)
            all_terminal = false
        end
    end
    info[:search_time_us] = CPUtime_us() - start_us
    info[:tree_queries] = nquery

    if all_terminal
        throw(AllSamplesTerminal(b))
    end

    h = 1
    best_node = first(t.children[h])
    best_v = t.v[best_node]
    @assert !isnan(best_v)
    for node in t.children[h][2:end]
        if t.v[node] >= best_v
            best_v = t.v[node]
            best_node = node
        end
    end

    return t.a_labels[best_node]
end

action(p::DAEPOMCPPlanner, b) = first(action_info(p, b))

"""
    simulate(p::DAEPOMCPPlanner, s, b, hnode::BasicPOMCP.POMCPObsNode, steps::Int)
Executes the SIMULATE subroutine as defined by the POMCP algorithm

Returns: The estimated accumulated discounted reward value of the current belief node
"""
function simulate(p::DAEPOMCPPlanner, s, b, hnode::BasicPOMCP.POMCPObsNode, steps::Int)
    if steps == 0 || isterminal(p.problem, s)
        return 0.0
    end

    t = hnode.tree
    h = hnode.node

    ltn = log(t.total_n[h])
    best_nodes = empty!(p._best_node_mem)
    best_criterion_val = -Inf
    for node in t.children[h]
        n = t.n[node]
        if n == 0 && ltn <= 0.0
            criterion_value = t.v[node]
        elseif n == 0 && t.v[node] == -Inf
            criterion_value = Inf
        else
            criterion_value = t.v[node] + p.solver.c*sqrt(ltn/n)
        end
        if criterion_value > best_criterion_val
            best_criterion_val = criterion_value
            empty!(best_nodes)
            push!(best_nodes, node)
        elseif criterion_value == best_criterion_val
            push!(best_nodes, node)
        end
    end
    ha = Base.rand(p.rng, best_nodes)
    a = t.a_labels[ha]
    sp, o, r = gen(DDNOut(:sp, :o, :r), p.problem, s, a, p.rng)
    bp = POMDPs.update(p.solver.belief_updater, b, a, o) #
    hao = get(t.o_lookup, (ha, o), 0)
    if hao == 0
        hao, acts = dae_insert_obs_node!(t, p.problem, p, ha, o, bp)
        v = rollout(p, sp, bp, acts, steps-1)
        R = r + discount(p.problem)*v
    else
        R = r + discount(p.problem)*simulate(p, sp, bp, BasicPOMCP.POMCPObsNode(t, hao), steps-1)
    end

    t.total_n[h] += 1
    t.n[ha] += 1
    t.v[ha] += (R-t.v[ha])/t.n[ha]
    return R
end

"""
    action(p::DAEPolicy, b)
Required method for POMDPs action function defining how DAEPolicy samples actions given belief b.

Returns: Action
"""
function action(p::DAEPolicy, b)
    action_vals = action_values(p.pomdp, b)
    acts = embed_actions(action_vals, p.C)
    rand(acts)
end

"""
action(p::AEPolicy, b)
Required method for POMDPs action function defining how AEPolicy samples actions given belief b.

Returns: Action
"""
function action(p::AEPolicy, b)
    Base.rand(p.acts)
end

"""
    rollout(p::DAEPOMCPPlanner, start_state, b, steps::Int)
    rollout(p::DAEPOMCPPlanner, start_state, b, acts, steps::Int)
A helper function to ROLLOUT a simulation using the POMDP gen function

Returns: The episode accumulated discounted rewards
"""
function rollout(p::DAEPOMCPPlanner, start_state, b, acts, steps::Int)
    sim = RolloutSimulator(p.rng,
                           steps)
    policy = AEPolicy(acts)
    # policy = GreedyPolicy(p.problem)
    return POMDPs.simulate(sim, p.problem, policy, p.solver.belief_updater, b, start_state)
end

function rollout(p::DAEPOMCPPlanner, start_state, b, steps::Int)
    sim = RolloutSimulator(p.rng,
                           steps)
    return POMDPs.simulate(sim, p.problem, p.policy, p.solver.belief_updater, b, start_state)
end
