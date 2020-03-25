
@with_kw mutable struct PAPOMCPOWSolver{RNG<:AbstractRNG} <: POMCPOW.AbstractPOMCPSolver
    belief_updater::Updater     = nothing
    eps::Float64                = 0.01
    max_depth::Int              = typemax(Int)
    criterion                   = POMCPOW.MaxUCB(1.0)
    final_criterion             = POMCPOW.MaxQ()
    tree_queries::Int           = 1000
    max_time::Float64           = Inf
    rng::RNG                    = Random.GLOBAL_RNG
    node_sr_belief_updater      = POMCPOW.POWNodeFilter()

    estimate_value::Any         = POMCPOW.RolloutEstimator(POMCPOW.RandomSolver(rng))

    enable_action_pw::Bool      = true
    check_repeat_obs::Bool      = true
    check_repeat_act::Bool      = true
    tree_in_info::Bool          = false

    alpha_observation::Float64  = 0.5
    k_observation::Float64      = 10.0
    alpha_action::Float64       = 0.5
    k_action::Float64           = 10.0
    init_V::Any                 = 0.0
    init_N::Any                 = 0
    next_action::Any            = RandomActionGenerator(rng)
    default_action::Any         = POMCPOW.ExceptionRethrow()
end

mutable struct PAPOMCPOWPlanner{P,NBU,C,NA,SE,IN,IV} <: Policy
    solver::PAPOMCPOWSolver
    problem::P
    node_sr_belief_updater::NBU
    criterion::C
    next_action::NA
    solved_estimate::SE
    init_N::IN
    init_V::IV
    tree::Union{Nothing, POMCPOWTree} # this is just so you can look at the tree later
end

function PAPOMCPOWPlanner(solver, problem::POMDP)
    PAPOMCPOWPlanner(solver,
                  problem,
                  solver.node_sr_belief_updater,
                  solver.criterion,
                  solver.next_action,
                  POMCPOW.convert_estimator(solver.estimate_value, solver, problem),
                  solver.init_N,
                  solver.init_V,
                  nothing)
end

POMDPs.action(pomcp::PAPOMCPOWPlanner, b) = first(action_info(pomcp, b))

function POMCPOW.make_tree(p::PAPOMCPOWPlanner{P, NBU}, b) where {P, NBU}
    S = POMCPOW.statetype(P)
    A = POMCPOW.actiontype(P)
    O = POMCPOW.obstype(P)
    B = POMCPOW.belief_type(NBU,P)
    return POMCPOWTree{B, A, O, typeof(b)}(b, 2*min(100_000, p.solver.tree_queries))
end

function action_info(pomcp::PAPOMCPOWPlanner{P,NBU}, b; tree_in_info=false) where {P,NBU}
    A = actiontype(P)
    info = Dict{Symbol, Any}()
    tree = POMCPOW.make_tree(pomcp, b)
    pomcp.tree = tree
    local a::A
    try
        a = POMCPOW.search(pomcp, tree, b, info)
        if pomcp.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end
    catch ex
        println("EXCP")
        a = convert(A, default_action(pomcp.solver.default_action, pomcp.problem, b, ex))
    end
    return a, info
end

function POMCPOW.search(pomcp::PAPOMCPOWPlanner, tree::POMCPOWTree, b, info::Dict{Symbol,Any}=Dict{Symbol,Any}())
    all_terminal = true
    i = 0
    start_us = CPUtime_us()
    for i in 1:pomcp.solver.tree_queries
        s = rand(pomcp.solver.rng, tree.root_belief)
        if !POMDPs.isterminal(pomcp.problem, s)
            max_depth = min(pomcp.solver.max_depth, ceil(Int, log(pomcp.solver.eps)/log(discount(pomcp.problem))))
            simulate(pomcp, POWTreeObsNode(tree, 1), s, b, max_depth)
            all_terminal = false
        end
        if CPUtime_us() - start_us >= pomcp.solver.max_time*1e6
            break
        end
    end
    info[:search_time_us] = CPUtime_us() - start_us
    info[:tree_queries] = i

    if all_terminal
        throw(AllSamplesTerminal(tree.root_belief))
    end

    best_node = POMCPOW.select_best(pomcp.solver.final_criterion, POWTreeObsNode(tree,1), pomcp.solver.rng)

    return tree.a_labels[best_node]
end


function simulate(pomcp::PAPOMCPOWPlanner, h_node::POWTreeObsNode{B,A,O}, s::S, b, d) where {B,S,A,O}
    tree = h_node.tree
    h = h_node.node

    sol = pomcp.solver

    if POMDPs.isterminal(pomcp.problem, s) || d <= 0
        return 0.0
    end

    if sol.enable_action_pw
        total_n = tree.total_n[h]
        if length(tree.tried[h]) <= sol.k_action*total_n^sol.alpha_action
            a = POMCPOW.next_action(pomcp.next_action, pomcp.problem, b, POWTreeObsNode(tree, h))
            if !sol.check_repeat_act || !haskey(tree.o_child_lookup, (h,a))
                POMCPOW.push_anode!(tree, h, a,
                            init_N(pomcp.init_N, pomcp.problem, POWTreeObsNode(tree, h), a),
                            init_V(pomcp.init_V, pomcp.problem, POWTreeObsNode(tree, h), a),
                            sol.check_repeat_act)
            end
        end
    else # run through all the actions
        if isempty(tree.tried[h])
            action_space_iter = POMDPs.actions(pomcp.problem, s)
            anode = length(tree.n)
            for a in action_space_iter
                push_anode!(tree, h, a,
                            init_N(pomcp.init_N, pomcp.problem, POWTreeObsNode(tree, h), a),
                            init_V(pomcp.init_V, pomcp.problem, POWTreeObsNode(tree, h), a),
                            false)
            end
        end
    end
    total_n = tree.total_n[h]

    best_node = POMCPOW.select_best(pomcp.criterion, h_node, pomcp.solver.rng)
    a = tree.a_labels[best_node]

    new_node = false
    if tree.n_a_children[best_node] <= sol.k_observation*(tree.n[best_node]^sol.alpha_observation)

        sp, o, r = gen(DDNOut(:sp, :o, :r), pomcp.problem, s, a, sol.rng)

        if sol.check_repeat_obs && haskey(tree.a_child_lookup, (best_node,o))
            hao = tree.a_child_lookup[(best_node, o)]
        else
            new_node = true
            hao = length(tree.sr_beliefs) + 1
            push!(tree.sr_beliefs,
                  POMCPOW.init_node_sr_belief(pomcp.node_sr_belief_updater,
                                      pomcp.problem, s, a, sp, o, r))
            push!(tree.total_n, 0)
            push!(tree.tried, Int[])
            push!(tree.o_labels, o)

            if sol.check_repeat_obs
                tree.a_child_lookup[(best_node, o)] = hao
            end
            tree.n_a_children[best_node] += 1
        end
        push!(tree.generated[best_node], o=>hao)
    else

        sp, r = gen(DDNOut(:sp, :r), pomcp.problem, s, a, sol.rng)

    end

    if r == Inf
        @warn("POMCPOW: +Inf reward. This is not recommended and may cause future errors.")
    end

    if new_node
        R = r + POMDPs.discount(pomcp.problem)*POMCPOW.estimate_value(pomcp.solved_estimate, pomcp.problem, sp, POWTreeObsNode(tree, hao), d-1)
    else
        pair = rand(sol.rng, tree.generated[best_node])
        o = pair.first
        hao = pair.second
        POMCPOW.push_weighted!(tree.sr_beliefs[hao], pomcp.node_sr_belief_updater, s, sp, r)
        sp, r = rand(sol.rng, tree.sr_beliefs[hao])

        bp = POMDPs.update(pomcp.solver.belief_updater, b, a, o)
        R = r + POMDPs.discount(pomcp.problem)*simulate(pomcp, POWTreeObsNode(tree, hao), sp, bp, d-1)
    end

    tree.n[best_node] += 1
    tree.total_n[h] += 1
    if tree.v[best_node] != -Inf
        tree.v[best_node] += (R-tree.v[best_node])/tree.n[best_node]
    end

    return R
end

function POMDPs.solve(solver::PAPOMCPOWSolver, problem::POMDP)
    return PAPOMCPOWPlanner(solver, problem)
end
