
struct GreedyPolicy <: Policy
    m::POMDP
end

function POMDPs.action(p::GreedyPolicy, b::GaussianProcessBelief)
    available_actions = actions(p.m, b)
    locations = collect(item[2] for item in available_actions)
    action_array = cartind_to_array(locations)
    mu_sigma = GaussianProcessSolve(b, action_array)
    action_idx = findmax(mu_sigma[1])[2]
    action_location = locations[action_idx]
    action = (true, action_location)
end

function POMDPs.action(pol::GreedyPolicy, b::WildfireBelief)
    mu_sig = action_values(pol.m, b)
    max_idx = argmax(mu_sig.mu)
    return mu_sig.act[max_idx]
end

function greediest_actions(p::WildfirePOMDP)
    # The greediest strategy is to place the actions to a neighbour of a keepout zone in each timestep. To choose the most greediest strategy, it should be the one closest to the fire.
    all_actions = []

    for fname in fieldnames(typeof(p.area_indices))
        acts = getfield(p.area_indices,fname)
        acts_corners = []
        append!(acts_corners, acts[:,1])
        append!(acts_corners, acts[:,end])
        append!(acts_corners, acts[1,:])
        append!(acts_corners, acts[end,:])

        for a in acts_corners
            x,y = a.I
            neighbours = vec(collect(CartesianIndices((x-1:x+1, y-1:y+1))))
            append!(all_actions,neighbours)
        end

    end

    unique!(all_actions)
    available_actions = [item for item in all_actions if item∈actions(p)]
end


function POMDPs.action(pol::GreedyPolicy, b::WildfireBelief)
    x_obs_locs = Array_to_CartInd(b.x_obs)
    locations = [item for item in pol.loc if item∉x_obs_locs]

    if isempty(locations)
        locations = actions(pol.m,b)
        @warn "Greedy was able to wall all of the corners"
    end

    inv_weights = dist_to_fire(pol.m, b, locations)
    most_greedy_action = argmin(inv_weights)

    return locations[most_greedy_action]
end
