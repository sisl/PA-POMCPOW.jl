# PA-POMCPOW.jl
PA-POMCPOW is an online POMDP tree search solver that introduces action prioritization to the progressive widening of the POMCPOW algorithm. For more information, see the paper at https://arxiv.org/pdf/2010.03599.pdf. For information on the original POMCPOW, see https://arxiv.org/abs/1709.06196.

PA-POMCPOW solves problems defined using the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface. In addtion to the elements required by POMCPOW, PA-POMCPOW also requires definition of the expected information gain term as specified in the paper. 

## Installation

```julia
import Pkg
Pkg.add(https://github.com/sisl/PA-POMCPOW.jl)
```
## Usage
```julia 
using POMDPs
using POMDPModelTools
using POMDPSimulators
using D3Trees

m = ExamplePOMDP()
ds0 = POMDPs.initialstate_distribution(m)
s0 = rand(ds0)
up = ExampleBeliefUpdater()
b0 = initialize_belief(up, s0)

C = [0:0.1:2.0;]

action_selector = ActionSelector(example_pomdp_action_value_function, C)

solver = PAPOMCPOWSolver(tree_queries=1000,next_action=action_selector, belief_updater=up, max_depth=25)
planner = solve(solver, m)

hr = HistoryRecorder(max_steps=100)
hist = simulate(hr, pomdp, planner)
for (s, b, a, r, sp, o) in hist
    @show s, a, r, sp
end
```
The algorithm behavior is determined by the keyword argument values passed to the solver constructor. To enact the prioritized action selection, an `ActionSelector` object must be passed in the `next_action` argument. Without this, the algorithm defaults to the behavior of POMCPOW. All other keyword arguments control behavior as described in the original POMCPOW paper and in [POMCPOW.jl](https://github.com/JuliaPOMDP/POMCPOW.jl).

## Action Value Function
To solve a POMDP using PA-POMCPOW requires an `action_values` function to be defined for the problem. The function should have the arguments and returns shown below.
```julia
function action_values(p::POMDP, b::Belief)
    # YOUR CODE HERE
    return (mu=mu, sig=sig, act=act)
end
```
The `mu` and `sig` fields of the returned `NamedTuple` are the mean and standard deviation of the expected immediate reward of taking each action in the vector returned in the `act` field.
    
