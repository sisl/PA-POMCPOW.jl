

"""
    LEIso <: Isotropic{SqEuclidean}
Isotropic Linear Exponential kernel (covariance)
```math
k(x,x') = σ²\\exp(- abs(x - x')/ℓ)
```
with length scale ``ℓ`` and signal standard deviation ``σ``.
"""
mutable struct LEIso{T<:Real} <: GaussianProcesses.Isotropic{GaussianProcesses.SqEuclidean}
    "Length scale"
    ℓ::T
    "Signal variance"
    σ2::T
    "Priors for kernel parameters"
    priors::Array
end

"""
Linear Exponential kernel function
    LEIso(ll::T, lσ::T)
# Arguments:
- `ll::Real`: length scale (given on log scale)
- `lσ::Real`: signal standard deviation (given on log scale)
"""
LEIso(ll::T, lσ::T) where T = LEIso{T}(exp(ll), exp(2 * lσ), [])

function GaussianProcesses.set_params!(se::LEIso, hyp::AbstractVector)
    length(hyp) == 2 || throw(ArgumentError("Linear exponential has two parameters, received $(length(hyp))."))
    se.ℓ, se.σ2 = exp(hyp[1]), exp(2 * hyp[2])
end

GaussianProcesses.get_params(se::LEIso{T}) where T = T[log(se.ℓ), log(se.σ2) / 2]
GaussianProcesses.get_param_names(se::LEIso) = [:ll, :lσ]
GaussianProcesses.num_params(se::LEIso) = 2

GaussianProcesses.cov(se::LEIso, r::Number) = se.σ2*exp(-r/se.ℓ)

@inline dk_dll(se::LEIso, r::Real) = r/se.ℓ*GaussianProcesses.cov(se,r)
@inline function dk_dθp(se::LEIso, r::Real, p::Int)
    if p==1
        return dk_dll(se, r)
    elseif p==2
        return dk_dlσ(se, r)
    else
        return NaN
    end
end
