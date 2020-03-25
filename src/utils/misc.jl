function cartind_to_array(X)
    flattened = X[:]
    out = zeros(Float64, 2, 0)
    for i=1:length(flattened)
        x = reshape(Float64[flattened[i][1], flattened[i][2]], 2, 1)
        out = [out x]
    end
    return out
end
average(a::AbstractArray) = sum(a::AbstractArray)/length(a::AbstractArray)

sigmoid(z::Real) = 1.0 / (1.0 + exp(-z))

expo_activator(z::Real, base::Real) = (base^z - 1.0)/(base - 1.0)

# An exponential decay function guarenteed to be zero at depth of full_greedy, and 1.0 at depth=1. Then, clamps the value to be between [0,1].
expo_decay(d::Real, zeta::Real, full_greedy::Int) = clamp((zeta^(d-1) - zeta^(full_greedy-1)) / (1 - zeta^(full_greedy-1)), 0, 1)

Base.abs(a::CartesianIndex) = a > CartesianIndex(0,0) ? a : -a

function euclidean_dist(a::CartesianIndex,b::CartesianIndex)
    diff = abs(a-b).I
    return sqrt(diff[1]^2 + diff[2]^2)
end

function normalize!(a::AbstractArray)  # normalizes any-size Array in-place.
    norma = maximum(a)
    for i = firstindex(a):lastindex(a)
        a[i] /= norma
    end
end

function stretch!(a::AbstractArray, lower_lim::Real, upper_lim::Real)  # stretches (forces given extremas, interpolates others in-between) any-size Array in-place.
    mini, maxi = minimum(a), maximum(a)
    norma = (maxi-mini)/(upper_lim-lower_lim)

    b = (a .- mini) ./ norma .+ lower_lim

    for i = firstindex(a):lastindex(a)
        a[i] = b[i]
    end
end

function stretch(a::AbstractArray, lower_lim::Real, upper_lim::Real)  # same as above, but not in-place.
    mini, maxi = minimum(a), maximum(a)
    norma = (maxi-mini)/(upper_lim-lower_lim)

    b = (a .- mini) ./ norma .+ lower_lim
    return b
end

function euclidean_dist(a::CartesianIndex,b::CartesianIndex)
    diff = abs(a-b).I
    return sqrt(diff[1]^2 + diff[2]^2)
end

function reshape_gp_samples(X, idxs, dim)
    flattened_idxs = idxs[:]
    Y = zeros(Float64, dim, dim)
    for i=1:length(flattened_idxs)
        idx = flattened_idxs[i]
        y = X[i]
        Y[idx] = y
    end
    return Y
end
average(x) = sum(x)/length(x)

distance(a::CartesianIndex, b::CartesianIndex) = abs2.(collect(a.I) - collect(b.I)) |> sum |> sqrt

function check_clossness_constraint(X, best_point_loc, closeness_constraint)
    for idx in 1:size(X,2)
        sensor_in_X = CartesianIndex(Int.(X[:,idx])...)

        # @show sensor_in_X
        # @show distance(best_point_loc, sensor_in_X)

        if distance(best_point_loc, sensor_in_X) < closeness_constraint
            return false  # new location does not satisfy constraint.
        end
    end
    return true  # new location satisfies constraint.
end

"""
If the filename exists, add a number to it and then save. Prevents overwrite.
filename: Name of file. Don't put filetype (e.g. "png") to the end of filename! Defaults to ".png".
reset: Delete everything in folder before saving? Set to true or false.
"""
function savefig_recursive(plt_obj, filename, reset; dir="Figures")

    if reset
        run(`rm -rf ./$dir`)
        mkdir("$dir")  # recreate the folder.
        filedir = dir*"/"*filename
        savefig(plt_obj, filedir)
    else
        dir in readdir() ? nothing : mkdir("$dir")  # if the folder exists, do nothing, otherwise, create it.
        i = 1
        itemname = filename
        while true in [occursin(itemname, item) for item in readdir(dir)]
            itemname = filename*string(i)
            i += 1
        end
        filedir = dir*"/"*itemname
        savefig(plt_obj, filedir)
    end

    return nothing
end
