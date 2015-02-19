module HuberLR

export LinearRegression, fit!, predict

using Optim

type LinearRegression
    δ::Float64
    w::Vector{Float64}
    function LinearRegression(δ)
        if δ < 0.
            error("δ should be non-negative")
        end
        new(δ, Float64[])
    end
end

function Base.show(io::IO, lr::LinearRegression)
    print(io, "LinearRegression:\n")
    print(io, "    δ: $(lr.δ)\n")
    ws = ASCIIString[]
    for w in lr.w
        s = @sprintf "%.4f" w
        push!(ws, s)
    end
    print(io, "    w: $(join(ws, ", "))")
end

immutable Data
    x::Matrix{Float64}
    y::Vector{Float64}
    function Data(x, y)
        if size(x, 2) != length(y)
            error("the number of data points must match")
        end
        new(x, y)
    end
end

Base.size(data::Data) = length(data.y)
dim(data::Data) = size(data.x, 1)

function fit!(lr::LinearRegression, x::Matrix{Float64}, y::Vector{Float64}; method=:l_bfgs, show_trace=false)
    D, N = size(x)
    x = vcat(ones(N)', x)
    w₀ = randn(D + 1)
    data = Data(x, y)
    if method ∈ [:l_bfgs, :bfgs]
        opt = optimize(w -> loss(w, lr, data), (w, ∇) -> ∇loss!(w, ∇, lr, data), w₀, method=method, show_trace=show_trace)
    elseif method ∈ [:nelder_mead]
        opt = optimize(w -> loss(w, lr, data), w₀, method=method, show_trace=show_trace)
    else
        error("method $method is not supported")
    end
    lr.w = opt.minimum
    lr
end

function predict(lr::LinearRegression, x::Matrix{Float64})
    D, N = size(x)
    if isempty(lr.w)
        error("the model is not yet trained")
    elseif length(lr.w) != D + 1
        error("dimention mismatch")
    end
    ŷ = Array(Float64, N)
    @inbounds for n in 1:N
        s = lr.w[1]
        for j in 1:D
            s += lr.w[j+1] * x[j,n]
        end
        ŷ[n] = s
    end
    ŷ
end

# dot product along the n-th column
function dotat(w::Vector{Float64}, x::Matrix{Float64}, n::Int)
    s = 0.
    @inbounds for i in 1:length(w)
        s += w[i] * x[i,n]
    end
    s
end

# Huber loss function
function loss(w::Vector{Float64}, lr::LinearRegression, data::Data)
    N = size(data)
    δ = lr.δ
    ∑loss = 0.
    @inbounds for n in 1:N
        r = data.y[n] - dotat(w, data.x, n)
        ∑loss += ifelse(abs(r) < δ, r^2 / 2., abs(r) * δ - δ^2 / 2.)
    end
    ∑loss
end

# first derivative of Huber loss function
function ∇loss!(w::Vector{Float64}, ∇::Vector{Float64}, lr::LinearRegression, data::Data)
    N = size(data)
    D = dim(data)
    δ = lr.δ
    fill!(∇, 0.)
    @inbounds for n in 1:N
        r = data.y[n] - dotat(w, data.x, n)
        ∂loss = ifelse(abs(r) < δ, r, sign(r) * δ)
        for j in 1:D
            ∂r = -data.x[j,n]
            ∇[j] += ∂loss * ∂r
        end
    end
    ∇
end

end # module
