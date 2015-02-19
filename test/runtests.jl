using HuberLR
using Base.Test

let
    # true parameters
    w = [1., 2, 3., 4.]
    b = 5.

    # generate sample points
    srand(20150218)
    x = randn(length(w), 150)
    y = mapslices(x, 1) do x
        w ⋅ x + b + randn() + (rand() > .9 ? abs(randn()) * 5 : 0.)
    end |> vec
    #writedlm("data.tsv", vcat(x, y')')

    # JIT compile
    lr = LinearRegression(.5)
    fit!(lr, x[:,1:5], y[1:5])

    # robust regression
    lr = LinearRegression(1.)
    @time fit!(lr, x, y)
    println(lr)
    @show norm(predict(lr, x) .- y)
    Δw = norm(lr.w .- [b, w])

    # quadratic loss (not robust)
    lr = LinearRegression(Inf)
    @time fit!(lr, x, y)
    println(lr)
    @show norm(predict(lr, x) .- y)
    Δw′ = norm(lr.w .- [b, w])

    #@show Δw Δw′
    @test Δw < Δw′ < 1.
end
