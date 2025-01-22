using Dates

@testset "TSConcentrations" begin
    
    # Constructor whithout timestamp
    ts1 = TSConcentrations(zeros(Float64, 10, 2))
    @test isa(values(ts1), Matrix{Float64})
    @test isa(timestamp(ts1), Vector{Float64})
    @test size(ts1) == (10, 2)
    @test size(timestamp(ts1), 1) == 10

    # Constructor with timestamp
    ts2 = TSConcentrations(collect(1:9), zeros(Float32, 9, 4))
    @test isa(values(ts2), Matrix{Float32})
    @test isa(timestamp(ts2), Vector{Int64})
    @test size(ts2) == (9, 4)
    @test size(timestamp(ts2), 1) == 9

    # Constructor with Dates
    ts3 = TSConcentrations([DateTime(2013,7,1,12,30+i) for i in 1:7], zeros(Bool, 7, 2))
    @test isa(timestamp(ts3), Vector{DateTime})

    # DimensionMismatch
    @test_throws DimensionMismatch TSConcentrations(collect(1:9), zeros(10, 2))

end