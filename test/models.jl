
@testset "Models" begin
    @testset "Base" begin
        @test_throws ArgumentError ASPSimulator.set_default_system(:asm2)
        @test ASPSimulator.get_nh4_index() == ASPSimulator.get_indexes_from_symbols(:nh4)[1]
        @test ASPSimulator.get_no3_index() == ASPSimulator.get_indexes_from_symbols(:no3)[1]
        @test ASPSimulator.get_o2_index() == ASPSimulator.get_indexes_from_symbols(:o2)[1]

    end

    @testset "ASM1" begin
        ASPSimulator.set_default_system(:asm1)
        @test ASPSimulator.get_default_system() == :asm1
        @test ASPSimulator.get_nh4_index() == 10
        @test ASPSimulator.get_no3_index() == 9
        @test ASPSimulator.get_o2_index() == 8
        @test ASPSimulator.get_control_index() == 14
        @test ASPSimulator.get_number_variables() == 14

        # Test ASM1
        core = ASPSimulator.ODECore(ASPSimulator.get_default_system());
        ts_asm1_1 = step!(core, redox_control())
        @test timestamp(ts_asm1_1)[1] ≈ core.current_t
        @test values(ts_asm1_1)[1, :] ≈ core.current_state
        
        ts_asm1_2 = multi_step!(core, redox_control(), 100)
        @test size(ts_asm1_2, 1) == 100

        ts_asm1_3 = multi_step!(core, redox_control(), Day(1))
        @test size(ts_asm1_3, 1) == Int(1/core.fixed_dt)

        # Test RedoxControl, FixedControl, ClockControl
    end

    @testset "ASM1 Simplified" begin
        ASPSimulator.set_default_system(:asm1_simplified)
        @test ASPSimulator.get_default_system() == :asm1_simplified
        @test ASPSimulator.get_nh4_index() == 4
        @test ASPSimulator.get_no3_index() == 3
        @test ASPSimulator.get_o2_index() == 2
        @test ASPSimulator.get_control_index() == 6
    end
end
