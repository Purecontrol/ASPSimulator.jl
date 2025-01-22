# ASPSimulator.jl

ASPSimulator.jl is a lightweight Julia package designed for simulating Ordinary Differential Equations (ODEs) in the context of Activated Sludge Processes (ASP).  

## Quick Start Example  

Below is a basic example demonstrating how to use the package:  

```julia
using ASPSimulator, Dates, OrdinaryDiffEq, Plots

# Select the system model
system = :asm1

# Initialize the ODE core
core = ASPSimulator.ODECore(system, variable_inlet_concentration=false)

# Run the simulation for 14 days with specified aeration and waiting times
ts_asm1 = multi_step!(core, clock_control(t_aerating = 30.0, t_waiting = 60.0), Day(14))

# Plot the results
plot(ts_asm1[:, :nh4])
```  

## Supported Systems  

Currently, the following ASP models are supported:  

- [ASM1 (Activated Sludge Model No. 1)](https://iwaponline.com/ebooks/book/96/Activated-Sludge-Models-ASM1-ASM2-ASM2d-and-ASM3?redirectedFrom=PDF)  
- Simplified ASM1  
- Linear ASM1  
