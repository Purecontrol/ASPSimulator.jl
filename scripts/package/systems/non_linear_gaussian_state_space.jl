@doc raw"""

`GaussianNonLinearStateSpaceSystem(M_t::Function, H_t::Function, R_t::Function, Q_t::Function, dt::Float64)`

Definition of the system fonctions ``M_t, H_t, R_t, Q_t`` for nonlinear gaussian state space models with a fixed timestep of dt.

```math
\begin{gather*}
    \begin{aligned}
        \x{t+1} &= M_t (\x{t} , u(t)) + \eta_{t} \quad &\eta_{t} \sim \mathcal{N}(0, R_t)\\
        y_{t}   &=  H_t (\x{t}) + \epsilon_{t} \quad &\epsilon_{t} \sim \mathcal{N}(0, Q_t)\\
    \end{aligned}
\end{gather*}
```

where:

* ``x_t`` is a ``n_X \times 1`` vector
* ``y_t`` is a ``n_Y \times 1`` vector
* ``u_t`` is a ``n_U \times 1`` vector
* ``M_t`` is a ``n_X -> n_X`` function
* ``H_t`` is a ``n_X -> n_Y`` function
* ``R_t`` is a ``n_X \times n_X`` matrix
* ``Q_t`` is a ``n_Y \times n_Y`` matrix
"""
mutable struct GaussianNonLinearStateSpaceSystem <: StateSpaceSystem

    # General components of gaussian non linear state space systems 
    M_t::Function
    H_t::Function
    R_t::Function
    Q_t::Function

    # Time between two states
    dt::Float64

    function GaussianNonLinearStateSpaceSystem(M_t, H_t, R_t, Q_t, dt)

        return new(M_t, H_t, R_t, Q_t, dt)
    end

end

function transition(ssm::GaussianNonLinearStateSpaceSystem, current_x, exogenous_variables, control_variables, parameters) 

    return ssm.M_t(current_x, exogenous_variables, control_variables, parameters, t)

end

function observation(ssm::GaussianNonLinearStateSpaceSystem, current_x, exogenous_variables, parameters) 

    return ssm.H_t(current_x, exogenous_variables, parameters, t)

end
