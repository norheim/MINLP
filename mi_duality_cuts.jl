using JuMP, PiecewiseLinearOpt, Gurobi, MosekTools, LinearAlgebra
GRB_ENV = Gurobi.Env();

# NOTE: This is likely to run slow the first time because of Julia compilations

# Simple example model (with unique solution)
m = Model(optimizer_with_attributes(Mosek.Optimizer, 
    "QUIET" => true, 
    "INTPNT_CO_TOL_DFEAS" => 1e-7
    ));
@variable(m, x)
@variable(m, y)
x1 = [x-3,1,y]
x2 = [-0.6x+1,1,y] #change to 0.6 to 0.5 to get a non unique solution
@constraint(m, c1, x1 in MOI.ExponentialCone())
@constraint(m, c2, x2 in MOI.ExponentialCone()) 
@constraint(m, c3, x==0)
@objective(m, Min, y);


# MIO model driven by Gurobi
mio = Model(optimizer_with_attributes(
            () -> Gurobi.Optimizer(GRB_ENV), #"OutputFlag" => 0
        ));

# Dual cuts
function dual_cuts(cb_data)
    xval = callback_value(cb_data, x)
    #println(xval)
    set_normalized_rhs(c3, xval)
    optimize!(m)
    duals = [dual(c1), dual(c2)]
    xconic = [x1, x2]
    for (idx, d) in enumerate(duals)
        if norm(d) >= 1e-4
            con = @build_constraint(dot(d, xconic[idx])>=0)
            MOI.submit(mio, MOI.LazyConstraint(cb_data), con)
        end
    end
end

@variable(mio, x>=0, Int)  # Positivity constraints from conic model
@variable(mio, y>=0)       # Positivity constraints from conic model
MOI.set(mio, MOI.LazyConstraintCallback(), dual_cuts)
@objective(mio, Min, y)
optimize!(mio)

value(x) # Should be 2, time to run ~ 0.1sec