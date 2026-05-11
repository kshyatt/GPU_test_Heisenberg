module Preconditioning

using PEPSKit, TensorKit, OptimKit, KrylovKit

# Define a custom preconditioner that updates based on the previous two energy values and the gradient norm
mutable struct DynamicPreconditioner
    E_prev    :: Union{Nothing, Float64}
    E_prevprev:: Union{Nothing, Float64}
    grad_norm :: Union{Nothing, Float64}

    DynamicPreconditioner() = new(nothing, nothing, nothing)
end

# Update the preconditioner state with the latest energy and gradient norm
function update_precond!(P::DynamicPreconditioner, E::Float64, grad_norm::Float64)
    P.E_prevprev = P.E_prev
    P.E_prev     = E
    P.grad_norm  = grad_norm
    return P
end

# This function applies the preconditioner to the input gradient `g_in`
# for a specific site, using the environment and the previous energy values to
# compute a new gradient.
function _apply_P(g_in, env, δ, site, norm_pref)
    (r, c) = site

    @tensor opt=true Ng[-1; -2 -3 -4 -5] :=
        g_in[-1; 3 11 7 4] *
        env.corners[1, mod1(r-1,end), mod1(c-1,end)][1; 2] *
        env.edges[1,   mod1(r-1,end), mod1(c,  end)][2 3 -2; 8] *
        env.corners[2, mod1(r-1,end), mod1(c+1,end)][8; 10] *
        env.edges[2,   mod1(r,  end), mod1(c+1,end)][10 11 -3; 12] *
        env.corners[3, mod1(r+1,end), mod1(c+1,end)][12; 9] *
        env.edges[3,   mod1(r+1,end), mod1(c,  end)][9 7 -4; 6] *
        env.corners[4, mod1(r+1,end), mod1(c-1,end)][6; 5] *
        env.edges[4,   mod1(r,  end), mod1(c-1,end)][5 4 -5; 1]

    return Ng / norm_pref + δ * g_in
end

# The main function that applies the preconditioner to the entire gradient `g`.
# It iterates over each site in the PEPS, computes the new gradient using `_apply_P`,
# and returns the updated gradient. If there are not enough previous energy values,
# it simply returns the original gradient.
function (P::DynamicPreconditioner)(x, g)
    # Skip until we have two energy values to compare
    (P.E_prev === nothing || P.E_prevprev === nothing) && return g

    peps, env = x[1], x[2]
    Lx, Ly   = size(peps)
    g_new    = copy(g)
    δ        = max(1e-9, (P.grad_norm)^2)

    for r in 1:Lx, c in 1:Ly
        norm_pref = PEPSKit._contract_site((r, c), InfiniteSquareNetwork(peps), env)
        op(v)     = _apply_P(v, env, δ, (r, c), norm_pref)

        g_out, _ = KrylovKit.linsolve(
            op, g[r,c], g[r,c];
            maxiter   = 5,
            krylovdim = 60,
            isposdef  = true,
            verbosity = KrylovKit.SILENT_LEVEL,
        )
        g_new[r, c] = g_out
    end

    return g
end

const _PRECONDITIONER = Ref{Union{Nothing, DynamicPreconditioner}}(nothing)
OptimKit._precondition(x, g) = _PRECONDITIONER[] === nothing ? g : _PRECONDITIONER[](x, g)

"""
    make_finalize(P) -> finalize!

Installs `P` as the global OptimKit preconditioner and returns a `finalize!`
closure that updates `P` at the end of each optimizer iteration.
"""
function make_finalize(P::DynamicPreconditioner)   
    _PRECONDITIONER[] = P
    @info "Custom preconditioner installed in OptimKit."

    function finalize!(x, f, g, numiter)
        println("Iteration $numiter — updating preconditioner")
        update_precond!(P, f, norm(g))
        return x, f, g
    end

    return finalize!
end


end