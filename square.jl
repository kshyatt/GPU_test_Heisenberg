
using TensorKit, PEPSKit, OptimKit, Random, JLD2, LinearAlgebra, KrylovKit

# Define parameters
lattice = InfiniteSquare(2, 2)

# Implement Hamiltonian
H = heisenberg_XYZ(InfiniteSquare(2, 2); Jx = 1.0, Jy = 1.0, Jz = 1.0);


D_peps = 2
χ_env = 12

peps0 = PEPSKit.peps_normalize(InfinitePEPS(ComplexSpace(2), ComplexSpace(D_peps); unitcell=(2, 2)));
env0 = CTMRGEnv(peps0, ComplexSpace(χ_env));

boundary_alg = SimultaneousCTMRG(;
    tol = 1.0e-9, # 1e-8
    trunc = truncrank(χ_env),
    maxiter = 150,
);

gradient_alg = LinSolver(;
    solver_alg = KrylovKit.GMRES(;
        tol = 1e-9, maxiter = 1, verbosity = 3, krylovdim = 50,
    ),
    iterscheme = :fixed,
);

optimizer_alg = LBFGS(32; maxiter=5, gradtol=1e-5, verbosity=3, linesearch = HagerZhangLineSearch(;c₁ = 1e-4, verbosity = 2, maxiter=10, maxfg = 10))

println("-----------------")
println("PARAMETERS")
println("PEPS bond dimension: ", D_peps)
println("Environment bond dimension: ", χ_env)
println("-----------------")


# Preconditioning
include("Preconditioning.jl")
import .Preconditioning as pc
PC = pc.DynamicPreconditioner()
custom_finalize! = pc.make_finalize(PC)

algs = PEPSOptimize(;
    boundary_alg = boundary_alg,
    optimizer_alg = optimizer_alg,
    gradient_alg = gradient_alg,
    reuse_env=true,
);

println(">>> Starting fixedpoint...")
peps, env, E, info = fixedpoint(
    H, peps0, env0, algs; finalize! = custom_finalize!
);