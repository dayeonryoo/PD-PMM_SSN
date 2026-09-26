# Notes on the revised "Implementation details" section

`docs/implementation_details.tex` is a drop-in replacement for Section 4 of
`main_upgrade.tex`. It replaces everything from `\section{Implementation details}` up to
`\section{Numerical results}`. It was compiled inside the full paper (pdflatex, TeX Live 2023)
without errors, undefined references or overfull boxes.

## Structure

The section now goes from the big picture to the details:

| Subsection | Content |
|---|---|
| 4.1 Overview | setup phase (model construction, scaling, lifting, trivial check, initialization), the three nested loops, termination statuses |
| 4.2 PMM outer loop | initialization, the steps of one iteration in the order the code runs them, parameter updates |
| 4.3 SSN inner loop | active sets, Newton direction, exact linesearch with its fallback, convergence and stagnation tests |
| 4.4 PCG | matrix-free normal equations, stopping rule, warm start, iterative refinement, the three-stage failure handling |
| 4.5 Preconditioner factorizations | choice of callback, then four rules: reuse, low-rank update, in-place modification, refactorization |
| 4.6 Ruiz scaling and unscaling | scaling procedure, unscaled tests, definition of `res_SSN`, recovery of the solution |
| 4.7 Infeasibility detection | setup check, certificates, tolerances and thresholds |

Table 1 is grouped by component, gives each parameter's subsection, marks the three user-settable
parameters with a dagger, and has one value per row where that reads better.

Labels referenced elsewhere in the paper are unchanged: `sec: implementation details`,
`tab:params` and `subsec: algorithm settings`. The last one is now the PMM subsection, which is
what Section 2.2 points to. New labels: `subsec: implementation overview`,
`subsec: SSN implementation`, `subsec: PCG implementation`,
`subsec: preconditioner factorizations`, `subsec: scaling implementation`,
`subsec: infeasibility implementation`, `eqn: multiplier update safeguard` and
`eqn: descaled SSN residual`.

## Corrections relative to the previous draft

| # | Previous text | Code | Reference |
|---|---|---|---|
| 1 | Table: rho_0 = 10^2 | rho_0 = 10^7 (the old prose already said 10^7) | `include/ksp_qp.hpp:139` |
| 2 | Table: "N_o: ordering selection before locking = 3" | Ordering selection (AMD vs METIS) was removed in `0458de0`. The constant 3 now limits the number of Cholesky vs LDL^T callback selections, so it is renamed N_c | `include/ssn.tpp:642` |
| 3 | Table: eps_k bounds [tol, min{10^-2, r^max}] | eps_k stays in [tol, 10^-2]. r^max_{k+1} only caps the increase after a linesearch failure | `include/ksp_qp.tpp:702-715` |
| 4 | "If the PMM residual does not drop below 90%, (mu, rho) are scaled by 1.1" | Applies only when SSN ended neither `Optimal` nor `LineSearchFailed` (iteration cap reached or stagnated) | `include/ksp_qp.tpp:712` |
| 5 | Multiplier update if res_SSN <= 100 res_PMM ("current" residual); otherwise the duals "revert to cached values" | Compared with r^max_k, the residual of the iterate *before* the SSN solve. If the test fails, y1 and z are simply left unchanged | `include/ksp_qp.tpp:842-848` |
| 6 | Low-rank update if the number of changes is "less than 50", relative to the previous SSN iteration | Used for 0 < h+p+q <= 50, counted relative to the *most recent factorization*. Also rejected if mu, rho or the callback changed since that factorization, or if the LU of S_Lambda is rank-deficient | `include/schur_preconditioner.hpp:627-651, 694, 822` |
| 7 | After a PCG failure "the iterative Krylov method is completely abandoned" | Confirmed, and permanent for the rest of the solve, including the iterative-refinement solves | `include/ssn.tpp:160-162` |
| 8 | Low-rank updates suppressed "until a successful solve or a new PMM iteration" | Once suppressed, they stay off until the next PMM iteration. Before that point, the failure counter is also reset by a successful solve that used a low-rank-updated preconditioner | `include/ssn.hpp:305`, `include/ssn.tpp:258` |
| 9 | Order of operations in a PMM iteration not stated | SSN; total-SSN cap; accept (x, y2); safeguarded multiplier update; optimality test; infeasibility tests; parameter update; time and interrupt checks | `include/ksp_qp.tpp:917-990` |
| 10 | "Model construction" described as solver pre-processing | Done by the MPS/SIF interface (`to_kspqp`), with eps_eq = 1e-12 | `include/mps_format_parser.hpp:23` |

## Defaults that disagree across the code (you need to decide)

- **Time limit.** The `KSP_QP` member (`include/ksp_qp.hpp:128`), the Python binding
  (`python/ksp_qp_bind.cpp:127`) and the README all say 600 s. However, `Problem<T>` defaults to
  60 s (`include/problem.hpp:33`), and `KSP_QP` always copies this value from `Problem<T>`. The
  C++ drivers also default to 60 s. Table 1 says 600 s: either change `Problem<T>` to 600 s or
  change the table to 60 s.
- **Maximum PMM iterations.** `Problem<T>` and `KSP_QP` use 3000 (`include/problem.hpp:32`). The
  Python binding defaults to 10^9 (`python/ksp_qp_bind.cpp:126`). Table 1 says 3000.

## Other parts of the paper that disagree with the code (not changed)

- **Algorithm 1.** It does not show the safeguarded multiplier update or the infeasibility tests.
  It also updates the parameters before the convergence test; the code tests convergence first.
  This is harmless, but Section 4.2 now states the order the code actually uses.
- **Algorithm 2.** `r_max` should be `r^max_{k+1}`. The input line still contains `\spr{(tbd)}`.
- **Algorithm 3.** On stagnation, SSN returns the latest iterate u_{k_{j+1}}, not u_{k_j}
  (`include/ssn.tpp:867`).
- **Section 3.3.**
  - It treats P_{k_j} as the preconditioner of the current SSN iteration. In the code it is the
    most recent factorization, which may be several iterations old.
  - It says nothing is done when h+p+q = 0. In the code, if the active sets change and later
    return to those of that factorization (rank 0 relative to it), the preconditioner is
    refactorized from scratch rather than reused. Reusing it would be a small optimization.
- **Appendix A.2 (exact linesearch).** It says N = 0 gives tau = 1. The code returns tau = 1
  only when the direction is negligible (eta < 100 eps_mach). Otherwise it returns the exact
  minimizer of the quadratic, tau = -psi'(0)/eta (`include/ssn.tpp:508-553`). The two agree for
  an exact Newton direction, but not for inexact or steepest-descent directions.
- **Appendix A.4 (infeasibility).**
  - Delta x and Delta y2 are differences between consecutive PMM iterates, not between SSN
    iterates (`include/ksp_qp.tpp:918-919, 962, 966`).
  - The dual conditions on B mix ||Delta x||_inf and S_dinf. The code uses the unscaled norm
    S_dinf throughout (`include/ksp_qp.tpp:794-821`).
  - In condition 1, A, B and the Delta y's are the scaled quantities, since D_2^{-1} is applied
    outside.

## Possible issue in the code

When the multiplier update is skipped, `delta_y1` and `delta_z` keep the values of the last
update (`include/ksp_qp.tpp:848`). They are then combined with a fresh Delta y2 in the primal
certificate (`include/ksp_qp.tpp:962`). Section 4.7 describes this behaviour as it is. If it is
not intended, consider resetting them, or skipping the primal test, whenever the update is
skipped.
