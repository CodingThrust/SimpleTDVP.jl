"""
    tdvp1sweep(dt, A, H [, F; kwargs...])

Run one sweep (left to right and back) of the one-site TDVP algorithm, updates `A` and `F` in
place, returns MPS tensors `A` and environments `F`. Assumes A starts in right
canonical form.
"""
function tdvp1sweep!(dt, A, H, F = nothing; verbose = true, kwargs...)
    N = length(A)

    if F == nothing
        F = Vector{Any}(undef, N+2)
        F[1] = fill!(similar(H[1], (1,1,1)), 1)
        F[N+2] = fill!(similar(H[1], (1,1,1)), 1)
        for k = N:-1:1
            F[k+1] = updaterightenv(A[k], M[k], F[k+2])
        end
    end

    AC = A[1]
    for k = 1:N-1
        AC, info = exponentiate(x->applyH1(x, F[k], F[k+2], M[k]), -im*dt, AC; ishermitian = true, kwargs...)

        if verbose
            E = dot(AC, applyH1(AC, F[k], F[k+2], M[k]))
            println("Sweep L2R: AC site $k -> energy $E")
        end

        AL, C = qr(reshape(AC, size(AC,1)*size(AC,2), :))
        A[k] = reshape(Matrix(AL), size(AC))
        F[k+1] = updateleftenv(A[k], M[k], F[k])

        # TODO: backward evolution of C

        if verbose
            E = dot(C, applyH0(C, F[k+1], F[k+2]))
            println("Sweep L2R: C between site $k and $(k+1) -> energy $E")
        end

        @tensor AC[-1,-2,-3] := C[-1,1] * A[k+1][1,-2,-3]
    end
    k = N
    AC, info = exponentiate(x->applyH1(x, F[k], F[k+2], M[k]), -im*dt, AC; ishermitian = true, kwargs...)

    if verbose
        E = dot(AC, applyH1(AC, F[k], F[k+2], M[k]))
        println("Sweep L2R: AC site $k -> energy $E")
    end

    for k = N-1:-1:1
        C, AR = lq(reshape(AC, size(AC,1), :))
        # it's actually better to do qr of transpose and transpose back

        A[k+1] = reshape(Matrix(AR), size(AC))
        F[k+2] = updaterightenv(A[k+1], M[k+1], F[k+3])

        # TODO: backward evolution of C

        if verbose
            E = dot(C, applyH0(C, F[k+1], F[k+2]))
            println("Sweep R2L: C between site $k and $(k+1) -> energy $E")
        end

        @tensor AC[:] := A[k][-1,-2,1] * C[1,-3]
        AC, info = exponentiate(x->applyH1(x, F[k], F[k+2], M[k]), -im*dt, AC; ishermitian = true, kwargs...)

        if verbose
            E = dot(AC, applyH1(AC, F[k], F[k+2], M[k]))
            println("Sweep R2L: AC site $k -> energy $E")
        end
    end
    A[1] = AC
    return A, F
end