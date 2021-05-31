using LinearAlgebra

function grasta_step(U,v,w,η,ρ,iters,mask = Vector{Bool}(true,size(v));assume_orthogonal = true)

    vsub = @view v[mask]


    #that's a lot of copying
    Usub = U[mask,:] #using views was causing issues with pinv :(

    nsub,d = size(Usub)
    s = zeros(nsub)
    y = zeros(nsub)
    

    bg_guess = larb_orthogonal_alt(Usub,vsub,w,s,y,ρ,iters,assume_orthogonal)

    Γ1 = y .+ ρ * (bg_guess .- vsub .+ s) #3.8
    Γ2 = Usub' * Γ1 #3.9
    #3.10 split into steps for efficiency
    Γ = -1 * U * Γ2
    Γ[mask] .+ Γ1

    norm_w = norm(w)

    σ = norm(Γ) * norm_w

    #3.13
    #@show size(U * w)
    #@show size(Γ)
    U .+=  ((cos(η * σ) .- 1) * (U * w) / norm_w^2 .- sin(η * σ) * Γ/σ) * w'



end