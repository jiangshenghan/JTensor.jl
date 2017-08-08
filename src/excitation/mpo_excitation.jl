
"""
Given the ground state of an infinite translationally invariant MPO operator, obtain the (topologically trivial) excited states with momentum p

excition ansatz can be represented as
                 ... ---Al---B---Ar--- ...
sum_n exp(ipn) x        |    |   |
                            (sn)

For tensor T, legs order as (left,right,up,down)
legs orders for Al,Ar,B are (left,right,down)

returns
"""
function mpo_excitation(T_init,p,Al,Ar,Fl,Fr,λ;elemtype=Complex128)

    print("MPO excitation!")

    #initialization
    chi=size(Al,1)
    @show chi,p
    Dh,Dv=size(T,1,3)
    tol=1e-12
    T=T_init/λ #normalize MPO
    B=rand(elemtype,chi,chi,Dv)
    B=jcontract([B,conj(Al),Al],[[1,-2,2],[1,3,2],[-1,3,-3]]) #fix left gauge of B

    #solve eigen equation to obtain B
    #=TODO: 
    1. Use bicgstab in IterativeSolver.jl to obtain LB and RB
    =#
    Teff=MPO_Heff(...)
    λB,B,_,Bniter,Bnmult=eigs(Teff,nev=1,v0=B[:],tol=tol)
    λB=λB[1]
    B=B[:,1]
    B=reshape(B[:],chi,chi,Dv)


    #check if B in the left gauge
    @show norm(B[:]-jcontract([B,conj(Al),Al],[[1,-2,2],[1,3,2],[-1,3,-3]])[:]) #fix left gauge of B

    return B
end
