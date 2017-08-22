#TODO:Debug

"""
Given the ground state of an infinite translationally invariant MPO operator, obtain n=nev (topologically trivial) excited states with momentum p

excition ansatz can be represented as
                 ... ---Al---B---Ar--- ...
sum_n exp(ipn)          |    |   |
                            (sn)

For tensor T, legs order as (left,right,up,down)
legs orders for Al,Ar,B are (left,right,down)

B is fixed at left gauge
 ---B----     ---Vl--XB-
 |  |      =  |  |       = 0
 ---Al*--     ---Al*--
 leg order of XB is (left,right)

returns X
"""
function mpo_excitation(T,p,Al,Ar,C,Fl,Fr,λ;elemtype=Complex128,nev=3)

    print("MPO excitation!")

    #initialization
    chi=size(Al,1)
    @show chi,p
    Dh,Dv=size(T,1,3)
    tol=1e-12
    Ml=reshape(permutedims(Al,[3,1,2]),chi,:)
    Vl=nullspace(Ml)
    Vl=permutedims(reshape(Vi,chi,Dv,:),[1,3,2])
    #Fr=Fr/jcontract([Fl,Fr],[[1,2,3],[1,2,3]]) #normalize Fl,Fr

    #solve eigen equation to obtain B
    XB=rand(elemtype,size(Vl,2)*chi)
    Teff=MPO_Heff([T/λ,Al,Ar,C,Fl,Fr,Vl],p)
    λB,XB,_,Bniter,Bnmult=eigs(Teff,nev=nev,v0=XB,tol=tol)

    return λB,XB
end
