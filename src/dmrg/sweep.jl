"""
Sweep for finite DMRG algorithm

dir=1, start from right canonical assumed, sweep from left to right
dir=-1, start from left canonical assumed, sweep from right to left

MPS A, effective Hamiltonian Fl, Fr will be modified after use this function
"""

function sweep_dmrg!(A,T,Fl,Fr,χ;dir=1,elemtype=Complex128)
    L=length(A)
    @show dir
    if dir==1
        for j=1:L-1
            A[j],A[j+1],Fl[j+1],_,_=one_step_optimization_dmrg_sweep(j,A,T,Fl,Fr,χ,dir=dir,elemtype=elemtype)
        end
    elseif dir==-1
        for j=L-1:-1:1
            A[j],A[j+1],Fr[j],_,_=one_step_optimization_dmrg_sweep(j,A,T,Fl,Fr,χ,dir=dir,elemtype=elemtype)
        end
    end
end

"""
Perform one step optimization for site (js,js+1), js=1,...,L-1

#     ----- A2c ------
#    /    /    \       \             --A2c--
# Fl[js]-T[js]-T[js+1]-Fr[js+1] = λ   /  \
#   \    |     |       /

cut-off bond dim χ

svd decomp:
# --A2c-- = --A[js]--A[js+1]--
#  / \        |      |

for dir=1, (js,js+1)->(js+1,js+2), tensor at js from non-canonical to left-canonical, tensor at js+1 from right-canonical to non-canonical

for dir=-1, (js,js+1)->(js-1,js), tensor at js+1 from non-canonical to right-canonical, tensor at js from left-canonical to non-canonical

legs of T: lrud
legs of A: lrd
legs of Fl,Fr: umd
legs of A2c: l,d(js),d(js+1),r

update A[js],A[js+1],Fl[js+1](dir=1),Fr[js](dir=-1)

return (Al_update,Ar_update,F_update,λ,S)
"""
function one_step_optimization_dmrg_sweep(js,A,T,Fl,Fr,χ;dir=1,elemtype=Complex128,ncv=20)
    @show js
    d,Dl=size(A[js])[[3,1]];
    Dc,Dr=size(A[js+1])[[1,2]]
    L=length(A)

    #obtain updated two site tensor
    A2c=jcontract([A[js],A[js+1]],[[-1,1,-2],[1,-4,-3]])
    Heff=LinearMap([Fl[js],A2c,T[js],T[js+1],Fr[js+1]],[[1,2,-1],[1,3,4,5],[2,7,3,-2],[7,6,4,-3],[5,6,-4]],2,elemtype=elemtype) 
    eigs_res=eigs(Heff,nev=1,v0=A2c[:],ncv=ncv,which=:SR)
    λ,A2c=eigs_res[1:2]
    λ=λ[1];
    A2c=reshape(A2c,Dl*d,d*Dr)
    print("total energy: ")
    @show λ
    print("energy_per_site:  ")
    @show λ/L

    #svd decomp and cut bond dim
    svdres=svdfact(A2c)
    U,S,Vt=svdres[:U],svdres[:S],svdres[:Vt]
    χc=min(χ,length(S))
    S=S[1:χc]
    S=S/norm(S)
    U=U[:,1:χc]
    Vt=Vt[1:χc,:]
    println("singular value:")
    @show χc,S
    #calculate entanglement entropy between site js and site js+1
    entropy=sum(x->-x^2*2*log(x),S)
    @show entropy
    if χc==χ 
        cut_err=S[end] 
        @show cut_err
    end
    println()

    #obtain updated A[js], A[js+1], Fl[js+1](dir=1), Fr[js](dir=-1)
    Al_update=[]
    Ar_update=[]
    F_update=[]
    if dir==1 #step in left to right sweep
        Al_update=permutedims(reshape(U,Dl,d,χc),[1,3,2])
        Ar_update=permutedims(reshape(diagm(S)*Vt,χc,d,Dr),[1,3,2])
        F_update=jcontract([Fl[js],Al_update,T[js],conj(Al_update)],[[1,2,3],[1,-1,4],[2,-2,4,5],[3,-3,5]])
    elseif dir==-1 #step in right to left sweep
        Al_update=permutedims(reshape(U*diagm(S),Dl,d,χc),[1,3,2])
        Ar_update=permutedims(reshape(Vt,χc,d,Dr),[1,3,2])
        F_update=jcontract([Fr[js+1],Ar_update,T[js+1],conj(Ar_update)],[[1,2,3],[-1,1,4],[-2,2,4,5],[-3,3,5]])
    end

    return Al_update,Ar_update,F_update,λ,S
end
