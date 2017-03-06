
"""
Apply MPO to MPS, and obtained the truncated MPS with canonical form

---A[i-1]---A[i]---A[i+1]---
   |        |      |
---T[i-1]---T[i]---T[i+1]---   
   |        |      |

For tensor T, legs order as (left,right,up,down)
legs orders for A are (left,right,down)

convention:
--Bl[i]--C[i]--Br[i+1]-- 
  ||     ||    ||

returns (Bl,Br,C)
"""
function sl_mult_mpo_mps(A,T,chi,Fl=[],Fr=[];ep=1e-8,elemtype=Complex128,ncv=20)

    #initialization
    N=size(T,1)
    Dh,Dv=size(T[1],1,3)
    DA=size(A[1],1)
    @printf("chi=%d, ep=%e \n N=%d, Dh=%d, Dv=%d, DA=%d\n",chi,ep,N,Dh,Dv,DA)

    Tc=conj(T)
    if Fl==[] Fl=rand(elemtype,DA,Dh,DA,Dh) end
    if Fr==[] Fr=rand(elemtype,DA,Dh,DA,Dh) end

    #left fixed point
    left_tensor_list=[]
    push!(left_tensor_list,Fl)
    for il=1:N
        append!(left_tensor_list,[A[il],T[il],Tc[il],conj(A[il])])
    end
    left_legs_list=[]
    push!(left_legs_list,[1,2,3,4])
    for il=1:N
        legs_list_il=[[1,8,5],[2,9,5,6],[4,11,7,6],[3,10,7]]
        legs_list_il+=(il-1)*7
        if il==N
            legs_list_il[1][2]=-1
            legs_list_il[2][2]=-2
            legs_list_il[3][2]=-4
            legs_list_il[4][2]=-3
        end
        append!(left_legs_list,legs_list_il)
    end
    leftlm=LinearMap(left_tensor_list,left_legs_list,1,elemtype=elemtype)
    leig_res=eigs(leftlm,nev=1,v0=Fl[:],tol=max(ep,1e-15),ncv=ncv)
    λl,Fl=leig_res[1:2]
    λl=λl[1]
    Fl=reshape(Fl,DA*Dh,DA*Dh)
    Fl=1/2*(Fl+Fl')
    L=eigfact(Fl)
    println("left eigs:")
    println(L[:values])
    L=reshape(L[:vectors]*diagm(sqrt(abs(L[:values]))),DA,Dh,DA*Dh)
    L=permutedims(L,[3,1,2])

    #right fixed point
    right_tensor_list=[]
    push!(right_tensor_list,Fr)
    for ir=N:-1:1
        append!(right_tensor_list,[A[ir],T[ir],Tc[ir],conj(A[ir])])
    end
    right_legs_list=[]
    push!(right_legs_list,[1,2,3,4])
    for ir=1:N
        legs_list_ir=[[8,1,5],[9,2,5,6],[11,4,7,6],[10,3,7]]
        legs_list_ir+=(ir-1)*7
        if ir==N
            legs_list_ir[1][1]=-1
            legs_list_ir[2][1]=-2
            legs_list_ir[3][1]=-4
            legs_list_ir[4][1]=-3
        end
        append!(right_legs_list,legs_list_ir)
    end
    rightlm=LinearMap(right_tensor_list,right_legs_list,1,elemtype=elemtype)
    reig_res=eigs(rightlm,nev=1,v0=Fr[:],tol=max(ep,1e-15),ncv=ncv)
    λr,Fr=reig_res[1:2]
    λr=λr[1]
    Fr=reshape(Fr,DA*Dh,DA*Dh)
    Fr=1/2*(Fr+Fr')
    R=eigfact(Fr)
    println("right eigs:")
    println(R[:values])
    R=reshape(R[:vectors]*diagm(sqrt(abs(R[:values]))),DA,Dh,DA*Dh)
    R=permutedims(R,[3,1,2])

    @printf("eig info: \n λl=%f+i%f \n λr=%f+i%f \n lniter=%d, lnmult=%d \n rniter=%d, rnmult=%d \n",real(λl),imag(λl),real(λr),imag(λr),leig_res[4],leig_res[5],reig_res[4],reig_res[5])

    #truncate singular value
    svdres=svdfact(jcontract([L,R],[[-1,1,2],[-2,1,2]]))
    U,S,Vt=svdres[:U],svdres[:S],svdres[:Vt]
    chic=chi
    while (chic<DA*Dh)
        if (S[chic]-S[chic+1])/S[chic]<1e-5
            chic+=1
        else break
        end
    end
    chic=min(chic,DA*Dh)
    U=U[:,1:chic]
    S=S/norm(S)
    S=S[1:chic]
    Vt=Vt[1:chic,:]
    Sinv=S.\1

    L=jcontract([conj(U),L],[[1,-1],[1,-2,-3]])
    R=jcontract([conj(Vt),R],[[-1,1],[1,-2,-3]])
    Linv=jcontract([R,diagm(Sinv)],[[1,-2,-3],[1,-1]])
    Rinv=jcontract([diagm(Sinv),L],[[-1,1],[1,-2,-3]])
    C=[diagm(S) for i=1:N]

    #TODO:modify the following algorithm
    #TODO: chic can be position dependent
    #left (not strictly) canonical 
    Bl=[zeros(elemtype,chic,chic,Dv) for il=1:N]
    Ls=[zeros(elemtype,chic,DA,Dh) for il=1:N]
    Ls[N]=L
    for il=1:N-1
        LAT=jcontract([Ls[il==1?N:il-1],A[il],T[il]],[[-1,1,2],[1,-3,3],[2,-4,3,-2]])
        resil=svdfact(reshape(LAT,chic*Dv,DA*Dh))
        Bl[il]=permutedims(reshape(resil[:U][:,1:chic],chic,Dv,chic),[1,3,2])
        Ls[il]=diagm(resil[:S][1:chic])*resil[:Vt][1:chic,:]
        Ls[il]=reshape(Ls,chic,DA,Dh)
    end
    Bl[N]=jcontract([Ls[N-1],A[N],T[N],Linv],[[-1,1,2],[1,4,3],[2,5,3,-3],[-2,4,5]])
    Bl[N]/=norm(Bl[N])

    #right canonical
    Br=[zeros(elemtype,chic,chic,Dv) for ir=1:N]
    Rs=[zeros(elemtype,chic,DA,Dh) for ir=1:N]
    Rs[1]=R
    for ir=N:-1:2
        RAT=jcontract([Rs[il%N+1],A[ir],T[ir]],[[-1,1,2],[-3,1,3],[-4,2,3,-2]])
        resir=svdfact(reshape(RAT,chic*Dv,DA*Dh))
        Br[ir]=permutedims(reshape(resir[:U][:,1:chic],chic,Dv,chic),[3,1,2])
        Rs=diagm(resir[:S][1:chic])*resir[:Vt][1:chic,:]
        Rs=reshape(Rs,chic,DA,Dh)
    end
    Br[1]=jcontract([Rinv,A[1],T[1],Rs],[[-1,1,2],[1,4,3],[2,5,3,-3],[-2,4,5]])
    Br[1]/=norm(Br[1])

    @printf("chic=%d\n",chic)
    println("singular values:")
    for i=1:N println(diag(C[i])) end
    println()
    

    return Bl,Br,C
end

