
"""
Apply MPO to MPS, and obtained the truncated MPS with canonical form

---A[i-1]---A[i]---A[i+1]---
   |        |      |
---T[i-1]---T[i]---T[i+1]---   
   |        |      |

For tensor T, legs order as (left,right,up,down)
legs orders for A are (left,right,down)

convention:
Bl[i]=diagm(C[i-1])*B[i]
Br[i]=B[i]*diagm(C[i])

returns (B,C)
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

    @printf("chic=%d\n",chic)
    println("S:")
    println(S)
    println("Sinv:")
    println(Sinv)

    LN=jcontract([conj(U),L],[[1,-1],[1,-2,-3]])
    R1=jcontract([conj(Vt),R],[[-1,1],[1,-2,-3]])
    LNinv=jcontract([R1,diagm(Sinv)],[[1,-2,-3],[1,-1]])
    R1inv=jcontract([diagm(Sinv),LN],[[-1,1],[1,-2,-3]])

    B=[zeros(elemtype,chic,chic,Dv) for i=1:N]
    C=[S for i=1:N]

    #one site
    if (N==1)
        B[1]=jcontract([R1inv,A[1],T[1],LNinv],[[-1,1,2],[1,4,3],[2,5,3,-3],[-2,4,5]])
        B[1]/=vecnorm(jcontract([diagm(C[1]),B[1]],[[-1,1],[1,-2,-3]]))/sqrt(chic) #normalization such that Bl,Br (almost) unitary
        println("singular values:")
        println(C[1])
        println()
        return B,C
    end
    
    #get canonical form for mult sites
    #gamma_tensor= --LN==AT==R1--
    #                   /..\
    gamma_tensor=LN
    gamma_legs=[-1,1,2]
    for ig=1:N
        gamma_tensor=jcontract([gamma_tensor,A[ig],T[ig]],[gamma_legs,[1,-ig-2,3],[2,-ig-3,3,-ig-1]])
        insert!(gamma_legs,ig+1,-ig-1)
    end
    gamma_tensor=jcontract([gamma_tensor,R1],[gamma_legs,[-N-2,1,2]])

    for ig=1:N-1
        gamma_svd=svdfact(reshape(gamma_tensor,chic*Dv,div(length(gamma_tensor),chic*Dv)))
        P=reshape(gamma_svd[:U][:,1:chic],chic,Dv,chic)
        B[ig]=jcontract([diagm(C[ig==1?N:ig-1].\1),P],[[-1,1],[1,-3,-2]])
        C[ig]=(gamma_svd[:S]/norm(gamma_svd[:S]))[1:chic]
        @printf("singular values at %d: \n",ig)
        println(C[ig])

        if ig<N-1
            gamma_tensor=diagm(C[ig])*gamma_svd[:Vt][1:chic,:]
        else
            B[N]=jcontract([reshape(gamma_svd[:Vt][1:chic,:],chic,Dv,chic),diagm(Sinv)],[[-1,-3,1],[1,-2]])
            @printf("singular values at %d: \n",N)
            println(C[N])
        end
    end

    println()

    return B,C
end

