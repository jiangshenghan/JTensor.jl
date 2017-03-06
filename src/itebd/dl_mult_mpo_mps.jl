
"""
Apply MPO to MPS, and obtained the truncated MPS with canonical form

---A[i-1]---A[i]---A[i+1]---
   ||       ||     ||
===TT[i-1]==TT[i]==TT[i+1]==   
   ||       ||     ||

Here, TT lable double layer tensor. For single layer tensors T, legs order as (phys,left,right,up,down)
legs orders for A are (left,right,down_ket,down_bra)

returns (B,C,Fl,Fr)
"""
function dl_mult_mpo_mps(A,T,chi,Fl=[],Fr=[];ep=1e-8,elemtype=Complex128,ncv=20)

    #initialization
    N=size(T,1)
    d,Dh,Dv=size(T[1],1,2,4)
    DA=size(A[1],1)
    @printf("chi=%d, ep=%e \n N=%d, d=%d, Dh=%d, Dv=%d, DA=%d\n",chi,ep,N,d,Dh,Dv,DA)

    Tc=conj(T)
    if Fl==[] Fl=rand(elemtype,DA,Dh,Dh,DA,Dh,Dh) end
    if Fr==[] Fr=rand(elemtype,DA,Dh,Dh,DA,Dh,Dh) end

    #left fixed point
    left_tensor_list=[]
    push!(left_tensor_list,Fl)
    for il=1:N
        append!(left_tensor_list,[A[il],T[il],Tc[il],Tc[il],T[il],conj(A[il])])
    end
    left_legs_list=[]
    push!(left_legs_list,[1,2,3,4,5,6])
    for il=1:N
        legs_list_il=[[1,15,7,8],[9,2,16,7,10],[9,3,17,8,11],[12,5,19,13,10],[12,6,20,14,11],[4,18,13,14]]
        legs_list_il+=(il-1)*14
        if il==N
            legs_list_il[1][2]=-1
            legs_list_il[2][3]=-2
            legs_list_il[3][3]=-3
            legs_list_il[4][3]=-5
            legs_list_il[5][3]=-6
            legs_list_il[6][2]=-4
        end
        append!(left_legs_list,legs_list_il)
    end
    leftlm=LinearMap(left_tensor_list,left_legs_list,1,elemtype=elemtype)
    leig_res=eigs(leftlm,nev=1,v0=Fl[:],tol=max(ep,1e-15),ncv=ncv)
    λl,Fl=leig_res[1:2]
    λl=λl[1]
    Fl=reshape(Fl,DA*Dh^2,DA*Dh^2)
    Fl=1/2*(Fl+Fl')
    L=eigfact(Fl)
    println("left eigs:")
    println(L[:values])
    L=reshape(L[:vectors]*diagm(sqrt(abs(L[:values]))),DA,Dh,Dh,DA*Dh^2)
    L=permutedims(L,[4,1,2,3])

    #right fixed point
    right_tensor_list=[]
    push!(right_tensor_list,Fr)
    for ir=N:-1:1
        append!(right_tensor_list,[A[ir],T[ir],Tc[ir],Tc[ir],T[ir],conj(A[ir])])
    end
    right_legs_list=[]
    push!(right_legs_list,[1,2,3,4,5,6])
    for ir=1:N
        legs_list_ir=[[15,1,7,8],[9,16,2,7,10],[9,17,3,8,11],[12,19,5,13,10],[12,20,6,14,11],[18,4,13,14]]
        legs_list_ir+=(ir-1)*14
        if ir==N
            legs_list_ir[1][1]=-1
            legs_list_ir[2][2]=-2
            legs_list_ir[3][2]=-3
            legs_list_ir[4][2]=-5
            legs_list_ir[5][2]=-6
            legs_list_ir[6][1]=-4
        end
        append!(right_legs_list,legs_list_ir)
    end
    rightlm=LinearMap(right_tensor_list,right_legs_list,1,elemtype=elemtype)
    reig_res=eigs(rightlm,nev=1,v0=Fr[:],tol=max(ep,1e-15),ncv=ncv)
    λr,Fr=reig_res[1:2]
    λr=λr[1]
    Fr=reshape(Fr,DA*Dh^2,Dh^2*DA)
    Fr=1/2*(Fr+Fr')
    R=eigfact(Fr)
    println("right eigs:")
    println(R[:values])
    R=reshape(R[:vectors]*diagm(sqrt(abs(R[:values]))),DA,Dh,Dh,DA*Dh^2)
    R=permutedims(R,[4,1,2,3])

    Fl=reshape(Fl,DA,Dh,Dh,DA,Dh,Dh)
    Fr=reshape(Fr,DA,Dh,Dh,DA,Dh,Dh)

    @printf("eig info: \n λl=%f+i%f \n λr=%f+i%f \n lniter=%d, lnmult=%d \n rniter=%d, rnmult=%d \n",real(λl),imag(λl),real(λr),imag(λr),leig_res[4],leig_res[5],reig_res[4],reig_res[5])

    #truncate singular value
    svdres=svdfact(jcontract([L,R],[[-1,1,2,3],[-2,1,2,3]]))
    U,S,Vt=svdres[:U],svdres[:S],svdres[:Vt]
    chic=chi
    while (chic<DA*Dh^2)
        if (S[chic]-S[chic+1])/S[chic]<1e-5
            chic+=1
        else break
        end
    end
    chic=min(chic,DA*Dh^2)
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

    LN=jcontract([conj(U),L],[[1,-1],[1,-2,-3,-4]])
    R1=jcontract([conj(Vt),R],[[-1,1],[1,-2,-3,-4]])
    LNinv=jcontract([R1,diagm(Sinv)],[[1,-2,-3,-4],[1,-1]])
    R1inv=jcontract([diagm(Sinv),LN],[[-1,1],[1,-2,-3,-4]])

    B=[zeros(elemtype,chic,chic,Dv,Dv) for i=1:N]
    C=[S for i=1:N]

    #one site
    if (N==1)
        B[1]=jcontract([R1inv,A[1],T[1],Tc[1],LNinv],[[-1,1,2,3],[1,7,4,5],[6,2,8,4,-3],[6,3,9,5,-4],[-2,7,8,9]])
        B[1]/=vecnorm(jcontract([diagm(C[1]),B[1]],[[-1,1],[1,-2,-3,-4]]))/sqrt(chic) #normalization such that Bl,Br (almost) unitary
        println("singular values:")
        println(C[1])
        println()
        return B,C,Fl,Fr
    end

    #canonical forms for mult sites
    #gamma_tensor= --LN==ATT==R1--
    #                   /...\
    gamma_tensor=LN
    gamma_legs=[-1,1,2,3]
    for ig=1:N
        gamma_tensor=jcontract([gamma_tensor,A[ig],T[ig],Tc[ig]],[gamma_legs,[1,-2*ig-2,4,5],[6,2,-2*ig-3,4,-2*ig],[6,3,-2*ig-4,5,-2*ig-1]])
        splice!(gamma_legs,2ig:2ig-1,[-2*ig,-2*ig-1])
    end
    gamma_tensor=jcontract([gamma_tensor,R1],[gamma_legs,[-2N-2,1,2,3]])
    
    for ig=1:N-1
        gamma_svd=svdfact(reshape(gamma_tensor,chic*Dv^2,div(length(gamma_tensor),chic*Dv^2)))
        P=reshape(gamma_svd[:U][:,1:chic],chic,Dv,Dv,chic)
        B[ig]=jcontract([diagm(C[ig==1?N:ig-1].\1),P],[[-1,1],[1,-3,-4,-2]])
        C[ig]=(gamma_svd[:S]/norm(gamma_svd[:S]))[1:chic]
        @printf("singular values at %d: \n",ig)
        println(C[ig])

        if ig<N-1
            gamma_tensor=diagm(C[ig])*gamma_svd[:Vt][1:chic,:]
        else
            B[N]=jcontract([reshape(gamma_svd[:Vt][1:chic,:],chic,Dv,Dv,chic),diagm(Sinv)],[[-1,-3,-4,1],[1,-2]])
            @printf("singular values at %d: \n",N)
            println(C[N])
        end
    end

    println()
    flush(STDOUT)

    return B,C,Fl,Fr
end
