
"""
Apply MPO to MPS, and obtained the truncated MPS with canonical form

---A[i-1]---A[i]---A[i+1]---
   ||       ||     ||
===TT[i-1]==TT[i]==TT[i+1]==   
   ||       ||     ||

Here, TT lable double layer tensor. For single layer tensors T, legs order as (phys,left,right,up,down)
legs orders for A are (left,right,down_ket,down_bra)

returns (Bl,Br,C)
"""
function dl_mult_mpo_mps(A,T,chi,Fl=[],Fr=[];ep=1e-8,elemtype=Complex128,ncv=40)

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
    leigs_res=eigs(leftlm,nev=1,v0=Fl[:],tol=max(ep,1e-15),ncv=ncv)
    λl,Fl=leigs_res[1:2]
    λl=λl[1]
    Fl=reshape(Fl,DA*Dh^2,DA*Dh^2)
    Fl=1/2*(Fl+Fl')
    L=cholfact(Fl)
    L=reshape(transpose(L[:L]),DA*Dh^2,DA,Dh,Dh)

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
    reigs_res=eigs(rightlm,nev=1,v0=Fr[:],tol=max(ep,1e-15),ncv=ncv)
    λr,Fr=reigs_res[1:2]
    λr=λr[1]
    Fr=reshape(Fr,DA*Dh^2,Dh^2*DA)
    Fr=1/2*(Fr+ctranspose(Fr))
    R=cholfact(Fr)
    R=reshape(transpose(R[:L]),DA*Dh^2,DA,Dh,Dh)

    @printf("eig iter info: \n lniter=%d, lnmult=%d \n rniter=%d, rnmult=%d \n",leig_res[4],leig_res[5],reig_res[4],reig_res[5])

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
    U=U[:,1:chic]
    S=S[1:chic]
    Vt=Vt[1:chic,:]
    Sinv=S.\1

    L=jcontract([conj(U),L],[[1,-1],[1,-2,-3,-4]])
    R=jcontract([conj(Vt),R],[[-1,1],[1,-2,-3,-4]])
    C=[diagm(S) for i=1:N]

    #TODO: chic can be position dependent
    #left (not strictly) canonical 
    copy!(Bl,A)
    Ls=L
    for il=1:N-1
        LATT=jcontract([Ls,A[il],T[il],Tc[il]],[[-1,1,2,3],[1,-4,4,5],[6,2,-5,4,-2],[6,3,-6,5,-3]])
        resil=svdfact(reshape(LATT,chic*Dv^2,DA*Dh^2))
        Bl[il]=permutedims(reshape(resil[:U][:,1:chic],chic,Dv,Dv,chic),[1,4,2,3])
        C[il]=diagm(resil[:S][1:chic])
        Ls=C[il]*resil[:Vt][1:chic,:]
        Ls=reshape(Ls,chic,DA,Dh,Dh)
    end
    Bl[N]=jcontract([Ls,A[N],T[N],Tc[N],R,diagm(Sinv)],[[-1,1,2,3],[1,7,4,5],[6,2,8,4,-3],[6,3,9,5,-4],[10,7,8,9],[10,-2]])

    #right canonical
    copy!(Br,A)
    Rs=R
    for ir=N:-1:2
        RATT=jcontract([Rs,A[ir],T[ir],Tc[ir]],[[-1,1,2,3],[-4,1,4,5],[6,-5,2,4,-2],[6,-6,3,5,-3]])
        resir=svdfact(reshape(RATT,chic*Dv^2,DA*Dh^2))
        Br[ir]=permutedims(reshape(resir[:U][:,1:chic],chic,Dv,Dv,chic),[4,1,2,3])
        Rs=diagm(resir[:S][1:chic])*resir[:Vt][1:chic,:]
        Rs=reshape(Rs,chic,DA,Dh,Dh)
    end
    Br[1]=jcontract([diagm(Sinv),L,A[1],T[1],Tc[1],Rs],[[-1,1],[1,2,3,4],[2,8,5,6],[7,3,9,5,-3],[7,4,10,6,-4],[-2,8,9,10]])

    println("singular values:")
    for i=1:N println(diag(C[i])) end
    

    return Bl,Br,C
end
