

"""
Obtain fixed point for a double layer mpo with double unit cell, where the mpo is obtained by contracting one row of single layer square PEPS tensors. Here translational invriant is assumed

For the following case
---Al----Bl----Ac----Br----
   ||    ||    ||    ||
===TTA===TTB===TTA===TTB---
   ||    ||    ||    ||
Here, TTA,TTB lable double layer tensor. For single layer tensor TA,TB, legs order as (phys,left,right,up,down)
legs orders for Al,Ar,Bl,Br are (left,right,down_ket,down_bra)
ep indicates the precision (how far from the optimal state) that one wants obtain

returns (Al,Ar,Ac,Bl,Br,Bc,C1,C2,FAl,FAr,FBl,FBr,free_energy,err_mean)
"""
function dl_two_vumps(TA,TB,χ,Al=[],Ar=[],Bl=[],Br=[],Ac=[],Bc=[],C1=[],C2=[],FAl=[],FAr=[],FBl=[],FBr=[];ep=1e-12,e0=1e-1,maxiter=50,elemtype=Complex128,ncv=20)

    @printf("χ=%d, ep=%e, e0=%e \n",χ,ep,e0)
    #initialization
    d,Dh,Dv=size(TA,1,2,4)
    @printf("d=%d, Dh=%d, Dv=%d \n",d,Dh,Dv)

    if Al==[] Al=permutedims(reshape(qr(rand(elemtype,χ*Dv^2,χ))[1],χ,Dv,Dv,χ),[1,4,2,3]) end
    if Ar==[] Ar=permutedims(Al,[2,1,3,4]) end
    if Bl==[] Bl=permutedims(reshape(qr(rand(elemtype,χ*Dv^2,χ))[1],χ,Dv,Dv,χ),[1,4,2,3]) end
    if Br==[] Br=permutedims(Bl,[2,1,3,4]) end
    if Ac==[] Ac=rand(elemtype,χ,χ,Dv,Dv) end
    if Bc==[] Bc=rand(elemtype,χ,χ,Dv,Dv) end
    if C1==[] C1=rand(elemtype,χ,χ) end
    if C2==[] C2=rand(elemtype,χ,χ) end
    if FAl==[] FAl=rand(elemtype,χ,Dh,Dh,χ) end
    if FAr==[] FAr=rand(elemtype,χ,Dh,Dh,χ) end
    if FBl==[] FBl=rand(elemtype,χ,Dh,Dh,χ) end
    if FBr==[] FBr=rand(elemtype,χ,Dh,Dh,χ) end

    free_energy=0.

    err_FAl=err_FAr=err_FBl=err_FBr=err_Ac=err_Bc=err_C1=err_C2=err_Al=err_Ar=err_Bl=err_Br=err_mean=e0

    for iter=1:maxiter
        leftlm=LinearMap([FAl,Al,TA,conj(TA),conj(Al),Bl,TB,conj(TB),conj(Bl)],[[1,2,3,4],[1,10,5,6],[7,2,11,5,8],[7,3,12,6,9],[4,13,8,9],[10,-1,14,15],[16,11,-2,14,17],[16,12,-3,15,18],[13,-4,17,18]],1,elemtype=elemtype)
        leig_res=eigs(leftlm,nev=1,v0=FAl[:],tol=max(ep/100,err_mean/200,1e-15),ncv=ncv)
        λAl,vAl=leig_res
        λAl=λAl[1]
        err_FAl=1-abs(dot(vAl[:],FAl[:]))/(norm(vAl[:])*norm(FAl[:]))
        FAl=reshape(vAl[:],χ,Dh,Dh,χ)
        vBl=jcontract([FAl,Al,TA,conj(TA),conj(Al)],[[1,2,3,4],[1,-1,5,6],[7,2,-2,5,8],[7,3,-3,6,9],[4,-4,8,9]])
        err_FBl=1-abs(dot(vBl[:],FBl[:]))/(norm(vBl[:])*norm(FBl[:]))
        FBl=vBl

        rightlm=LinearMap([FAr,Ar,TA,conj(TA),conj(Ar),Br,TB,conj(TB),conj(Br)],[[1,2,3,4],[10,1,5,6],[7,11,2,5,8],[7,12,3,6,9],[13,4,8,9],[-1,10,14,15],[16,-2,11,14,17],[16,-3,12,15,18],[-4,13,17,18]],1,elemtype=elemtype)
        reig_res=eigs(rightlm,nev=1,v0=FAr[:],tol=max(ep/100,err_mean/200,1e-15),ncv=ncv)
        λAr,vAr=reig_res
        λAr=λAr[1]
        err_FAr=1-abs(dot(vAr[:],FAr[:]))/(norm(vAr[:])*norm(FAr[:]))
        FAr=reshape(vAr[:],χ,Dh,Dh,χ)
        vBr=jcontract([FAr,Ar,TA,conj(TA),conj(Ar)],[[1,2,3,4],[-1,1,5,6],[7,-2,2,5,8],[7,-3,3,6,9],[-4,4,8,9]])
        err_FBr=1-abs(dot(vBr[:],FBr[:]))/(norm(vBr[:])*norm(FBr[:]))
        FBr=vBr
        FAr=FAr/abs(jcontract([FAl,FAr],[[1,2,3,4],[1,2,3,4]]))
        FBr=FBr/abs(jcontract([FBl,FBr],[[1,2,3,4],[1,2,3,4]]))

        Aclm=LinearMap([FAl,Ac,TA,conj(TA),FAr],[[1,2,3,-1],[1,7,4,5],[6,2,8,4,-3],[6,3,9,5,-4],[7,8,9,-2]],2,elemtype=elemtype)
        Aceig_res=eigs(Aclm,nev=1,v0=Ac[:],tol=max(ep/100,err_mean/200,1e-15),ncv=ncv)
        λAc,vAc=Aceig_res
        λAc=λAc[1]
        err_Ac=1-abs(dot(vAc[:],Ac[:]))/(norm(vAc[:])*norm(Ac[:]))
        Ac=reshape(vAc[:],χ,χ,Dv,Dv)

        Bclm=LinearMap([FBl,Bc,TB,conj(TB),FBr],[[1,2,3,-1],[1,7,4,5],[6,2,8,4,-3],[6,3,9,5,-4],[7,8,9,-2]],2,elemtype=elemtype)
        λBc,vBc=eigs(Bclm,nev=1,v0=Bc[:],tol=max(ep/100,err_mean/200,1e-15),ncv=ncv)
        λBc=λBc[1]
        err_Bc=1-abs(dot(vBc[:],Bc[:]))/(norm(vBc[:])*norm(Bc[:]))
        Bc=reshape(vBc[:],χ,χ,Dv,Dv)

        C1lm=LinearMap([FBl,C1,FAr],[[1,2,3,-1],[1,4],[4,2,3,-2]],2,elemtype=elemtype)
        λC1,vC1=eigs(C1lm,nev=1,v0=C1[:],tol=max(ep/100,err_mean/200,1e-15),ncv=ncv)
        λC1=λC1[1]
        err_C1=1-abs(dot(vC1[:],C1[:]))/(norm(vC1[:])*norm(C1[:]))
        C1=reshape(vC1[:],χ,χ)

        C2lm=LinearMap([FAl,C2,FBr],[[1,2,3,-1],[1,4],[4,2,3,-2]],2,elemtype=elemtype)
        λC2,vC2=eigs(C2lm,nev=1,v0=C2[:],tol=max(ep/100,err_mean/200,1e-15),ncv=ncv)
        λC2=λC2[1]
        err_C2=1-abs(dot(vC2[:],C2[:]))/(norm(vC2[:])*norm(C2[:]))
        C2=reshape(vC2[:],χ,χ)

        @printf("eig iter info: \n lniter=%d, lnmult=%d \n rniter=%d, rnmult=%d \n Aciter=%d, Acnmult=%d \n",leig_res[4],leig_res[5],reig_res[4],reig_res[5],Aceig_res[4],Aceig_res[5])
        println("BA singular values:")
        println(svd(C1)[2])
        println("AB singular values:")
        println(svd(C2)[2])

        UAc,PAc=polardecomp(reshape(permutedims(Ac,[1,3,4,2]),χ*Dv^2,χ))
        UC1,PC1=polardecomp(C1)
        Al=permutedims(reshape(UAc*UC1',χ,Dv,Dv,χ),[1,4,2,3])

        UAc,PAc=polardecomp(reshape(permutedims(Ac,[2,3,4,1]),χ*Dv^2,χ))
        UC2,PC2=polardecomp(transpose(C2))
        Ar=permutedims(reshape(UAc*UC2',χ,Dv,Dv,χ),[4,1,2,3])

        UBc,PBc=polardecomp(reshape(permutedims(Bc,[1,3,4,2]),χ*Dv^2,χ))
        UC2,PC2=polardecomp(C2)
        Bl=permutedims(reshape(UBc*UC2',χ,Dv,Dv,χ),[1,4,2,3])

        UBc,PBc=polardecomp(reshape(permutedims(Bc,[2,3,4,1]),χ*Dv^2,χ))
        UC1,PC1=polardecomp(transpose(C1))
        Br=permutedims(reshape(UBc*UC1',χ,Dv,Dv,χ),[4,1,2,3])

        err_Al=vecnorm(Ac-jcontract([Al,C1],[[-1,1,-3,-4],[1,-2]]))
        err_Ar=vecnorm(Ac-jcontract([C2,Ar],[[-1,1],[1,-2,-3,-4]]))
        err_Bl=vecnorm(Bc-jcontract([Bl,C2],[[-1,1,-3,-4],[1,-2]]))
        err_Br=vecnorm(Bc-jcontract([C1,Br],[[-1,1],[1,-2,-3,-4]]))
        err_mean=mean([err_Al,err_Ar,err_Bl,err_Br])

        free_energy=mean([λAl,λAr])

        @printf("iteration %d \n free energy \n λAl: %.16f + i %e \n λAr: %.16f + i %e  \n λAc: %.16f + i %e \n λBc: %.16f + i %e \n λC1: %.16f + i %e \n λC2: %.16f + i %e \n (λAc*λBc)/(λC1*λC2): %.16f + i %e \n error in prediction \n err_FAl: %.16e \n err_FAr: %.16e \n err_FBl: %.16e \n err_FBr: %.16e \n err_Ac: %.16e \n err_Bc: %.16e \n err_C1: %.16e \n err_C2: %.16e \n err_Al: %.16e \n err_Ar: %.16e \n err_Bl: %.16e \n err_Br: %.16e \n \n", iter,real(λAl),imag(λAl),real(λAr),imag(λAr),real(λAc),imag(λAc),real(λBc),imag(λBc),real(λC1),imag(λC1),real(λC2),imag(λC2),real((λAc*λBc)/(λC1*λC2)),imag((λAc*λBc)/(λC1*λC2)),err_FAl,err_FAr,err_FBl,err_FBr,err_Ac,err_Bc,err_C1,err_C2,err_Al,err_Ar,err_Bl,err_Br)
        flush(STDOUT)

        if err_mean<ep break end
    end

    return Al,Ar,Ac,Bl,Br,Bc,C1,C2,FAl,FAr,FBl,FBr,free_energy,err_mean

end
