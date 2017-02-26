
# this file stores functions which obtain fixed point for an mpo transfer matrix

"""
Obtain fixed point for an mpo. Here translational invriant is assumed

For the following case
---Al---Ac---Ar---
   |    |    |
---T----T----T----
   |    |    |
For tensor T, legs order as (left,right,up,down)
legs orders for Al,Ar,Ac are (left,right,down)
ep indicates the precision (how far from the optimal state) that one wants obtain
e0 is used to set initial tol to obtain eigenvectors

returns (Al,Ar,Ac,C,Fl,Fr,free_energy,err_mean)
Fl,Fr are left and right eigenvectors, with legs orders (up,middle,down)
"""
function square_mpofp(T,χ,Al=[],Ar=[],Ac=[],C=[],Fl=[],Fr=[];ep=1e-12,e0=1e-1,maxiter=50,elemtype=Complex128)

    @printf("χ=%d, maxiter=%d, ep=%e, e0=%e \n",χ,maxiter,ep,e0)
    #initialization
    Dh,Dv=size(T,1,3)
    @printf("Dh=%d, Dv=%d \n",Dh,Dv)

    if Al==[] Al=permutedims(reshape(qr(rand(elemtype,χ*Dv,χ))[1],χ,Dv,χ),[1,3,2]) end
    if Ar==[] Ar=permutedims(Al,[2,1,3]) end
    if Ac==[] Ac=rand(elemtype,χ,χ,Dv) end
    if C==[] C=rand(elemtype,χ,χ) end
    if Fl==[] Fl=rand(elemtype,χ,Dh,χ) end
    if Fr==[] Fr=rand(elemtype,χ,Dh,χ) end

    free_energy=0.

    errFE=err_Fl=err_Fr=err_Ac=err_C=err_Al=err_Ar=err_mean=e0

    for iter=1:maxiter
        leftlm=LinearMap([Fl,Al,T,conj(Al)],[[1,2,3],[1,-1,4],[2,-2,4,5],[3,-3,5]],1,elemtype=elemtype)
        λl,vl=eigs(leftlm,nev=1,v0=Fl[:],tol=max(ep/100,err_mean/200,1e-15))
        λl=λl[1]
        err_Fl=1-abs(dot(vl[:],Fl[:]))/(norm(vl[:])*norm(Fl[:]))
        Fl=reshape(vl[:],χ,Dh,χ)

        rightlm=LinearMap([Fr,Ar,T,conj(Ar)],[[1,2,3],[-1,1,4],[-2,2,4,5],[-3,3,5]],1,elemtype=elemtype)
        λr,vr=eigs(rightlm,nev=1,v0=Fr[:],tol=max(ep/100,err_mean/200,1e-15))
        λr=λr[1]
        err_Fr=1-abs(dot(vr[:],Fr[:]))/(norm(vr[:])*norm(Fr[:]))
        Fr=reshape(vr[:],χ,Dh,χ)
        Fr=Fr/abs(jcontract([Fl,Fr],[[1,2,3],[1,2,3]]))

        Aclm=LinearMap([Fl,Ac,T,Fr],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]],2,elemtype=elemtype)
        λAc,vAc=eigs(Aclm,nev=1,v0=Ac[:],tol=max(ep/100,err_mean/200,1e-15))
        λAc=λAc[1]
        err_Ac=1-abs(dot(vAc[:],Ac[:]))/(norm(vAc[:])*norm(Ac[:]))
        Ac=reshape(vAc[:],χ,χ,Dv)

        Clm=LinearMap([Fl,C,Fr],[[1,2,-1],[1,3],[3,2,-2]],2,elemtype=elemtype)
        λC,vC=eigs(Clm,nev=1,v0=C[:],tol=max(ep/100,err_mean/200,1e-15))
        λC=λC[1]
        err_C=1-abs(dot(vC[:],C[:]))/(norm(vC[:])*norm(C[:]))
        C=reshape(vC[:],χ,χ)

        UAc,PAc=polardecomp(reshape(permutedims(Ac,[1,3,2]),χ*Dv,χ))
        UC,PC=polardecomp(C)
        Al=permutedims(reshape(UAc*UC',χ,Dv,χ),[1,3,2])

        UAc,PAc=polardecomp(reshape(permutedims(Ac,[2,3,1]),χ*Dv,χ))
        UC,PC=polardecomp(transpose(C))
        Ar=permutedims(reshape(UAc*UC',χ,Dv,χ),[3,1,2])

        err_Al=vecnorm(Ac-jcontract([Al,C],[[-1,1,-3],[1,-2]]))
        err_Ar=vecnorm(Ac-jcontract([C,Ar],[[-1,1],[1,-2,-3]]))
        err_mean=mean([err_Al,err_Ar])
        errFE=1-abs(mean([λl,λr])*λC/λAc)

        free_energy=λAc/λC

        @printf("iteration %d \n free energy \n λl: %.16f + i %e \n λr: %.16f + i %e  \n λAc/λC: %.16f + i %e  \n error in prediction \n errFE: %.16e \n err_Fl: %.16e \n err_Fr: %.16e \n err_Ac: %.16e \n err_C: %.16e \n err_Al: %.16e \n err_Ar %.16e \n \n",iter,real(λl),imag(λl),real(λr),imag(λr),real(λAc/λC),imag(λAc/λC),errFE,err_Fl,err_Fr,err_Ac,err_C,err_Al,err_Ar)
        flush(STDOUT)

        if err_mean<ep break end
    end

    leftlm=LinearMap([Fl,Al,T,conj(Al)],[[1,2,3],[1,-1,4],[2,-2,4,5],[3,-3,5]],1,elemtype=elemtype)
    λl,vl=eigs(leftlm,nev=1,v0=Fl[:],tol=max(ep/100,err_mean/200,1e-15))
    err_Fl=1-abs(dot(vl[:],Fl[:]))/(norm(vl[:])*norm(Fl[:]))
    Fl=reshape(vl[:],χ,Dh,χ)

    rightlm=LinearMap([Fr,Ar,T,conj(Ar)],[[1,2,3],[-1,1,4],[-2,2,4,5],[-3,3,5]],1,elemtype=elemtype)
    λr,vr=eigs(rightlm,nev=1,v0=Fr[:],tol=max(ep/100,err_mean/200,1e-15))
    err_Fr=1-abs(dot(vr[:],Fr[:]))/(norm(vr[:])*norm(Fr[:]))
    Fr=reshape(vr[:],χ,Dh,χ)
    Fr=Fr/abs(jcontract([Fl,C,conj(C),Fr],[[1,2,3],[1,4],[3,5],[4,2,5]])) #Normalization

    Ac=jcontract([Al,C],[[-1,1,-3],[1,-2]])

    return Al,Ar,Ac,C,Fl,Fr,free_energy,err_mean

end

"""
Obtain fixed point for an mpo with 2 site per unit cell.

For the following case
---Al---Bl---Ac---Br---
   |    |    |    |
---T----T----T----T----
   |    |    |    |

For tensor T, legs order as (left,right,up,down)
legs orders for Al,Ar,Ac and Bl,Br,Bc are (left,right,down)
ep indicates the precision (how far from the optimal state) that one wants obtain

We have
LA*A=Al*LB, LB*B=Bl*LA, A*RA=RB*Ar, B*RB=RA*Br
C1=LB*RA, C2=LA*RB
Al*C1=C2*Ar=Ac, Bl*C2=C1*Br=Bc
     Al  Bl         Br  Ar
FAl  TA  TB         TB  TA  FAr
     Al' Bl'        Br' Ar'

     Bl  Al         Ar  Br
FBl  TB  TA         TA  TB  FBr
     Bl' Al'        Ar' Br'

returns (Al,Ar,Ac,Bl,Br,Bc,C1,C2,FAl,FAr,FBl,FBr,free_energy,err_mean)
"""
function square_duc_mpofp(TA,TB,χ,Al=[],Ar=[],Bl=[],Br=[],Ac=[],Bc=[],C1=[],C2=[],FAl=[],FAr=[],FBl=[],FBr=[];ep=1e-12,e0=1e-1,maxiter=50,elemtype=Complex128)

    @printf("χ=%d, maxiter=%d, ep=%e, e0=%e \n",χ,maxiter,ep,e0)
    #initialization
    Dh,Dv=size(TA,1,3)
    @printf("Dh=%d, Dv=%d \n",Dh,Dv)

    if Al==[] Al=permutedims(reshape(qr(rand(elemtype,χ*Dv,χ))[1],χ,Dv,χ),[1,3,2]) end
    if Ar==[] Ar=permutedims(Al,[2,1,3]) end
    if Bl==[] Bl=permutedims(reshape(qr(rand(elemtype,χ*Dv,χ))[1],χ,Dv,χ),[1,3,2]) end
    if Br==[] Br=permutedims(Bl,[2,1,3]) end

    if Ac==[] Ac=rand(elemtype,χ,χ,Dv) end
    if Bc==[] Bc=rand(elemtype,χ,χ,Dv) end
    if C1==[] C1=rand(elemtype,χ,χ) end
    if C2==[] C2=rand(elemtype,χ,χ) end
    if FAl==[] FAl=rand(elemtype,χ,Dh,χ) end
    if FAr==[] FAr=rand(elemtype,χ,Dh,χ) end
    if FBl==[] FBl=rand(elemtype,χ,Dh,χ) end
    if FBr==[] FBr=rand(elemtype,χ,Dh,χ) end

    free_energy=0.

    err_FAl=err_FAr=err_FBl=err_FBr=err_Ac=err_Bc=err_C1=err_C2=err_Al=err_Ar=err_Bl=err_Br=err_mean=e0

    for iter=1:maxiter
        leftlm=LinearMap([FAl,Al,TA,conj(Al),Bl,TB,conj(Bl)],[[1,2,3],[1,6,4],[2,7,4,5],[3,8,5],[6,-1,9],[7,-2,9,10],[8,-3,10]],1,elemtype=elemtype)
        λAl,vAl=eigs(leftlm,nev=1,v0=FAl[:],tol=max(ep/100,err_mean/200,1e-15))
        λAl=λAl[1]
        err_FAl=1-abs(dot(vAl[:],FAl[:]))/(norm(vAl[:])*norm(FAl[:]))
        FAl=reshape(vAl[:],χ,Dh,χ)
        vBl=jcontract([FAl,Al,TA,conj(Al)],[[1,2,3],[1,-1,4],[2,-2,4,5],[3,-3,5]])
        err_FBl=1-abs(dot(vBl[:],FBl[:]))/(norm(vBl[:])*norm(FBl[:]))
        FBl=vBl

        rightlm=LinearMap([FAr,Ar,TA,conj(Ar),Br,TB,conj(Br)],[[1,2,3],[6,1,4],[7,2,4,5],[8,3,5],[-1,6,9],[-2,7,9,10],[-3,8,10]],1,elemtype=elemtype)
        λAr,vAr=eigs(rightlm,nev=1,v0=FAr[:],tol=max(ep/100,err_mean/200,1e-15))
        λAr=λAr[1]
        err_FAr=1-abs(dot(vAr[:],FAr[:]))/(norm(vAr[:])*norm(FAr[:]))
        FAr=reshape(vAr[:],χ,Dh,χ)
        vBr=jcontract([FAr,Ar,TA,conj(Ar)],[[1,2,3],[-1,1,4],[-2,2,4,5],[-3,3,5]])
        err_FBr=1-abs(dot(vBr[:],FBr[:]))/(norm(vBr[:])*norm(FBr[:]))
        FBr=vBr
        FAr=FAr/abs(jcontract([FAl,FAr],[[1,2,3],[1,2,3]]))
        FBr=FBr/abs(jcontract([FBl,FBr],[[1,2,3],[1,2,3]]))

        Aclm=LinearMap([FAl,Ac,TA,FAr],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]],2,elemtype=elemtype)
        λAc,vAc=eigs(Aclm,nev=1,v0=Ac[:],tol=max(ep/100,err_mean/200,1e-15))
        λAc=λAc[1]
        err_Ac=1-abs(dot(vAc[:],Ac[:]))/(norm(vAc[:])*norm(Ac[:]))
        Ac=reshape(vAc[:],χ,χ,Dv)

        Bclm=LinearMap([FBl,Bc,TB,FBr],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]],2,elemtype=elemtype)
        λBc,vBc=eigs(Bclm,nev=1,v0=Bc[:],tol=max(ep/100,err_mean/200,1e-15))
        λBc=λBc[1]
        err_Bc=1-abs(dot(vBc[:],Bc[:]))/(norm(vBc[:])*norm(Bc[:]))
        Bc=reshape(vBc[:],χ,χ,Dv)

        C1lm=LinearMap([FBl,C1,FAr],[[1,2,-1],[1,3],[3,2,-2]],2,elemtype=elemtype)
        λC1,vC1=eigs(C1lm,nev=1,v0=C1[:],tol=max(ep/100,err_mean/200,1e-15))
        λC1=λC1[1]
        err_C1=1-abs(dot(vC1[:],C1[:]))/(norm(vC1[:])*norm(C1[:]))
        C1=reshape(vC1[:],χ,χ)

        C2lm=LinearMap([FAl,C2,FBr],[[1,2,-1],[1,3],[3,2,-2]],2,elemtype=elemtype)
        λC2,vC2=eigs(C2lm,nev=1,v0=C2[:],tol=max(ep/100,err_mean/200,1e-15))
        λC2=λC2[1]
        err_C2=1-abs(dot(vC2[:],C2[:]))/(norm(vC2[:])*norm(C2[:]))
        C2=reshape(vC2[:],χ,χ)

        UAc,PAc=polardecomp(reshape(permutedims(Ac,[1,3,2]),χ*Dv,χ))
        UC1,PC1=polardecomp(C1)
        Al=permutedims(reshape(UAc*UC1',χ,Dv,χ),[1,3,2])

        UAc,PAc=polardecomp(reshape(permutedims(Ac,[2,3,1]),χ*Dv,χ))
        UC2,PC2=polardecomp(transpose(C2))
        Ar=permutedims(reshape(UAc*UC2',χ,Dv,χ),[3,1,2])

        UBc,PBc=polardecomp(reshape(permutedims(Bc,[1,3,2]),χ*Dv,χ))
        UC2,PC2=polardecomp(C2)
        Bl=permutedims(reshape(UBc*UC2',χ,Dv,χ),[1,3,2])

        UBc,PBc=polardecomp(reshape(permutedims(Bc,[2,3,1]),χ*Dv,χ))
        UC1,PC1=polardecomp(transpose(C1))
        Br=permutedims(reshape(UBc*UC1',χ,Dv,χ),[3,1,2])

        err_Al=vecnorm(Ac-jcontract([Al,C1],[[-1,1,-3],[1,-2]]))
        err_Ar=vecnorm(Ac-jcontract([C2,Ar],[[-1,1],[1,-2,-3]]))
        err_Bl=vecnorm(Bc-jcontract([Bl,C2],[[-1,1,-3],[1,-2]]))
        err_Br=vecnorm(Bc-jcontract([C1,Br],[[-1,1],[1,-2,-3]]))
        err_mean=mean([err_Al,err_Ar,err_Bl,err_Br])

        free_energy=mean([λAl,λAr])

        @printf("iteration %d \n free energy \n λAl: %.16f + i %e \n λAr: %.16f + i %e  \n λAc: %.16f + i %e \n λBc: %.16f + i %e \n λC1: %.16f + i %e \n λC2: %.16f + i %e \n (λAc*λBc)/(λC1*λC2): %.16f + i %e \n error in prediction \n err_FAl: %.16e \n err_FAr: %.16e \n err_FBl: %.16e \n err_FBr: %.16e \n err_Ac: %.16e \n err_Bc: %.16e \n err_C1: %.16e \n err_C2: %.16e \n err_Al: %.16e \n err_Ar: %.16e \n err_Bl: %.16e \n err_Br: %.16e \n \n", iter,real(λAl),imag(λAl),real(λAr),imag(λAr),real(λAc),imag(λAc),real(λBc),imag(λBc),real(λC1),imag(λC1),real(λC2),imag(λC2),real((λAc*λBc)/(λC1*λC2)),imag((λAc*λBc)/(λC1*λC2)),err_FAl,err_FAr,err_FBl,err_FBr,err_Ac,err_Bc,err_C1,err_C2,err_Al,err_Ar,err_Bl,err_Br)
        flush(STDOUT)

        if err_mean<ep break end
    end

    return Al,Ar,Ac,Bl,Br,Bc,C1,C2,FAl,FAr,FBl,FBr,free_energy,err_mean

end



"""
Obtain fixed point for a double layer mpo, where the mpo is obtained by contracting one row of single layer square PEPS tensors. Here translational invriant is assumed

For the following case
---Al----Ac----Ar----
   ||    ||    ||
===TT====TT====TT====
   ||    ||    ||
Here, TT lable double layer tensor. For single layer tensor T, legs order as (phys,left,right,up,down)
legs orders for Al,Ar are (left,right,down_ket,down_bra)
ep indicates the precision (how far from the optimal state) that one wants obtain
Fl,Fr are left and right eigenvectors, with legs orders (up,middle_ket,middle_bra,down)

returns (Al,Ar,Ac,C,Fl,Fr,free_energy,err_mean)
"""
function square_dlmpofp(T,χ,Al=[],Ar=[],Ac=[],C=[],Fl=[],Fr=[];ep=1e-12,e0=1e-1,maxiter=50,elemtype=Complex128,ncv=20)

    @printf("χ=%d, ep=%e, e0=%e \n",χ,ep,e0)
    #initialization
    d,Dh,Dv=size(T,1,2,4)
    @printf("d=%d, Dh=%d, Dv=%d \n",d,Dh,Dv)

    Tc=conj(T);
    if Al==[] Al=permutedims(reshape(qr(rand(elemtype,χ*Dv^2,χ))[1],χ,Dv,Dv,χ),[1,4,2,3]) end
    if Ar==[] Ar=permutedims(Al,[2,1,3,4]) end
    if Fl==[] Fl=rand(elemtype,χ,Dh,Dh,χ) end
    if Fr==[] Fr=rand(elemtype,χ,Dh,Dh,χ) end
    if Ac==[] Ac=rand(elemtype,χ,χ,Dv,Dv) end
    if C==[] C=rand(elemtype,χ,χ) end

    free_energy=0.

    errFE=err_Fl=err_Fr=err_Ac=err_C=err_Al=err_Ar=err_mean=e0

    for iter=1:maxiter
        leftlm=LinearMap([Fl,Al,T,Tc,conj(Al)],[[1,2,3,4],[1,-1,5,6],[7,2,-2,5,8],[7,3,-3,6,9],[4,-4,8,9]],1,elemtype=elemtype)
        leig_res=eigs(leftlm,nev=1,v0=Fl[:],tol=max(ep/100,err_mean/200,1e-15),ncv=ncv)
        λl,vl=leig_res
        λl=λl[1]
        err_Fl=1-abs(dot(vl[:],Fl[:]))/(norm(vl[:])*norm(Fl[:]))
        Fl=reshape(vl[:],χ,Dh,Dh,χ)

        rightlm=LinearMap([Fr,Ar,T,Tc,conj(Ar)],[[1,2,3,4],[-1,1,5,6],[7,-2,2,5,8],[7,-3,3,6,9],[-4,4,8,9]],1,elemtype=elemtype)
        reig_res=eigs(rightlm,nev=1,v0=Fr[:],tol=max(ep/100,err_mean/200,1e-15),ncv=ncv)
        λr,vr=reig_res
        λr=λr[1]
        err_Fr=1-abs(dot(vr[:],Fr[:]))/(norm(vr[:])*norm(Fr[:]))
        Fr=reshape(vr[:],χ,Dh,Dh,χ)
        Fr=Fr/abs(jcontract([Fl,Fr],[[1,2,3,4],[1,2,3,4]]))

        Aclm=LinearMap([Fl,Ac,T,Tc,Fr],[[1,2,3,-1],[1,7,4,5],[6,2,8,4,-3],[6,3,9,5,-4],[7,8,9,-2]],2,elemtype=elemtype)
        Aceig_res=eigs(Aclm,nev=1,v0=Ac[:],tol=max(ep/100,err_mean/200,1e-15),ncv=ncv)
        λAc,vAc=Aceig_res
        λAc=λAc[1]
        err_Ac=1-abs(dot(vAc[:],Ac[:]))/(norm(vAc[:])*norm(Ac[:]))
        Ac=reshape(vAc[:],χ,χ,Dv,Dv)

        Clm=LinearMap([Fl,C,Fr],[[1,2,3,-1],[1,4],[4,2,3,-2]],2,elemtype=elemtype)
        Ceig_res=eigs(Clm,nev=1,v0=C[:],tol=max(ep/100,err_mean/200,1e-15),ncv=ncv)
        λC,vC=Ceig_res
        λC=λC[1]
        err_C=1-abs(dot(vC[:],C[:]))/(norm(vC[:])*norm(C[:]))
        C=reshape(vC[:],χ,χ)

        @printf("eig iter info: \n lniter=%d, lnmult=%d \n rniter=%d, rnmult=%d \n Aciter=%d, Acnmult=%d \n Citer=%d, Cnmult=%d \n",leig_res[4],leig_res[5],reig_res[4],reig_res[5],Aceig_res[4],Aceig_res[5],Ceig_res[4],Ceig_res[5])
        println("singlue values:")
        println(svd(C)[2])

        UAc,PAc=polardecomp(reshape(permutedims(Ac,[1,3,4,2]),χ*Dv^2,χ))
        UC,PC=polardecomp(C)
        Al=permutedims(reshape(UAc*UC',χ,Dv,Dv,χ),[1,4,2,3])

        UAc,PAc=polardecomp(reshape(permutedims(Ac,[2,3,4,1]),χ*Dv^2,χ))
        UC,PC=polardecomp(transpose(C))
        Ar=permutedims(reshape(UAc*UC',χ,Dv,Dv,χ),[4,1,2,3])

        err_Al=vecnorm(Ac-jcontract([Al,C],[[-1,1,-3,-4],[1,-2]]))
        err_Ar=vecnorm(Ac-jcontract([C,Ar],[[-1,1],[1,-2,-3,-4]]))
        err_mean=mean([err_Al,err_Ar])
        errFE=1-abs(mean([λl,λr])*λC/λAc)

        free_energy=λAc/λC

        @printf("iteration %d \n free energy \n λl: %.16f + i %e \n λr: %.16f + i %e  \n λAc/λC: %.16f + i %e  \n error in prediction \n errFE: %.16e \n err_Fl: %.16e \n err_Fr: %.16e \n err_Ac: %.16e \n err_C: %.16e \n err_Al: %.16e \n err_Ar %.16e \n \n",iter,real(λl),imag(λl),real(λr),imag(λr),real(λAc/λC),imag(λAc/λC),errFE,err_Fl,err_Fr,err_Ac,err_C,err_Al,err_Ar)
        flush(STDOUT)

        if err_mean<ep break end
    end

    leftlm=LinearMap([Fl,Al,T,Tc,conj(Al)],[[1,2,3,4],[1,-1,5,6],[7,2,-2,5,8],[7,3,-3,6,9],[4,-4,8,9]],1,elemtype=elemtype)
    λl,vl=eigs(leftlm,nev=1,v0=Fl[:],tol=max(ep/100,err_mean/200,1e-15),ncv=ncv)
    err_Fl=1-abs(dot(vl[:],Fl[:]))/(norm(vl[:])*norm(Fl[:]))
    Fl=reshape(vl[:],χ,Dh,Dh,χ)

    rightlm=LinearMap([Fr,Ar,T,Tc,conj(Ar)],[[1,2,3,4],[-1,1,5,6],[7,-2,2,5,8],[7,-3,3,6,9],[-4,4,8,9]],1,elemtype=elemtype)
    λr,vr=eigs(rightlm,nev=1,v0=Fr[:],tol=max(ep/100,err_mean/200,1e-15),ncv=ncv)
    err_Fr=1-abs(dot(vr[:],Fr[:]))/(norm(vr[:])*norm(Fr[:]))
    Fr=reshape(vr[:],χ,Dh,Dh,χ)
    Fr=Fr/abs(jcontract([Fl,C,conj(C),Fr],[[1,2,3,4],[1,5],[4,6],[5,2,3,6]])) #Normalization

    Ac=jcontract([Al,C],[[-1,1,-3,-4],[1,-2]])

    return Al,Ar,Ac,C,Fl,Fr,free_energy,err_mean

end


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
other definition is similiar as in square_duc_mpofp

returns (Al,Ar,Ac,Bl,Br,Bc,C1,C2,FAl,FAr,FBl,FBr,free_energy,err_mean)
"""
function square_duc_dlmpofp(TA,TB,χ,Al=[],Ar=[],Bl=[],Br=[],Ac=[],Bc=[],C1=[],C2=[],FAl=[],FAr=[],FBl=[],FBr=[];ep=1e-12,e0=1e-1,maxiter=50,elemtype=Complex128,ncv=20)

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
