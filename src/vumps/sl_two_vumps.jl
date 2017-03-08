
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
function sl_two_vumps(TA,TB,chi,Al=[],Ar=[],Bl=[],Br=[],Ac=[],Bc=[],C1=[],C2=[],FAl=[],FAr=[],FBl=[],FBr=[];ep=1e-12,e0=1e-1,maxiter=50,elemtype=Complex128)

    @printf("chi=%d, maxiter=%d, ep=%e, e0=%e \n",chi,maxiter,ep,e0)
    #initialization
    Dh,Dv=size(TA,1,3)
    @printf("Dh=%d, Dv=%d \n",Dh,Dv)

    if Al==[] Al=permutedims(reshape(qr(rand(elemtype,chi*Dv,chi))[1],chi,Dv,chi),[1,3,2]) end
    if Ar==[] Ar=permutedims(Al,[2,1,3]) end
    if Bl==[] Bl=permutedims(reshape(qr(rand(elemtype,chi*Dv,chi))[1],chi,Dv,chi),[1,3,2]) end
    if Br==[] Br=permutedims(Bl,[2,1,3]) end

    if Ac==[] Ac=rand(elemtype,chi,chi,Dv) end
    if Bc==[] Bc=rand(elemtype,chi,chi,Dv) end
    if C1==[] C1=rand(elemtype,chi,chi) end
    if C2==[] C2=rand(elemtype,chi,chi) end
    if FAl==[] FAl=rand(elemtype,chi,Dh,chi) end
    if FAr==[] FAr=rand(elemtype,chi,Dh,chi) end
    if FBl==[] FBl=rand(elemtype,chi,Dh,chi) end
    if FBr==[] FBr=rand(elemtype,chi,Dh,chi) end

    free_energy=0.

    err_FAl=err_FAr=err_FBl=err_FBr=err_Ac=err_Bc=err_C1=err_C2=err_Al=err_Ar=err_Bl=err_Br=err_mean=e0

    for iter=1:maxiter
        leftlm=LinearMap([FAl,Al,TA,conj(Al),Bl,TB,conj(Bl)],[[1,2,3],[1,6,4],[2,7,4,5],[3,8,5],[6,-1,9],[7,-2,9,10],[8,-3,10]],1,elemtype=elemtype)
        λAl,vAl=eigs(leftlm,nev=1,v0=FAl[:],tol=max(ep/100,err_mean/200,1e-15))
        λAl=λAl[1]
        err_FAl=1-abs(dot(vAl[:],FAl[:]))/(norm(vAl[:])*norm(FAl[:]))
        FAl=reshape(vAl[:],chi,Dh,chi)
        vBl=jcontract([FAl,Al,TA,conj(Al)],[[1,2,3],[1,-1,4],[2,-2,4,5],[3,-3,5]])
        err_FBl=1-abs(dot(vBl[:],FBl[:]))/(norm(vBl[:])*norm(FBl[:]))
        FBl=vBl

        rightlm=LinearMap([FAr,Ar,TA,conj(Ar),Br,TB,conj(Br)],[[1,2,3],[6,1,4],[7,2,4,5],[8,3,5],[-1,6,9],[-2,7,9,10],[-3,8,10]],1,elemtype=elemtype)
        λAr,vAr=eigs(rightlm,nev=1,v0=FAr[:],tol=max(ep/100,err_mean/200,1e-15))
        λAr=λAr[1]
        err_FAr=1-abs(dot(vAr[:],FAr[:]))/(norm(vAr[:])*norm(FAr[:]))
        FAr=reshape(vAr[:],chi,Dh,chi)
        vBr=jcontract([FAr,Ar,TA,conj(Ar)],[[1,2,3],[-1,1,4],[-2,2,4,5],[-3,3,5]])
        err_FBr=1-abs(dot(vBr[:],FBr[:]))/(norm(vBr[:])*norm(FBr[:]))
        FBr=vBr
        FAr=FAr/abs(jcontract([FAl,FAr],[[1,2,3],[1,2,3]]))
        FBr=FBr/abs(jcontract([FBl,FBr],[[1,2,3],[1,2,3]]))

        Aclm=LinearMap([FAl,Ac,TA,FAr],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]],2,elemtype=elemtype)
        λAc,vAc=eigs(Aclm,nev=1,v0=Ac[:],tol=max(ep/100,err_mean/200,1e-15))
        λAc=λAc[1]
        err_Ac=1-abs(dot(vAc[:],Ac[:]))/(norm(vAc[:])*norm(Ac[:]))
        Ac=reshape(vAc[:],chi,chi,Dv)

        Bclm=LinearMap([FBl,Bc,TB,FBr],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]],2,elemtype=elemtype)
        λBc,vBc=eigs(Bclm,nev=1,v0=Bc[:],tol=max(ep/100,err_mean/200,1e-15))
        λBc=λBc[1]
        err_Bc=1-abs(dot(vBc[:],Bc[:]))/(norm(vBc[:])*norm(Bc[:]))
        Bc=reshape(vBc[:],chi,chi,Dv)

        C1lm=LinearMap([FBl,C1,FAr],[[1,2,-1],[1,3],[3,2,-2]],2,elemtype=elemtype)
        λC1,vC1=eigs(C1lm,nev=1,v0=C1[:],tol=max(ep/100,err_mean/200,1e-15))
        λC1=λC1[1]
        err_C1=1-abs(dot(vC1[:],C1[:]))/(norm(vC1[:])*norm(C1[:]))
        C1=reshape(vC1[:],chi,chi)

        C2lm=LinearMap([FAl,C2,FBr],[[1,2,-1],[1,3],[3,2,-2]],2,elemtype=elemtype)
        λC2,vC2=eigs(C2lm,nev=1,v0=C2[:],tol=max(ep/100,err_mean/200,1e-15))
        λC2=λC2[1]
        err_C2=1-abs(dot(vC2[:],C2[:]))/(norm(vC2[:])*norm(C2[:]))
        C2=reshape(vC2[:],chi,chi)

        UAc,PAc=polardecomp(reshape(permutedims(Ac,[1,3,2]),chi*Dv,chi))
        UC1,PC1=polardecomp(C1)
        Al=permutedims(reshape(UAc*UC1',chi,Dv,chi),[1,3,2])

        UAc,PAc=polardecomp(reshape(permutedims(Ac,[2,3,1]),chi*Dv,chi))
        UC2,PC2=polardecomp(transpose(C2))
        Ar=permutedims(reshape(UAc*UC2',chi,Dv,chi),[3,1,2])

        UBc,PBc=polardecomp(reshape(permutedims(Bc,[1,3,2]),chi*Dv,chi))
        UC2,PC2=polardecomp(C2)
        Bl=permutedims(reshape(UBc*UC2',chi,Dv,chi),[1,3,2])

        UBc,PBc=polardecomp(reshape(permutedims(Bc,[2,3,1]),chi*Dv,chi))
        UC1,PC1=polardecomp(transpose(C1))
        Br=permutedims(reshape(UBc*UC1',chi,Dv,chi),[3,1,2])

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



