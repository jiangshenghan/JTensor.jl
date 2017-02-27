
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
function dl_one_vumps(T,χ,Al=[],Ar=[],Ac=[],C=[],Fl=[],Fr=[];ep=1e-12,e0=1e-1,maxiter=50,elemtype=Complex128,ncv=20)

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

