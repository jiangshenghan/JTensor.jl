
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
function sl_one_vumps(T,chi,Al=[],Ar=[],Ac=[],C=[],Fl=[],Fr=[];ep=1e-12,e0=1e-1,maxiter=50,elemtype=Complex128)

    @printf("chi=%d, maxiter=%d, ep=%e, e0=%e \n",chi,maxiter,ep,e0)
    #initialization
    Dh,Dv=size(T,1,3)
    @printf("Dh=%d, Dv=%d \n",Dh,Dv)

    if Al==[] Al=permutedims(reshape(qr(rand(elemtype,chi*Dv,chi))[1],chi,Dv,chi),[1,3,2]) end
    if Ar==[] Ar=permutedims(Al,[2,1,3]) end
    if Ac==[] Ac=rand(elemtype,chi,chi,Dv) end
    if C==[] C=rand(elemtype,chi,chi) end
    if Fl==[] Fl=rand(elemtype,chi,Dh,chi) end
    if Fr==[] Fr=rand(elemtype,chi,Dh,chi) end

    free_energy=0.

    errFE=err_Fl=err_Fr=err_Ac=err_C=err_Al=err_Ar=err_mean=e0

    for iter=1:maxiter
        leftlm=LinearMap([Fl,Al,T,conj(Al)],[[1,2,3],[1,-1,4],[2,-2,4,5],[3,-3,5]],1,elemtype=elemtype)
        λl,vl=eigs(leftlm,nev=1,v0=Fl[:],tol=max(ep/100,err_mean/200,1e-15))
        λl=λl[1]
        err_Fl=1-abs(dot(vl[:],Fl[:]))/(norm(vl[:])*norm(Fl[:]))
        Fl=reshape(vl[:],chi,Dh,chi)

        rightlm=LinearMap([Fr,Ar,T,conj(Ar)],[[1,2,3],[-1,1,4],[-2,2,4,5],[-3,3,5]],1,elemtype=elemtype)
        λr,vr=eigs(rightlm,nev=1,v0=Fr[:],tol=max(ep/100,err_mean/200,1e-15))
        λr=λr[1]
        err_Fr=1-abs(dot(vr[:],Fr[:]))/(norm(vr[:])*norm(Fr[:]))
        Fr=reshape(vr[:],chi,Dh,chi)
        Fr=Fr/abs(jcontract([Fl,Fr],[[1,2,3],[1,2,3]]))

        Aclm=LinearMap([Fl,Ac,T,Fr],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]],2,elemtype=elemtype)
        λAc,vAc=eigs(Aclm,nev=1,v0=Ac[:],tol=max(ep/100,err_mean/200,1e-15))
        λAc=λAc[1]
        err_Ac=1-abs(dot(vAc[:],Ac[:]))/(norm(vAc[:])*norm(Ac[:]))
        Ac=reshape(vAc[:],chi,chi,Dv)

        Clm=LinearMap([Fl,C,Fr],[[1,2,-1],[1,3],[3,2,-2]],2,elemtype=elemtype)
        λC,vC=eigs(Clm,nev=1,v0=C[:],tol=max(ep/100,err_mean/200,1e-15))
        λC=λC[1]
        err_C=1-abs(dot(vC[:],C[:]))/(norm(vC[:])*norm(C[:]))
        C=reshape(vC[:],chi,chi)

        UAc,PAc=polardecomp(reshape(permutedims(Ac,[1,3,2]),chi*Dv,chi))
        UC,PC=polardecomp(C)
        Al=permutedims(reshape(UAc*UC',chi,Dv,chi),[1,3,2])

        UAc,PAc=polardecomp(reshape(permutedims(Ac,[2,3,1]),chi*Dv,chi))
        UC,PC=polardecomp(transpose(C))
        Ar=permutedims(reshape(UAc*UC',chi,Dv,chi),[3,1,2])

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
    Fl=reshape(vl[:],chi,Dh,chi)

    rightlm=LinearMap([Fr,Ar,T,conj(Ar)],[[1,2,3],[-1,1,4],[-2,2,4,5],[-3,3,5]],1,elemtype=elemtype)
    λr,vr=eigs(rightlm,nev=1,v0=Fr[:],tol=max(ep/100,err_mean/200,1e-15))
    err_Fr=1-abs(dot(vr[:],Fr[:]))/(norm(vr[:])*norm(Fr[:]))
    Fr=reshape(vr[:],chi,Dh,chi)
    Fr=Fr/abs(jcontract([Fl,C,conj(C),Fr],[[1,2,3],[1,4],[3,5],[4,2,5]])) #Normalization

    Ac=jcontract([Al,C],[[-1,1,-3],[1,-2]])

    return Al,Ar,Ac,C,Fl,Fr,free_energy,err_mean

end



