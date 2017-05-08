
"""
Obtain fixed point for a single layer impo with magnetic translation symmetry
Fixed point MPS can be chosen as translational invariant

 --Al--Al--Ac--Ar--
   |   |   |   |
   J  J^2 J^3 J^4
   |   |   |   |
---T---T---T---T---
   |   |   |   |

We also assume internal symmetry J/Jc

    J
    |             |
J---T---J^-1 = ---T---
    |             |
   J^-1

Jc^-1---A---Jc = ---A---
        |           |
        J

For tensor T, legs order as (left,right,up,down)
legs orders for Al,Ar are (left,right,down)
ep indicates the precision (how far from the optimal state) that one wants obtain
Fl,Fr are left and right eigenvectors, with legs orders (up,middle,down)
parallel algorithm is implemented


returns (Al,Ar,Ac,C,Fl,Fr,free_energy,err,[λl,λr,λAc,λC])
"""
function sl_mag_trans_vumps(T,chi,Jc,Al=[],Ar=[],Ac=[],C=[],Fl=[],Fr=[];ep=1e-12,e0=1e-1,maxiter=50,elemtype=Complex128,ncv=20,nev=1,f0=false,λ0=[])

    #initialization
    Dh,Dv=size(T,1,3)

    if Al==[] Al=permutedims(reshape(qr(rand(elemtype,chi*Dv,chi))[1],chi,Dv,chi),[1,3,2]) end
    if Ar==[] Ar=permutedims(reshape(qr(rand(elemtype,chi*Dv,chi))[1],chi,Dv,chi),[1,3,2]) end
    if Fl==[] Fl=rand(elemtype,chi,Dh,chi) end
    if Fr==[] Fr=rand(elemtype,chi,Dh,chi) end
    if Ac==[] Ac=rand(elemtype,chi,chi,Dv) end
    if C==[] C=rand(elemtype,chi,chi) end

    free_energy=0.
    errFE=err_Fl=err_Fr=err_Ac=err_C=err_Al=err_Ar=err=e0

    @show chi,ep,e0,maxiter
    @show Dh,Dv

    for iter=1:maxiter
        #left fix point
        leftlm=LinearMap([Fl,Jc,Al,T,conj(Al)],[[1,2,3],[1,4],[4,-1,5],[2,-2,5,6],[3,-3,6]],1,elemtype=elemtype)
        λl,vl,_,lniter,lnmult=eigs(leftlm,nev=nev,ncv=ncv,v0=Fl[:],tol=max(ep/100,err/200,1e-15))
        @show λl
        @show lniter,lnmult
        #=
        if f0==false
            λl=λl[1]
            vl=vl[:,1]
        else
            ldiff=[1-abs(dot(vl[:,i],Fl[:]))/(norm(vl[:,i])*norm(Fl[:])) for i=1:nev]
            lind=findmin(ldiff)[2]
            λl=λl[lind]
            vl=vl[:,lind]
            @show ldiff
            @show lind
        end
        =#
        if f0==false
            λl=λl[1]
            vl=vl[:,1]
        else
            ldiff=abs((λl-λ0[1])/λ0[1])
            lind=findmin(ldiff)[2]
            λl=λl[lind]
            vl=vl[:,lind]
            @show ldiff
            @show lind
        end
        err_Fl=1-abs(dot(vl[:],Fl[:]))/(norm(vl[:])*norm(Fl[:]))
        Fl=reshape(vl[:],chi,Dh,chi)

        #right fix point
        rightlm=LinearMap([Fr,Jc,Ar,T,conj(Ar)],[[1,2,3],[4,1],[-1,4,5],[-2,2,5,6],[-3,3,6]],1,elemtype=elemtype)
        λr,vr,_,rniter,rnmult=eigs(rightlm,nev=nev,ncv=ncv,v0=Fr[:],tol=max(ep/100,err/200,1e-15))
        @show λr
        @show rniter,rnmult
        #=
        if f0==false
            λr=λr[1]
            vr=vr[:,1]
        else
            rdiff=[1-abs(dot(vr[:,i],Fr[:]))/(norm(vr[:,i])*norm(Fr[:])) for i=1:nev]
            rind=findmin(rdiff)[2]
            λr=λr[rind]
            vr=vr[:,rind]
            @show rdiff
            @show rind
        end
        =#
        if f0==false
            λr=λr[1]
            vr=vr[:,1]
        else
            rdiff=abs((λr-λ0[2])/λ0[2])
            rind=findmin(rdiff)[2]
            λr=λr[rind]
            vr=vr[:,rind]
            @show rdiff
            @show rind
        end
        err_Fr=1-abs(dot(vr[:],Fr[:]))/(norm(vr[:])*norm(Fr[:]))
        Fr=reshape(vr[:],chi,Dh,chi)
        #Fr=Fr/abs(jcontract([Fl,Fr],[[1,2,3],[1,2,3]]))
        @show jcontract([Fl,Fr],[[1,2,3],[1,2,3]])
        Fr=Fr/jcontract([Fl,Fr],[[1,2,3],[1,2,3]])

        #obtain C
        Aclm=LinearMap([Fl,Jc,Ac,T,Jc,Fr],[[1,2,-1],[1,3],[3,5,4],[2,6,4,-3],[5,7],[7,6,-2]],3,elemtype=elemtype)
        λAc,vAc,_,Acniter,Acnmult=eigs(Aclm,nev=nev,ncv=ncv,v0=Ac[:],tol=max(ep/100,err/200,1e-15))
        @show λAc
        @show Acniter,Acnmult
        #=
        if f0==false
            λAc=λAc[1]
            vAc=vAc[:,1]
        else
            Acdiff=[1-abs(dot(vAc[:,i],Ac[:]))/(norm(vAc[:,i])*norm(Ac[:])) for i=1:nev]
            Acind=findmin(Acdiff)[2]
            λAc=λAc[Acind]
            vAc=vAc[:,Acind]
            @show Acdiff
            @show Acind
        end
        =#
        if f0==false
            λAc=λAc[1]
            vAc=vAc[:,1]
        else
            Acdiff=abs((λAc-λ0[3])/λ0[3])
            Acind=findmin(Acdiff)[2]
            λAc=λAc[Acind]
            vAc=vAc[:,Acind]
            @show Acdiff
            @show Acind
        end
        err_Ac=1-abs(dot(vAc[:],Ac[:]))/(norm(vAc[:])*norm(Ac[:]))
        Ac=reshape(vAc[:],chi,chi,Dv)

        #obtain C
        Clm=LinearMap([Fl,C,Jc,Fr],[[1,2,-1],[1,3],[3,4],[4,2,-2]],2,elemtype=elemtype)
        λC,vC,_,Cniter,Cnmult=eigs(Clm,nev=nev,ncv=ncv,v0=C[:],tol=max(ep/100,err/200,1e-15))
        @show λC
        @show Cniter,Cnmult
        #=
        if f0==false
            λC=λC[1]
            vC=vC[:,1]
        else
            Cdiff=[1-abs(dot(vC[:,i],C[:]))/(norm(vC[:,i])*norm(C[:])) for i=1:nev]
            Cind=findmin(Cdiff)[2]
            λC=λC[Cind]
            vC=vC[:,Cind]
            @show Cdiff
            @show Cind
        end
        =#
        if f0==false
            λC=λC[1]
            vC=vC[:,1]
        else
            Cdiff=abs((λC-λ0[4])/λ0[4])
            Cind=findmin(Cdiff)[2]
            λC=λC[Cind]
            vC=vC[:,Cind]
            @show Cdiff
            @show Cind
        end
        err_C=1-abs(dot(vC[:],C[:]))/(norm(vC[:])*norm(C[:]))
        C=reshape(vC[:],chi,chi)



        #singlular values
        svals=svd(C)[2]
        svals/=max(svals...)
        @show svals

        #update Al
        UAc,PAc=polardecomp(reshape(permutedims(Ac,[1,3,2]),chi*Dv,chi))
        UC,PC=polardecomp(C)
        Al=permutedims(reshape(UAc*UC',chi,Dv,chi),[1,3,2])

        #update Ar
        UAc,PAc=polardecomp(reshape(permutedims(Ac,[2,3,1]),chi*Dv,chi))
        UC,PC=polardecomp(transpose(C))
        Ar=permutedims(reshape(UAc*UC',chi,Dv,chi),[3,1,2])

        #errors
        err_Al=vecnorm(Ac-jcontract([Al,C],[[-1,1,-3],[1,-2]]))
        err_Ar=vecnorm(Ac-jcontract([C,Ar],[[-1,1],[1,-2,-3]]))
        err=mean([err_Al,err_Ar])
        errFE=1-abs(mean([λl,λr])*λC/λAc)

        #free_energy=λAc/λC
        free_energy=mean([λl,λr])

        @show iter
        @show λl
        @show λr
        @show λAc/λC
        @show errFE,err_Fl,err_Fr,err_Ac,err_C,err_Al,err_Ar
        @show err
        println()
        flush(STDOUT)

        λ0=[λl,λr,λAc,λC]

        if err<ep break end
    end

    return Al,Ar,Ac,C,Fl,Fr,free_energy,err,λ0

end



function sl_mag_trans_vumps_test(T,chi,Jc,Al=[],Ar=[],Ac=[],C=[],Fl=[],Fr=[];ep=1e-12,e0=1e-1,maxiter=50,elemtype=Complex128,ncv=20,nev=1,f0=false,λ0=[])

    #initialization
    Dh,Dv=size(T,1,3)

    if Al==[] Al=permutedims(reshape(qr(rand(elemtype,chi*Dv,chi))[1],chi,Dv,chi),[1,3,2]) end
    if Ar==[] Ar=permutedims(reshape(qr(rand(elemtype,chi*Dv,chi))[1],chi,Dv,chi),[1,3,2]) end
    if Fl==[] Fl=rand(elemtype,chi,Dh,chi) end
    if Fr==[] Fr=rand(elemtype,chi,Dh,chi) end
    if Ac==[] Ac=rand(elemtype,chi,chi,Dv) end
    if C==[] C=rand(elemtype,chi,chi) end

    free_energy=0.
    errFE=err_Fl=err_Fr=err_Ac=err_C=err_Al=err_Ar=err=e0

    @show chi,ep,e0,maxiter
    @show Dh,Dv

    for iter=1:maxiter
        #left fix point
        leftlm=LinearMap([Fl,Jc,Al,T,conj(Al)],[[1,2,3],[1,4],[4,-1,5],[2,-2,5,6],[3,-3,6]],1,elemtype=elemtype)
        λl,vl,_,lniter,lnmult=eigs(leftlm,nev=nev,ncv=ncv,v0=Fl[:],tol=1e-10)
        @show λl
        @show lniter,lnmult
        #=
        if f0==false
            λl=λl[1]
            vl=vl[:,1]
        else
            ldiff=[1-abs(dot(vl[:,i],Fl[:]))/(norm(vl[:,i])*norm(Fl[:])) for i=1:nev]
            lind=findmin(ldiff)[2]
            λl=λl[lind]
            vl=vl[:,lind]
            @show ldiff
            @show lind
        end
        =#
        if f0==false
            λl=λl[1]
            vl=vl[:,1]
        else
            ldiff=abs((λl-λ0[1])/λ0[1])
            lind=findmin(ldiff)[2]
            λl=λl[lind]
            vl=vl[:,lind]
            @show ldiff
            @show lind
        end
        err_Fl=1-abs(dot(vl[:],Fl[:]))/(norm(vl[:])*norm(Fl[:]))
        Fl=reshape(vl[:],chi,Dh,chi)

        #right fix point
        rightlm=LinearMap([Fr,Jc,Ar,T,conj(Ar)],[[1,2,3],[4,1],[-1,4,5],[-2,2,5,6],[-3,3,6]],1,elemtype=elemtype)
        λr,vr,_,rniter,rnmult=eigs(rightlm,nev=nev,ncv=ncv,v0=Fr[:],tol=1e-10)
        @show λr
        @show rniter,rnmult
        #=
        if f0==false
            λr=λr[1]
            vr=vr[:,1]
        else
            rdiff=[1-abs(dot(vr[:,i],Fr[:]))/(norm(vr[:,i])*norm(Fr[:])) for i=1:nev]
            rind=findmin(rdiff)[2]
            λr=λr[rind]
            vr=vr[:,rind]
            @show rdiff
            @show rind
        end
        =#
        if f0==false
            λr=λr[1]
            vr=vr[:,1]
        else
            rdiff=abs((λr-λ0[2])/λ0[2])
            rind=findmin(rdiff)[2]
            λr=λr[rind]
            vr=vr[:,rind]
            @show rdiff
            @show rind
        end
        err_Fr=1-abs(dot(vr[:],Fr[:]))/(norm(vr[:])*norm(Fr[:]))
        Fr=reshape(vr[:],chi,Dh,chi)
        #Fr=Fr/abs(jcontract([Fl,Fr],[[1,2,3],[1,2,3]]))
        @show jcontract([Fl,Fr],[[1,2,3],[1,2,3]])
        Fr=Fr/jcontract([Fl,Fr],[[1,2,3],[1,2,3]])

        #obtain C
        Aclm=LinearMap([Fl,Jc,Ac,T,Jc,Fr],[[1,2,-1],[1,3],[3,5,4],[2,6,4,-3],[5,7],[7,6,-2]],3,elemtype=elemtype)
        λAc,vAc,_,Acniter,Acnmult=eigs(Aclm,nev=nev,ncv=ncv,v0=Ac[:],tol=1e-10)
        @show λAc
        @show Acniter,Acnmult
        #=
        if f0==false
            λAc=λAc[1]
            vAc=vAc[:,1]
        else
            Acdiff=[1-abs(dot(vAc[:,i],Ac[:]))/(norm(vAc[:,i])*norm(Ac[:])) for i=1:nev]
            Acind=findmin(Acdiff)[2]
            λAc=λAc[Acind]
            vAc=vAc[:,Acind]
            @show Acdiff
            @show Acind
        end
        =#
        if f0==false
            λAc=λAc[1]
            vAc=vAc[:,1]
        else
            Acdiff=abs((λAc-λ0[3])/λ0[3])
            Acind=findmin(Acdiff)[2]
            λAc=λAc[Acind]
            vAc=vAc[:,Acind]
            @show Acdiff
            @show Acind
        end
        err_Ac=1-abs(dot(vAc[:],Ac[:]))/(norm(vAc[:])*norm(Ac[:]))
        Ac=reshape(vAc[:],chi,chi,Dv)

        #obtain C
        Clm=LinearMap([Fl,C,Jc,Fr],[[1,2,-1],[1,3],[3,4],[4,2,-2]],2,elemtype=elemtype)
        λC,vC,_,Cniter,Cnmult=eigs(Clm,nev=nev,ncv=ncv,v0=C[:],tol=1e-10)
        @show λC
        @show Cniter,Cnmult
        #=
        if f0==false
            λC=λC[1]
            vC=vC[:,1]
        else
            Cdiff=[1-abs(dot(vC[:,i],C[:]))/(norm(vC[:,i])*norm(C[:])) for i=1:nev]
            Cind=findmin(Cdiff)[2]
            λC=λC[Cind]
            vC=vC[:,Cind]
            @show Cdiff
            @show Cind
        end
        =#
        if f0==false
            λC=λC[1]
            vC=vC[:,1]
        else
            Cdiff=abs((λC-λ0[4])/λ0[4])
            Cind=findmin(Cdiff)[2]
            λC=λC[Cind]
            vC=vC[:,Cind]
            @show Cdiff
            @show Cind
        end
        err_C=1-abs(dot(vC[:],C[:]))/(norm(vC[:])*norm(C[:]))
        C=reshape(vC[:],chi,chi)



        #singlular values
        svals=svd(C)[2]
        svals/=max(svals...)
        @show svals

        #update Al
        UAc,PAc=polardecomp(reshape(permutedims(Ac,[1,3,2]),chi*Dv,chi))
        UC,PC=polardecomp(C)
        Al=permutedims(reshape(UAc*UC',chi,Dv,chi),[1,3,2])

        #update Ar
        UAc,PAc=polardecomp(reshape(permutedims(Ac,[2,3,1]),chi*Dv,chi))
        UC,PC=polardecomp(transpose(C))
        Ar=permutedims(reshape(UAc*UC',chi,Dv,chi),[3,1,2])

        #errors
        err_Al=vecnorm(Ac-jcontract([Al,C],[[-1,1,-3],[1,-2]]))
        err_Ar=vecnorm(Ac-jcontract([C,Ar],[[-1,1],[1,-2,-3]]))
        err=mean([err_Al,err_Ar])
        errFE=1-abs(mean([λl,λr])*λC/λAc)

        #free_energy=λAc/λC
        free_energy=mean([λl,λr])

        @show iter
        @show λl
        @show λr
        @show λAc/λC
        @show errFE,err_Fl,err_Fr,err_Ac,err_C,err_Al,err_Ar
        @show err
        println()
        flush(STDOUT)

        λ0=[λl,λr,λAc,λC]

        if err<ep break end
    end

    return Al,Ar,Ac,C,Fl,Fr,free_energy,err,λ0

end
