
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


returns (Al,Ar,Ac,C,Fl,Fr,free_energy,err)
"""
function sl_mag_trans_vumps(T,chi,Jc,Al=[],Ar=[],Ac=[],C=[],Fl=[],Fr=[];ep=1e-12,e0=1e-1,maxiter=50,elemtype=Complex128,ncv=20,nev=1)

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
        λl,vl,_,lniter,lnmult=eigs(leftlm,nev=nev,v0=Fl[:],tol=max(ep/100,err/200,1e-15))
        @show λl
        @show lniter,lnmult
        λl=λl[1]
        vl=vl[:,1]
        err_Fl=1-abs(dot(vl[:],Fl[:]))/(norm(vl[:])*norm(Fl[:]))
        Fl=reshape(vl[:],chi,Dh,chi)

        #right fix point
        rightlm=LinearMap([Fr,Jc,Ar,T,conj(Ar)],[[1,2,3],[4,1],[-1,4,5],[-2,2,5,6],[-3,3,6]],1,elemtype=elemtype)
        λr,vr,_,rniter,rnmult=eigs(rightlm,nev=nev,v0=Fr[:],tol=max(ep/100,err/200,1e-15))
        @show λr
        @show rniter,rnmult
        λr=λr[1]
        vr=vr[:,1]
        err_Fr=1-abs(dot(vr[:],Fr[:]))/(norm(vr[:])*norm(Fr[:]))
        Fr=reshape(vr[:],chi,Dh,chi)
        Fr=Fr/abs(jcontract([Fl,Fr],[[1,2,3],[1,2,3]]))

        #obtain Ac
        Aclm=LinearMap([Fl,Jc,Ac,T,Jc,Fr],[[1,2,-1],[1,3],[3,5,4],[2,6,4,-3],[5,7],[7,6,-2]],3,elemtype=elemtype)
        λAc,vAc,_,Acniter,Acnmult=eigs(Aclm,nev=nev,v0=Ac[:],tol=max(ep/100,err/200,1e-15))
        @show λAc
        @show Acniter,Acnmult
        λAc=λAc[1]
        vAc=vAc[:,1]
        err_Ac=1-abs(dot(vAc[:],Ac[:]))/(norm(vAc[:])*norm(Ac[:]))
        Ac=reshape(vAc[:],chi,chi,Dv)

        #obtain C
        Clm=LinearMap([Fl,C,Jc,Fr],[[1,2,-1],[1,3],[3,4],[4,2,-2]],2,elemtype=elemtype)
        λC,vC,_,Cniter,Cnmult=eigs(Clm,nev=nev,v0=C[:],tol=max(ep/100,err/200,1e-15))
        @show λC
        @show Cniter,Cnmult
        λC=λC[1]
        vC=vC[:,1]
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

        free_energy=λAc/λC

        @show iter
        @show λl
        @show λr
        @show λAc/λC
        @show errFE,err_Fl,err_Fr,err_Ac,err_C,err_Al,err_Ar
        @show err
        println()
        flush(STDOUT)

        if err<ep break end
    end

    return Al,Ar,Ac,C,Fl,Fr,free_energy,err

end