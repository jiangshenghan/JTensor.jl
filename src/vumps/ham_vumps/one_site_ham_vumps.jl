"""
Obtain ground state using VUMPS with one site per uc.
c.f. Algorithm 2, 6 in 1701.07305

Hamiltonian is assumed to be MPO form Wh, which takes a lower triangular form, e.g, diagonal elements are identity operators up to some constant
      I  0  0
Wh ~  R  λI 0
      S  L  I
where abs(λ)<1

Al and Ar are normalized where largest eigenval to be 1
We assume there is no degenerate in largest eigval for Al, Ar (~injective MPS)

---Al--C--Ar---
   |      |             --Ac--  =  --Al--C--  =  --C--Ar--
---Wh-----Wh---           |          |                |
   |      |
For tensor T, legs order as (left,right,up,down)
legs orders for Al,Ar,Ac are (left,right,down)

returns (Al,Ar,C,Fl,Fr,errp)
Fl,Fr are left and right eigenvectors, with legs orders (up,middle,down)
"""
function one_site_ham_vumps(Wh,chi,Al=[],Ar=[],C=[];err0=1e-10,errp=0.1,maxiter=500,elemtype=Complex128)

    dw,dp=size(Wh,1,3)
    @show dp,dw,chi

    if Al==[] Al=permutedims(reshape(qr(rand(elemtype,chi*dp,chi))[1],chi,dp,chi),[1,3,2]) end
    if Ar==[] Ar=permutedims(Al,[2,1,3]) end
    if C==[] C=rand(elemtype,chi,chi) end
    Ac=jcontract([Al,C],[[-1,1,-3],[1,-2]])
    Fl=rand(elemtype,chi,dw,chi)
    Fr=rand(elemtype,chi,dw,chi)

    iter=1
    El=Er=entropy=0
    Rl=eye(elemtype,chi)
    Lr=eye(elemtype,chi)

    while errp>err0 && iter<maxiter
        @show iter
        errs=errp/100
        errh=errp/100
        @show errs,errh

        #obtain Rl(right fixed pt of Tl) and Lr(left fixed pt of Tr) with norm equal 1
        Tllm=LinearMap([Rl,Al,conj(Al)],[[1,2],[-1,1,3],[-2,2,3]],1,elemtype=elemtype)
        λRl,vRl=eigs(Tllm,nev=1,v0=Rl[:])
        @show λRl
        Rl=reshape(vRl,chi,chi)
        Rl=Rl/trace(Rl)

        Trlm=LinearMap([Lr,Ar,conj(Ar)],[[1,2],[1,-1,3],[2,-2,3]],1,elemtype=elemtype)
        λLr,vLr=eigs(Trlm,nev=1,v0=Lr[:])
        @show λLr
        Lr=reshape(vLr,chi,chi)
        Lr=Lr/trace(Lr)

        #obtain Fl and Fr using iterative method
        Fl[:,dw,:]=eye(elemtype,chi)
        for j=dw-1:-1:1
            Cl=sum(k->jcontract([Fl[:,k,:],Al,Wh[k,j,:,:],conj(Al)],[[1,4],[1,-1,2],[2,3],[4,-2,3]]),j+1:dw)
            if j==1 #for the case where λ=1
                El=jcontract([Cl,Rl],[[1,2],[1,2]]) #El is energy density
                @show El

                Tmlm=MultLinearMap([[Fl[:,1,:]],[Fl[:,1,:],-Al,conj(Al)],[Fl[:,1,:],Rl,eye(elemtype,chi)]],[[[-1,-2]],[[1,3],[1,-1,2],[3,-2,2]],[[1,2],[1,2],[-1,-2]]],[1,1,1],elemtype=elemtype)
                vFl1=Fl[:,1,:][:]
                #IterativeSolvers.gmres!(vFl1,Tmlm,Cl-El*eye(elemtype,chi),tol=errs,verbose=false)
                IterativeSolvers.bicgstabl!(vFl1,Tmlm,Cl-El*eye(elemtype,chi),tol=errs)
                Fl[:,1,:]=reshape(vFl1,chi,chi)

                #Debug mode
                #TWl=eye(elemtype,chi^2)-reshape(jcontract([Al,conj(Al)],[[-1,-3,2],[-2,-4,2]]),chi^2,chi^2)+reshape(jcontract([Rl,eye(elemtype,chi)],[[-1,-2],[-3,-4]]),chi^2,chi^2)
                #copy!(TWl,transpose(TWl))
                #bl=(Cl-El*eye(elemtype,chi))[:]
                #vl_test=Fl[:,1,:][:]
                #IterativeSolvers.gmres!(vl_test,TWl,bl,tol=errs)
                #@show vecnorm(TWl*vl_test-bl)
                #@show vecnorm(vFl1-vl_test)/vecnorm(vFl1)
                #Lresidue=Fl[:,1,:]-jcontract([Fl[:,1,:],Al,conj(Al)],[[1,3],[1,-1,2],[3,-2,2]])+jcontract([Fl[:,1,:],Rl],[[1,2],[1,2]])*eye(elemtype,chi)-Cl+El*eye(elemtype,chi)
                #@show vecnorm(Lresidue)
                #@show jcontract([Fl[:,1,:],Rl],[[1,2],[1,2]]),transpose(vl_test)*Rl[:]
                #@show vecdot(conj(bl),Rl[:])

            elseif vecnorm(Wh[j,j,:,:])<10*eps() #case where λ=0
                Fl[:,j,:]=Cl
            else #case where nonzero λ, with |λ|<1
                λ=vecnorm(Wh[j,j,:,:])/vecnorm(eye(dp))
                println("long range Hamiltonian\n")
                if vecnorm(Wh[j,j,:,:]-λ*eye(dp))>10*eps()
                    @printf "Invalid Hamiltonian MPO" 
                    @show j,Wh[j,j,:,:]
                    exit(1)
                end
                Tmlm=MultLinearMap([[Fl[:,j,:]],[Fl[:,j,:],-λ*Al,conj(Al)]],[[[-1,-2]],[[1,3],[1,-1,2],[3,-2,2]]],[1,1],elemtype=elemtype)
                vFlj=Fl[:,j,:][:]
                IterativeSolvers.gmres!(vFlj,Tmlm,Cl)
                Fl[:,j,:]=reshape(vFlj,chi,chi)
            end
        end

        Fr[:,1,:]=eye(elemtype,chi)
        for j=2:dw
            Cr=sum(k->jcontract([Fr[:,k,:],Ar,Wh[j,k,:,:],conj(Ar)],[[1,2],[-1,1,3],[3,4],[-2,2,4]]),1:j-1)
            if j==dw
                Er=jcontract([Cr,Lr],[[1,2],[1,2]]) #Er is energy density
                @show Er
                Tmlm=MultLinearMap([[Fr[:,dw,:]],[Fr[:,dw,:],-Ar,conj(Ar)],[Fr[:,dw,:],Lr,eye(elemtype,chi)]],[[[-1,-2]],[[1,2],[-1,1,3],[-2,2,3]],[[1,2],[1,2],[-1,-2]]],[1,1,1],elemtype=elemtype)

                vFrdw=Fr[:,dw,:][:]
                #IterativeSolvers.gmres!(vFrdw,Tmlm,Cr-Er*eye(elemtype,chi),tol=errs,verbose=false)
                IterativeSolvers.bicgstabl!(vFrdw,Tmlm,Cr-Er*eye(elemtype,chi),tol=errs)
                Fr[:,dw,:]=reshape(vFrdw,chi,chi)

                #Debug mode
                #TWr=eye(elemtype,chi^2)-reshape(jcontract([Ar,conj(Ar)],[[-1,-3,2],[-2,-4,2]]),chi^2,chi^2)+reshape(jcontract([eye(elemtype,chi),Lr],[[-1,-2],[-3,-4]]),chi^2,chi^2)
                #br=(Cr-Er*eye(elemtype,chi))[:]
                #vr_test=Fr[:,dw,:][:]
                #IterativeSolvers.gmres!(vr_test,TWr,br,tol=errs)
                #@show vecnorm(TWr*vr_test-br)
                #Rresidue=Fr[:,dw,:]-jcontract([Fr[:,dw,:],Ar,conj(Ar)],[[1,2],[-1,1,3],[-2,2,3]])+jcontract([Fr[:,dw,:],Lr],[[1,2],[1,2]])*eye(elemtype,chi)-Cr+Er*eye(elemtype,chi)
                #@show vecnorm(Rresidue)
                #@show vecnorm(vFrdw-vr_test)/vecnorm(vFrdw)
                #@show jcontract([Fr[:,dw,:],Lr],[[1,2],[1,2]]),transpose(vr_test)*Lr[:]
                #@show vecdot(conj(br),Lr[:])

            elseif vecnorm(Wh[j,j,:,:])<10*eps() #case where λ=0
                Fr[:,j,:]=Cr
            else #case where nonzero λ, with |λ|<1
                println("long range Hamiltonian\n")
                λ=vecnorm(Wh[j,j,:,:])/vecnorm(eye(dp))
                Tmlm=MultLinearMap([[Fr[:,j,:]],[Fr[:,j,:],-λ*Ar,conj(Ar)]],[[[-1,-2]],[[1,2],[-1,1,3],[-2,2,3]]],[1,1],elemtype=elemtype)
                vFrj=Fr[:,j,:][:]
                IterativeSolvers.gmres!(vFrj,Tmlm,Cr,verbose=false)
                Fr[:,j,:]=reshape(vFrj,chi,chi)
            end
        end

        #check if Fl and Fr is fixed point
        #=
        Fl_high_order=jcontract([Fl,Al,Wh,conj(Al)],[[1,2,3],[1,-1,4],[2,-2,4,5],[3,-3,5]])-Fl
        for j=1:dw
            Fl_high_order_diag=diag(Fl_high_order[:,j,:])
            Fl_high_order_off_diag=Fl_high_order[:,j,:]-diagm(Fl_high_order_diag)
            if vecnorm(Fl_high_order[:,j,:])>1e-5
                @show j,vecnorm(Fl_high_order_off_diag),vecnorm(Fl_high_order_diag/El-ones(elemtype,chi))
            end
        end
        Fr_high_order=jcontract([Fr,Ar,Wh,conj(Ar)],[[1,2,3],[-1,1,4],[-2,2,4,5],[-3,3,5]])-Fr
        for j=1:dw
            Fr_high_order_diag=diag(Fr_high_order[:,j,:])
            Fr_high_order_off_diag=Fr_high_order[:,j,:]-diagm(Fr_high_order_diag)
            if vecnorm(Fr_high_order[:,j,:])>1e-5
                @show j,vecnorm(Fr_high_order_off_diag),vecnorm(Fr_high_order_diag/Er-ones(elemtype,chi))
            end
        end
        =#

        Aclm=LinearMap([Fl,Ac,Wh,Fr],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]],2,elemtype=elemtype)
        λAc,vAc=eigs(Aclm,nev=1,which=:SR,v0=Ac[:],tol=errh)
        #λAc=λAc[1]
        vAc=vAc[:,1]
        Ac=reshape(vAc[:],chi,chi,dp)

        Clm=LinearMap([Fl,C,Fr],[[1,2,-1],[1,3],[3,2,-2]],2,elemtype=elemtype)
        λC,vC=eigs(Clm,nev=1,which=:SR,v0=C[:],tol=errh)
        #λC=λC[1]
        vC=vC[:,1]
        C=reshape(vC[:],chi,chi)
        @show λAc,λC

        #singular value and entanglement entropy
        svals=svdfact(C)[:S]
        svals=svals/vecnorm(svals)
        entropy=sum(x->-x^2*2log(x),svals)
        @show svals
        @show entropy

        #polar decomposition for Al and Ar
        UAc,PAc=polardecomp(reshape(permutedims(Ac,[1,3,2]),chi*dp,chi))
        UC,PC=polardecomp(C)
        Al=permutedims(reshape(UAc*UC',chi,dp,chi),[1,3,2])

        UAc,PAc=polardecomp(reshape(permutedims(Ac,[2,3,1]),chi*dp,chi))
        UC,PC=polardecomp(transpose(C))
        Ar=permutedims(reshape(UAc*UC',chi,dp,chi),[3,1,2])

        #svd algorithm for Al and Ar
        #AcC_svd=svdfact(reshape(jcontract([Ac,C'],[[-1,1,-2],[1,-3]]),chi*dp,chi))
        #Al=permutedims(reshape(AcC_svd[:U]*AcC_svd[:Vt],chi,dp,chi),[1,3,2])
        #CAc_svd=svdfact(reshape(jcontract([C',Ac],[[-1,1],[1,-2,-3]]),chi,chi*dp))
        #Ar=reshape(CAc_svd[:U]*CAc_svd[:Vt],chi,chi,dp)

        errl=vecnorm(Ac-jcontract([Al,C],[[-1,1,-3],[1,-2]]))
        errr=vecnorm(Ac-jcontract([C,Ar],[[-1,1],[1,-2,-3]]))
        errp=max(errl,errr)
        @show errl,errr,errp

        iter+=1
    end

    @show chi,maxiter,iter,err0,errp,El,Er,entropy

    return Al,Ar,C,Fl,Fr,errp
end
