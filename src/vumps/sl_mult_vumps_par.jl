
"""
Obtain fixed point for a single layer multi sites impo

For the following case
---Al[i-1]----Ac[i]----Ar[i+1]----
   |          |        | 
---T[i-1]-----T[i]-----T[i+1]-----
   |          |        | 
For tensors {T}, legs order as (left,right,up,down)
legs orders for {Al},{Ar} are (left,right,down)
ep indicates the precision (how far from the optimal state) that one wants obtain
{Fl},{Fr} are left and right eigenvectors, with legs orders (up,middle,down)
parallel algorithm is implemented

convention:
--Al[i]--C[i]--Ar[i+1]--  = --Ac[i]----Ar[i+1]-- = --Al[i]----Ac[i+1]--
  |            |              |        |             |        | 

returns (Al,Ar,Ac,C,Fl,Fr,free_energy,err)
"""
function sl_mult_vumps_par(T,chi,Al=[],Ar=[],Ac=[],C=[],Fl=[],Fr=[];ep=1e-12,e0=1e-1,maxiter=50,elemtype=Complex128,ncv=20)

    @printf("chi=%d, ep=%e, e0=%e \n",chi,ep,e0)

    #initialization
    N=size(T,1)
    Dh,Dv=size(T[1],1,3)
    @printf("N=%d, Dh=%d, Dv=%d \n",N,Dh,Dv)

    if Al==[] Al=[permutedims(reshape(qr(rand(elemtype,chi*Dv,chi))[1],chi,Dv,chi),[1,3,2]) for i=1:N] end
    if Ar==[] Ar=[permutedims(Al[i],[2,1,3]) for i=1:N] end
    if Fl==[] Fl=[rand(elemtype,chi,Dh,chi) for i=1:N] end
    if Fr==[] Fr=[rand(elemtype,chi,Dh,chi) for i=1:N] end
    if Ac==[] Ac=[rand(elemtype,chi,chi,Dv) for i=1:N] end
    if C==[] C=[rand(elemtype,chi,chi) for i=1:N] end

    free_energy=0.

    errFE=err_Fl=err_Fr=err_Ac=err_C=err_Al=err_Ar=err=e0

    for iter=1:maxiter
        #left fix point
        left_tensor_list=[]
        left_legs_list=[]
        push!(left_tensor_list,Fl[1])
        push!(left_legs_list,[1,2,3])
        for il=1:N
            append!(left_tensor_list,[Al[il],T[il],conj(Al[il])])
            legs_list_il=[[1,6,4],[2,7,4,5],[3,8,5]]
            legs_list_il+=(il-1)*5
            if (il==N)
                legs_list_il[1][2]=-1
                legs_list_il[2][2]=-2
                legs_list_il[3][2]=-3
            end
            append!(left_legs_list,legs_list_il)
        end
        leftlm=LinearMap(left_tensor_list,left_legs_list,1,elemtype=elemtype)
        leig_res=eigs(leftlm,nev=1,v0=Fl[1][:],tol=max(ep/100,err/100,1e-15),ncv=ncv)
        λl,vl=leig_res
        λl=λl[1]
        err_Fl=1-abs(dot(vl[:],Fl[1][:]))/(norm(vl[:])*norm(Fl[1][:]))
        Fl[1]=reshape(vl[:],chi,Dh,chi)
        for il=2:N
            vl=jcontract([Fl[il-1],Al[il],T[il],conj(Al[il])],[[1,2,3],[1,-1,4],[2,-2,4,5],[3,-3,5]])
            err_Fl=max(err_Fl,1-abs(dot(vl[:],Fl[il][:]))/(norm(vl[:])*norm(Fl[il][:])))
            Fl[il]=vl
        end

        #right fixed point
        right_tensor_list=[]
        right_legs_list=[]
        push!(right_tensor_list,Fr[1])
        push!(right_legs_list,[1,2,3])
        for ir=N:-1:1
            append!(right_tensor_list,[Ar[ir],T[ir],conj(Ar[ir])])
            legs_list_ir=[[6,1,4],[7,2,4,5],[8,3,5]]
            legs_list_ir+=(N-ir)*5
            if (ir==1)
                legs_list_ir[1][1]=-1
                legs_list_ir[2][1]=-2
                legs_list_ir[3][1]=-3
            end
            append!(right_legs_list,legs_list_ir)
        end
        rightlm=LinearMap(right_tensor_list,right_legs_list,1,elemtype=elemtype)
        reig_res=eigs(rightlm,nev=1,v0=Fr[N][:],tol=max(ep/100,err/100,1e-15),ncv=ncv)
        λr,vr=reig_res
        λr=λr[1]
        err_Fr=1-abs(dot(vr[:],Fr[N][:]))/(norm(vr[:])*norm(Fr[N][:]))
        Fr[N]=reshape(vr[:],chi,Dh,chi)
        for ir=N-1:-1:1
            vr=jcontract([Fr[ir+1],Ar[ir],T[ir],conj(Ar[ir])],[[1,2,3],[-1,1,4],[-2,2,4,5],[-3,3,5]])
            err_Fr=max(err_Fr,1-abs(dot(vr[:],Fr[ir][:]))/(norm(vr[:])*norm(Fr[ir][:])))
            Fr[ir]=vr
        end

        @printf("iteration %d, eig mult info: \n lniter=%d, lnmult=%d \n rniter=%d, rnmult=%d \n ",iter,leig_res[4],leig_res[5],reig_res[4],reig_res[5])

        err_Ac=err_C=0
        λAcC=1.;
        for ic=1:N
            Aclm=LinearMap([Fl[ic],Ac[ic],T[ic],Fr[ic]],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]],2,elemtype=elemtype)
            Aceig_res=eigs(Aclm,nev=1,v0=Ac[ic][:],tol=max(ep/100,err/100,1e-15),ncv=ncv)
            λAc,vAc=Aceig_res
            λAc=λAc[1]
            err_Ac=max(err_Ac,1-abs(dot(vAc[:],Ac[ic][:]))/(norm(vAc[:])*norm(Ac[ic][:])))
            Ac[ic]=reshape(vAc[:],chi,chi,Dv)

            Clm=LinearMap([Fl[ic==N?1:ic+1],C[ic],Fr[ic]],[[1,2,-1],[1,3],[3,2,-2]],2,elemtype=elemtype)
            Ceig_res=eigs(Clm,nev=1,v0=C[ic][:],tol=max(ep/100,err/100,1e-15),ncv=ncv)
            λC,vC=Ceig_res
            λC=λC[1]
            err_C=max(err_C,1-abs(dot(vC[:],C[ic][:]))/(norm(vC[:])*norm(C[ic][:])))
            C[ic]=reshape(vC[:],chi,chi)

            λAcC*=λAc/λC
            @printf("At site %d\n eig mult info:\n Acninter=%d, Acnmult=%d \n Cniter=%d, Cnmult=%d \n singular values:\n",ic,Aceig_res[4],Aceig_res[5],Ceig_res[4],Ceig_res[5])
            println(svd(C[ic])[2])
        end

        err_Al=err_Ar=0
        for is=1:N
            UAc,PAc=polardecomp(reshape(permutedims(Ac[is],[1,3,2]),chi*Dv,chi))
            UC,PC=polardecomp(C[is])
            Al[is]=permutedims(reshape(UAc*UC',chi,Dv,chi),[1,3,2])

            UAc,PAc=polardecomp(reshape(permutedims(Ac[is],[2,3,1]),chi*Dv,chi))
            UC,PC=polardecomp(transpose(C[is==1?N:is-1]))
            Ar[is]=permutedims(reshape(UAc*UC',chi,Dv,chi),[3,1,2])

            err_Al=max(err_Al,vecnorm(Ac[is]-jcontract([Al[is],C[is]],[[-1,1,-3],[1,-2]])))
            err_Ar=max(err_Ar,vecnorm(Ac[is]-jcontract([C[is==1?N:is-1],Ar[is]],[[-1,1],[1,-2,-3]])))
        end

        free_energy=mean([λr,λl])

        @printf("free energy \n λl: %.16f + i %e \n λr: %.16f + i %e  \n λAcC: %.16f + i %e \n error in prediction \n err_Fl: %.16e \n err_Fr: %.16e \n err_Ac: %.16e \n err_C: %.16e \n err_Al: %.16e \n err_Ar %.16e \n \n",real(λl),imag(λl),real(λr),imag(λr),real(λAcC),imag(λAcC),err_Fl,err_Fr,err_Ac,err_C,err_Al,err_Ar)
        flush(STDOUT)

        err=max(err_Al,err_Ar)
        if err<ep break end

    end

    return Al,Ar,Ac,C,Fl,Fr,free_energy,err

end

