
"""
Obtain fixed point for a double layer multi sites impo, where the mpo is obtained by contracting one row of single layer square PEPS tensors.

For the following case
---Al[i-1]----Ac[i]----Ar[i+1]----
   ||         ||       ||
===TT[i-1]====TT[i]====TT[i+1]====
   ||         ||       ||
Here, {TT} lable double layer tensor. For single layer tensors {T}, legs order as (phys,left,right,up,down)
legs orders for {Al},{Ar} are (left,right,down_ket,down_bra)
ep indicates the precision (how far from the optimal state) that one wants obtain
{Fl},{Fr} are left and right eigenvectors, with legs orders (up,middle_ket,middle_bra,down)
sequential algorithm is implemented

returns (Al,Ar,Ac,C,Fl,Fr,free_energy,err)
"""
function dl_mult_vumps_seq(T,chi,Al=[],Ar=[],Ac=[],C=[],Fl=[],Fr=[];ep=1e-12,e0=1e-1,maxiter=50,elemtype=Complex128)

    @printf("chi=%d, ep=%e, e0=%e \n",chi,ep,e0)

    #initialization
    N=size(T,1)
    d,Dh,Dv=size(T[1],1,2,4)
    @printf("N=%d, d=%d, Dh=%d, Dv=%d \n",N,d,Dh,Dv)

    Tc=conj(T)
    if Al==[] Al=[permutedims(reshape(qr(rand(elemtype,chi*Dv^2,chi))[1],chi,Dv,Dv,chi),[1,4,2,3]) for i=1:N] end
    if Ar==[] Ar=[permutedims(Al[i],[2,1,3,4]) for i=1:N] end
    if Fl==[] Fl=[rand(elemtype,chi,Dh,Dh,chi) for i=1:N] end
    if Fr==[] Fr=[rand(elemtype,chi,Dh,Dh,chi) for i=1:N] end
    if Ac==[] Ac=[rand(elemtype,chi,chi,Dv,Dv) for i=1:N] end
    if C==[] C=[rand(elemtype,chi,chi) for i=1:N] end

    free_energy=0.

    errFE=err_Fl=err_Fr=err_Ac=err_CL=err_CR=err_Al=err_Ar=err=e0

    for iter=1:maxiter
        err_max=0
        for ns=1:N

            #left fixed point
            left_tensor_list=[]
            push!(left_tensor_list,Fl[ns])
            for il=ns:(ns+N-1)
                jl=(il-1)%N+1
                append!(left_tensor_list,[Al[jl],T[jl],Tc[jl],conj(Al[jl])])
            end
            left_legs_list=[]
            push!(left_legs_list,[1,2,3,4])
            for il=0:N-1
                legs_list_il=[[1,10,5,6],[7,2,11,5,8],[7,3,12,6,9],[4,13,8,9]]
                legs_list_il+=il*9
                if il==N-1
                    legs_list_il[1][2]=-1
                    legs_list_il[2][3]=-2
                    legs_list_il[3][3]=-3
                    legs_list_il[4][2]=-4
                end
                append!(left_legs_list,legs_list_il)
            end
            leftlm=LinearMap(left_tensor_list,left_legs_list,1,elemtype=elemtype)
            leig_res=eigs(leftlm,nev=1,v0=Fl[ns][:],tol=max(ep/100,err/100,1e-15))
            λl,vl=leig_res
            λl=λl[1]
            err_Fl=1-abs(dot(vl[:],Fl[ns][:]))/(norm(vl[:])*norm(Fl[ns][:]))
            Fl[ns]=reshape(vl[:],chi,Dh,Dh,chi)

            #right fixed point
            right_tensor_list=[]
            push!(right_tensor_list,Fr[ns])
            for ir=ns:-1:(ns-N+1)
                jr=(ir+N-1)%N+1
                append!(right_tensor_list,[Ar[jr],T[jr],Tc[jr],conj(Ar)[jr]])
            end
            right_legs_list=[]
            push!(right_legs_list,[1,2,3,4])
            for ir=0:N-1
                legs_list_ir=[[10,1,5,6],[7,11,2,5,8],[7,12,3,6,9],[13,4,8,9]]
                legs_list_ir+=ir*9
                if ir==N-1
                    legs_list_ir[1][1]=-1
                    legs_list_ir[2][2]=-2
                    legs_list_ir[3][2]=-3
                    legs_list_ir[4][1]=-4
                end
                append!(right_legs_list,legs_list_ir)
            end
            rightlm=LinearMap(right_tensor_list,right_legs_list,1,elemtype=elemtype)
            reig_res=eigs(rightlm,nev=1,v0=Fr[ns][:],tol=max(ep/100,err/100,1e-15))
            λr,vr=reig_res
            λr=λr[1]
            err_Fr=1-abs(dot(vr[:],Fr[ns][:]))/(norm(vr[:])*norm(Fr[ns][:]))
            Fr[ns]=reshape(vr[:],chi,Dh,Dh,chi)
            #Fr[ns]=Fr[ns]/abs(jcontract([Fl[ns],Fr[ns]],[[1,2,3,4],[1,2,3,4]]))

            Aclm=LinearMap([Fl[ns],Ac[ns],T[ns],Tc[ns],Fr[ns]],[[1,2,3,-1],[1,7,4,5],[6,2,8,4,-3],[6,3,9,5,-4],[7,8,9,-2]],2,elemtype=elemtype)
            Aceig_res=eigs(Aclm,nev=1,v0=Ac[ns][:],tol=max(ep/100,err/100,1e-15))
            λAc,vAc=Aceig_res
            λAc=λAc[1]
            err_Ac=1-abs(dot(vAc[:],Ac[ns][:]))/(norm(vAc[:])*norm(Ac[ns][:]))
            Ac[ns]=reshape(vAc[:],chi,chi,Dv,Dv)

            nsl=(ns+N-2)%N+1
            CLlm=LinearMap([Fl[ns],C[nsl],Fr[nsl]],[[1,2,3,-1],[1,4],[4,2,3,-2]],2,elemtype=elemtype)
            CLeig_res=eigs(CLlm,nev=1,v0=C[nsl][:],tol=max(ep/100,err/100,1e-15))
            λCL,vCL=CLeig_res
            λCL=λCL[1]
            err_CL=1-abs(dot(vCL[:],C[nsl][:]))/(norm(vCL[:])*norm(C[nsl][:]))
            C[nsl]=reshape(vCL[:],chi,chi)

            CRlm=LinearMap([Fl[ns%N+1],C[ns],Fr[ns]],[[1,2,3,-1],[1,4],[4,2,3,-2]],2,elemtype=elemtype)
            CReig_res=eigs(CRlm,nev=1,v0=C[ns][:],tol=max(ep/100,err/100,1e-15))
            λCR,vCR=CReig_res
            λCR=λCR[1]
            err_CR=1-abs(dot(vCR[:],C[ns][:]))/(norm(vCR[:])*norm(C[ns][:]))
            C[ns]=reshape(vCR[:],chi,chi)

            @printf("iteration %d, site %d \n eig iter info: \n lniter=%d, lnmult=%d \n rniter=%d, rnmult=%d \n Aciter=%d, Acnmult=%d \n",iter,ns,leig_res[4],leig_res[5],reig_res[4],reig_res[5],Aceig_res[4],Aceig_res[5])
            println("CL singlue values:")
            println(svd(C[nsl])[2])
            println("CR singlue values:")
            println(svd(C[ns])[2])

            UAc,PAc=polardecomp(reshape(permutedims(Ac[ns],[1,3,4,2]),chi*Dv^2,chi))
            UC,PC=polardecomp(C[ns])
            Al[ns]=permutedims(reshape(UAc*UC',chi,Dv,Dv,chi),[1,4,2,3])

            UAc,PAc=polardecomp(reshape(permutedims(Ac[ns],[2,3,4,1]),chi*Dv^2,chi))
            UC,PC=polardecomp(transpose(C[nsl]))
            Ar[ns]=permutedims(reshape(UAc*UC',chi,Dv,Dv,chi),[4,1,2,3])

            err_Al=vecnorm(Ac[ns]-jcontract([Al[ns],C[ns]],[[-1,1,-3,-4],[1,-2]]))
            err_Ar=vecnorm(Ac[ns]-jcontract([C[nsl],Ar[ns]],[[-1,1],[1,-2,-3,-4]]))
            err_max=max(err_max,err_Al,err_Ar)

            free_energy=mean([λr,λl])

            @printf("free energy \n λl: %.16f + i %e \n λr: %.16f + i %e  \n error in prediction \n err_Fl: %.16e \n err_Fr: %.16e \n err_Ac: %.16e \n err_CL: %.16e \n err_CR: %.16e \n err_Al: %.16e \n err_Ar %.16e \n err_max %.16e \n \n",real(λl),imag(λl),real(λr),imag(λr),err_Fl,err_Fr,err_Ac,err_CL,err_CR,err_Al,err_Ar,err_max)
            flush(STDOUT)
        end
        err=err_max

        if err<ep break end
    end

    return Al,Ar,Ac,C,Fl,Fr,free_energy,err
end
