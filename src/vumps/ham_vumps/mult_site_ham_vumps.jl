"""
Obtain ground state using VUMPS with multiple sites per uc in a parallel manner.
c.f. Algorithm 2, 3, 6 in 1701.07305

Hamiltonian is assumed to be MPO form Wh, which takes a lower triangular form, e.g, diagonal elements are identity operators up to some constant
      I  0  0
Wh ~  R  λI 0
      S  L  I
where abs(λ)<1

For the following case
---Al[i-1]----Ac[i]----Ar[i+1]----
   |          |        | 
--Wh[i-1]----Wh[i]----Wh[i+1]-----
   |          |        | 

For tensor {Wh}, legs order as (left,right,up,down)
legs orders for {Al},{Ar},{Ac} are (left,right,down)

Al and Ar are normalized where largest eigenval to be 1
We assume there is no degenerate in largest eigval for Al, Ar (~injective MPS)

convention:
--Al[i]--C[i]--Ar[i+1]--  = --Ac[i]----Ar[i+1]-- = --Al[i]----Ac[i+1]--
  |            |              |        |             |        | 

returns (Al,Ar,C,Fl,Fr,errp)
Fl,Fr are left and right eigenvectors, with legs orders (up,middle,down)
"""
function mult_site_ham_vumps_par(Wh,chi,Al=[],Ar=[],C=[];err0=1e-10,errp=0.1,maxiter=500,elemtype=Complex128)
    Ns=size(Wh,1)
    dw,dp=size(Wh[1],1,3)
    @show Ns,dp,dw,chi

    if Al==[] Al=[permutedims(reshape(qr(rand(elemtype,chi*dp,chi))[1],chi,dp,chi),[1,3,2]) for i=1:Ns] end
    if Ar==[] Ar=[permutedims(Al[i],[2,1,3]) for i=1:Ns] end
    if C==[] C=[rand(elemtype,chi,chi) for i=1:Ns] end
    Ac=[jcontract([Al[i],C[i]],[[-1,1,-3],[1,-2]]) for i=1:Ns]
    Fl=[rand(elemtype,chi,dw,chi) for i=1:Ns]
    Fr=[rand(elemtype,chi,dw,chi) for i=1:Ns] 

    iter=1
    El=Er=0
    entropy=zeros(Ns)
    Rl=[rand(elemtype,chi,chi) for i=1:Ns]
    Lr=[rand(elemtype,chi,chi) for i=1:Ns]

    while errp>err0 && iter<maxiter
        @show iter
        errs=errp/100
        errh=errp/100
        @show errs,errh

        #TODO:check Lr and Rl
        #right eigenvector of Tl1...TlN
        Tl_from_right=[]
        push!(Tl_from_right,Rl[Ns])
        legs_from_right=[]
        push!(legs_from_right,[1,2])
        for j=Ns:-1:1
            push!(Tl_from_right,Al[j],conj(Al[j]))
            legsj=[[4,1,3],[5,2,3]]
            legsj+=(Ns-j)*3
            if j==1 
                legsj[1][1]=-1
                legsj[2][1]=-2
            end
            append!(legs_from_right,legsj)
        end
        Tllm=LinearMap(Tl_from_right,legs_from_right,1,elemtype=elemtype) 
        λRl,vRl=eigs(Tllm,nev=1,v0=Rl[Ns][:])
        Rl[Ns]=reshape(vRl,chi,chi)
        Rl[Ns]=Rl[Ns]/trace(Rl[Ns])
        @show λRl

        #obtain eigenvector of Tl_j...Tl_N+j-1
        for j=Ns-1:-1:1
            Rl[j]=jcontract([Rl[j+1],Al[j+1],conj(Al[j+1])],[[1,2],[-1,1,3],[-2,2,3]])
            Rl[j]=Rl[j]/trace(Rl[j])
        end

        #left eigenvector of Tr1...TrN
        Tr_from_left=[]
        push!(Tr_from_left,Lr[1])
        legs_from_left=[]
        push!(legs_from_left,[1,2])
        for j=1:Ns
            push!(Tr_from_left,Ar[j],conj(Ar[j]))
            legsj=[[1,4,3],[2,5,3]]
            legsj+=(j-1)*3
            if j==Ns
                legsj[1][2]=-1
                legsj[2][2]=-2
            end
            append!(legs_from_left,legsj)
        end
        Trlm=LinearMap(Tr_from_left,legs_from_left,1,elemtype=elemtype)
        λLr,vLr=eigs(Trlm,nev=1,v0=Lr[1][:])
        Lr[1]=reshape(vLr,chi,chi)
        Lr[1]=Lr[1]/trace(Lr[1])
        @show λLr

        #obtain eigenvector of Tr_j...Tr_N+j-1
        for j=2:Ns
            Lr[j]=jcontract([Lr[j-1],Ar[j-1],conj(Ar[j-1])],[[1,2],[1,-1,3],[2,-2,3]])
            Lr[j]=Lr[j]/trace(Lr[j])
        end

        #obtain Fl and Fr
        ##=
        El=obtain_Fl_il!(Fl,1,Al,Wh,Rl,errs=errs,elemtype=elemtype)
        for il=2:Ns
            Fl[il]=jcontract([Fl[il-1],Al[il-1],Wh[il-1],conj(Al[il-1])],[[1,2,3],[1,-1,4],[2,-2,4,5],[3,-3,5]])
            Eil=jcontract([Fl[il][:,1,:],Rl[il-1]],[[1,2],[1,2]])
            Fl[il][:,1,:]=Fl[il][:,1,:]-Eil*eye(elemtype,chi,chi)
            @show il,Eil
        end #for Fl

        Er=obtain_Fr_ir!(Fr,Ns,Ar,Wh,Lr;errs=errs,elemtype=elemtype)
        for ir=Ns-1:-1:1
            Fr[ir]=jcontract([Fr[ir+1],Ar[ir+1],Wh[ir+1],conj(Ar[ir+1])],[[1,2,3],[-1,1,4],[-2,2,4,5],[-3,3,5]])
            Eir=jcontract([Fr[ir][:,dw,:],Lr[ir+1]],[[1,2],[1,2]])
            Fr[ir][:,dw,:]=Fr[ir][:,dw,:]-Eir*eye(elemtype,chi,chi)
            @show ir,Eir
        end

        # =#
        #for i=1:Ns
        #    El=obtain_Fl_il!(Fl,i,Al,Wh,Rl,errs=errs,elemtype=elemtype)
        #    Er=obtain_Fr_ir!(Fr,i,Ar,Wh,Lr,errs=errs,elemtype=elemtype)
        #end

        #obtain Ac and C
        for ic=1:Ns
            Aclm=LinearMap([Fl[ic],Ac[ic],Wh[ic],Fr[ic]],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]],2,elemtype=elemtype)
            λAc,vAc=eigs(Aclm,nev=1,which=:SR,v0=Ac[ic][:],tol=errh)
            Ac[ic]=reshape(vAc[:],chi,chi,dp)

            Clm=LinearMap([Fl[ic==Ns?1:ic+1],C[ic],Fr[ic]],[[1,2,-1],[1,3],[3,2,-2]],2,elemtype=elemtype)
            λC,vC=eigs(Clm,nev=1,which=:SR,v0=C[ic][:],tol=errh)
            C[ic]=reshape(vC[:],chi,chi)
            @show ic,λAc,λC

            #singular value and EE
            svals=svdfact(C[ic])[:S]
            svals=svals/vecnorm(svals)
            entropy[ic]=sum(x->-x^2*2log(x),svals)
            @show ic,svals
        end #for Ac and C
        @show entropy

        #obtain Al and Ar
        errl=errr=0
        for is=1:Ns
            UAc,PAc=polardecomp(reshape(permutedims(Ac[is],[1,3,2]),chi*dp,chi))
            UC,PC=polardecomp(C[is])
            Al[is]=permutedims(reshape(UAc*UC',chi,dp,chi),[1,3,2])

            UAc,PAc=polardecomp(reshape(permutedims(Ac[is],[2,3,1]),chi*dp,chi))
            UC,PC=polardecomp(transpose(C[is==1?Ns:is-1]))
            Ar[is]=permutedims(reshape(UAc*UC',chi,dp,chi),[3,1,2])

            errl=max(errl,vecnorm(Ac[is]-jcontract([Al[is],C[is]],[[-1,1,-3],[1,-2]])))
            errr=max(errr,vecnorm(Ac[is]-jcontract([C[is==1?Ns:is-1],Ar[is]],[[-1,1],[1,-2,-3]])))
        end #for Al and Ar
        errp=max(errl,errr)
        @show errp

        iter+=1
    end #while

    @show chi,iter,err0,errp,El,Er,entropy

    return Al,Ar,C,Fl,Fr,errp
end #function

function obtain_Fl_il!(Fl,il,Al,Wh,Rl;errs=1e-10,elemtype=Complex128)
    Ns=length(Fl)
    chi,dw=size(Fl[1],1,2)
    El=0

    Fl[il][:,dw,:]=eye(elemtype,chi)
    for j=dw-1:-1:1
        #creat all possible int arrays with length Ns+1 and decreasing order and last element is j
        inds_all=[]
        for j_cod=1:(dw-j+1)^Ns
            ind_cod=j_cod
            inds=Int[]
            b_ind=true
            for m=1:Ns
                push!(inds,mod(ind_cod,dw-j+1)+j)
                ind_cod=div(ind_cod,dw-j+1)
                if (m>1 && inds[end-1]<inds[end]) || (m==1 && inds[1]==j)
                    b_ind=false 
                    break
                end
            end
            if b_ind==false continue end
            push!(inds,j)
            push!(inds_all,inds)
        end #for j_cod

        #obtain Cl
        Cl=zeros(elemtype,chi,chi)
        Cl_legs_list=[]
        push!(Cl_legs_list,[1,2])
        for js=1:Ns
            push!(Cl_legs_list,[4*js-3,4*js+1,4*js-1],[4*js-1,4*js],[4*js-2,4*js+2,4*js])
        end
        Cl_legs_list[end-2][2]=-1
        Cl_legs_list[end][2]=-2
        for inds in inds_all
            Cl_tensor_list=[]
            push!(Cl_tensor_list,Fl[il][:,inds[1],:])
            for (ms,js) in enumerate(circshift(1:Ns,1-il))
                push!(Cl_tensor_list,Al[js],Wh[js][inds[ms],inds[ms+1],:,:],conj(Al[js]))
            end
            Cl+=jcontract(Cl_tensor_list,Cl_legs_list)
        end #for Cl

        #get Fl[il][:,j,:]
        if  j==1
            El=jcontract([Cl,Rl[il==1?Ns:il-1]],[[1,2],[1,2]])/Ns #energy density
            @show il,El

            #obtain Tl_from_left and legs_from_left
            Tl_from_left=[]
            push!(Tl_from_left,Fl[il][:,1,:])
            for js=circshift(1:Ns,1-il) push!(Tl_from_left,Al[js],conj(Al[js])) end
            Tl_from_left[2]=-Tl_from_left[2]
            legs_from_left=[]
            push!(legs_from_left,[1,2])
            for js=1:Ns
                legsjs=[[1,4,3],[2,5,3]]
                legsjs+=(js-1)*3
                if js==Ns
                    legsjs[1][2]=-1
                    legsjs[2][2]=-2
                end
                append!(legs_from_left,legsjs)
            end

            Tmlm=MultLinearMap([[Fl[il][:,1,:]],Tl_from_left,[Fl[il][:,1,:],Rl[il==1?Ns:il-1],eye(elemtype,chi)]],[[[-1,-2]],legs_from_left,[[1,2],[1,2],[-1,-2]]],[1,1,1],elemtype=elemtype)
            vFl1=Fl[il][:,1,:][:]
            IterativeSolvers.bicgstabl!(vFl1,Tmlm,Cl-Ns*El*eye(elemtype,chi),tol=errs)
            Fl[il][:,1,:]=reshape(vFl1,chi,chi)
        elseif prod(k->vecnorm(Wh[k][j,j,:,:]),1:Ns)<10*eps()
            Fl[il][:,j,:]=Cl
        else
            throw("long range Hamiltonian not implemented")
            #TODO:long range Hamiltonian
        end
    end #for Fl[il]

    return El
end


function obtain_Fr_ir!(Fr,ir,Ar,Wh,Lr;errs=1e-10,elemtype=Complex128)
    Ns=length(Fr)
    chi,dw=size(Fr[1],1,2)
    Er=0

    Fr[ir][:,1,:]=eye(elemtype,chi)
    for j=2:dw
        #creat all possible int arrays with length Ns+1 and increasing order and last element is j
        inds_all=[]
        for j_cod=1:j^Ns
            ind_cod=j_cod
            inds=Int[]
            b_ind=true
            for m=Ns:-1:1
                push!(inds,mod(ind_cod,j)+1)
                ind_cod=div(ind_cod,j)
                if (m<Ns && inds[end-1]>inds[end]) || (m==Ns && inds[1]==j)
                    b_ind=false
                    break
                end
            end
            if b_ind==false continue end
            push!(inds,j)
            push!(inds_all,inds)
        end

        #obtain Cr
        Cr=zeros(elemtype,chi,chi)
        Cr_legs_list=[]
        push!(Cr_legs_list,[1,2])
        for js=1:Ns
            push!(Cr_legs_list,[4*js+1,4*js-3,4*js-1],[4*js-1,4*js],[4*js+2,4*js-2,4*js])
        end
        Cr_legs_list[end-2][1]=-1
        Cr_legs_list[end][1]=-2
        for inds in inds_all
            Cr_tensor_list=[]
            push!(Cr_tensor_list,Fr[ir][:,inds[1],:])
            for (ms,js)=enumerate(circshift(Ns:-1:1,ir-Ns))
                push!(Cr_tensor_list,Ar[js],Wh[js][inds[ms+1],inds[ms],:,:],conj(Ar[js]))
            end
            Cr+=jcontract(Cr_tensor_list,Cr_legs_list)
        end #for Cr

        #get Fr[ir][:,j,:]
        if j==dw
            Er=jcontract([Cr,Lr[ir==Ns?1:ir+1]],[[1,2],[1,2]])/Ns #energy density
            @show ir,Er

            #obtain Tr_from_right and legs_from_right
            Tr_from_right=[]
            push!(Tr_from_right,Fr[ir][:,dw,:])
            for js=circshift(Ns:-1:1,ir-Ns) push!(Tr_from_right,Ar[js],conj(Ar[js])) end
            Tr_from_right[2]=-Tr_from_right[2]
            legs_from_right=[]
            push!(legs_from_right,[1,2])
            for js=Ns:-1:1
                legsjs=[[4,1,3],[5,2,3]]
                legsjs+=(Ns-js)*3
                if js==1 
                    legsjs[1][1]=-1
                    legsjs[2][1]=-2
                end
                append!(legs_from_right,legsjs)
            end

            Tmlm=MultLinearMap([[Fr[ir][:,dw,:]],Tr_from_right,[Fr[ir][:,dw,:],Lr[ir==Ns?1:ir+1],eye(elemtype,chi)]],[[[-1,-2]],legs_from_right,[[1,2],[1,2],[-1,-2]]],[1,1,1],elemtype=elemtype)
            vFrdw=Fr[ir][:,dw,:][:]
            IterativeSolvers.bicgstabl!(vFrdw,Tmlm,Cr-Ns*Er*eye(elemtype,chi),tol=errs)
            Fr[ir][:,dw,:]=reshape(vFrdw,chi,chi)

        elseif prod(k->vecnorm(Wh[k][j,j,:,:]),1:Ns)<10*eps()
            Fr[ir][:,j,:]=Cr
        else
            throw("long range Hamiltonian not implemented")
            #TODO:long range Hamiltonian
        end 
    end #for Fr[ir]

    return Er
end
