
"""
Increase cutoff bond dimension for square pi flux
Here, TT,Al,Ar all contains two tensors
   
   ----A2c----
  /   /   \    \         ---A2c---
Fl--TT[1]--TT[2]--Fr  ~     | |
  \  |     |   /

legs_order for A2c: left,ld,right,rd
legs_order for Al,Ar: l,r,d
legs_order for TT: l,r,u,d
T_spin are for spin rep for a virtual leg of single layer PEPS tensor T
arrows for A2c: 1,1,-1,-1,1,1,-1
arrows for TT: 1,-1,-1,1

returns(Al_update,Ar_update,chi+dchi,chi_spin_final,Fl,Fr)
"""
function square_pi_flux_spin_sym_two_site_update(TT,Fl,Fr,Al,Ar,dchi,T_spin,chi_spin; A2c=[],cut_ratio=0.99,e0=1e-12,elemtype=Complex128,ncv=20)
    chi=size(Fl[1],1)
    DD=size(TT[1],1)
    D=Int(sqrt(DD))
    N=2
    @show chi,dchi,e0,cut_ratio

    if A2c==[] A2c=rand(elemtype,chi,DD,chi,DD) end
    #obtain updated A2c
    A2clm=LinearMap([Fl[1],A2c,TT[1],TT[2],Fr[2]],[[1,2,-1],[1,3,6,5],[2,4,3,-2],[4,7,5,-4],[6,7,-3]],2,elemtype=elemtype)
    A2ceig_res=eigs(A2clm,nev=1,v0=A2c[:],tol=max(e0/100,1e-15),ncv=ncv)
    λA2c,A2c=A2ceig_res
    λA2c=λA2c[1]
    A2c=reshape(A2c,chi,D,D,chi,D,D)
    MA2c=spin_singlet_space_from_cg([chi_spin,T_spin,T_spin,chi_spin,T_spin,T_spin],[1,1,-1,-1,1,-1])
    @show vecnorm(A2c-sym_tensor_proj(A2c,MA2c))
    A2c=sym_tensor_proj(A2c,MA2c)

    #get spin symmetric svd
    Us,Ss,Vts,val_spin=svd_spin_sym_tensor(A2c,[1,2,3],[chi_spin,T_spin,T_spin,chi_spin,T_spin,T_spin],[1,1,-1,-1,1,-1],larrow=1)
    svals=vcat(Ss...)
    svals/=max(svals...)
    Us=map(U->reshape(U,chi,DD,div(length(U),chi*DD)),Us)
    Vts=map(Vt->reshape(Vt,div(length(Vt),chi*DD),chi,DD),Vts)

    vals_order=sortperm(svals,rev=true)
    while svals[vals_order[chi+dchi+1]]/svals[vals_order[chi+dchi]]>cut_ratio dchi+=1 end
    vals_order=vals_order[1:chi+dchi]
    @show dchi
    @show svals[vals_order]

    spin_list=unique(val_spin)
    @show spin_list
    chi_spin_final=[]
    for s in spin_list
        flavor_deg=div(count(x->spin_qn_from_ind(x,val_spin)[1]==s,vals_order),Int(2*s+1))
        append!(chi_spin_final,s*ones(flavor_deg))
    end
    @show chi_spin,chi_spin_final

    #get position of original indices in the increased bond
    npos=[]
    ind=1
    for s in spin_list
        odeg=count(x->x==s,chi_spin)*Int(2s+1)
        ndeg=count(x->x==s,chi_spin_final)*Int(2s+1)
        append!(npos,collect(ind:ind+odeg-1))
        ind=ind+ndeg
        @show odeg,ndeg,npos
    end

    #update Al by adding small random tensor
    MA=spin_singlet_space_from_cg([chi_spin_final,chi_spin_final,T_spin,T_spin],[1,-1,1,-1]) 
    MA=reshape(MA,chi+dchi,chi+dchi,DD,size(MA)[end]) 

    Al_update=zeros(elemtype,chi+dchi,chi+dchi,DD)
    Ar_update=zeros(elemtype,chi+dchi,chi+dchi,DD)
    #Al_update=1e-3*vecnorm(Al[1])*svals[vals_order[chi+1]]/svals[vals_order[chi]]*rand(elemtype,chi+dchi,chi+dchi,DD)
    #Al_update=sym_tensor_proj(Al_update,MA)
    #@show vecnorm(Al[1]),vecnorm(Al_update)

    #Ar_update=1e-3*vecnorm(Ar[1])*svals[vals_order[chi+1]]/svals[vals_order[chi]]*rand(elemtype,chi+dchi,chi+dchi,DD)
    #Ar_update=sym_tensor_proj(Ar_update,MA)

    Al_update[npos,npos,:]+=Al[1]
    Ar_update[npos,npos,:]+=Ar[1]

    #=
    #update Al and Ar by adding small perturb
    Al_update=zeros(elemtype,chi+dchi,chi+dchi,DD)
    Ar_update=zeros(elemtype,chi+dchi,chi+dchi,DD)
    iter=oind=nind=1
    for s in spin_list
        odeg=count(x->x==s,chi_spin)*Int(2s+1)
        ndeg=count(x->x==s,chi_spin_final)*Int(2s+1)
        ofdeg=count(x->x==s,chi_spin)
        nfdeg=count(x->x==s,chi_spin_final)
        @show s,iter,oind,nind,odeg,ofdeg,ndeg,nfdeg
        @show size(Us[iter]),size(Vts[iter])
        @show Ss[iter][1:ndeg]
        if ndeg==0 
            iter+=1
            continue 
        end

        #fixing gauge using posqr on flavor sector only
        Al_flavor_block_update=permutedims(Us[iter][:,:,1:Int(2s+1): ndeg],[1,3,2])
        Al_flavor_block=Al[1][:,oind:Int(2s+1): oind+odeg-1,:]
        Ql=direct_sum(posqr(reshape(permutedims(Al_flavor_block,[2,3,1]),ofdeg,chi*DD))[1],diagm(ones(nfdeg-ofdeg)))
        Ql_update=posqr(reshape(permutedims(Al_flavor_block_update,[2,3,1]),nfdeg,chi*DD))[1]
        Wlf=Ql*Ql_update'
        Wl=zeros(elemtype,ndeg,ndeg)
        for i=1:size(Wlf,1),j=1:size(Wlf,2) 
            for z=1:Int(2s+1) Wl[(i-1)*Int(2s+1)+z,(j-1)*Int(2s+1)+z]=Wlf[i,j] end
        end
        @show Al_flavor_block
        @show Al_flavor_block_update
        @show Ql
        @show Ql_update
        @show Wlf
        @show Wl

        Ar_flavor_block_update=Vts[iter][1:Int(2s+1): ndeg,:,:]
        Ar_flavor_block=Ar[1][oind:Int(2s+1): oind+odeg-1,:,:]
        Qr=direct_sum(posqr(reshape(Ar_flavor_block,ofdeg,chi*DD))[1],diagm(ones(nfdeg-ofdeg)))
        Qr_update=posqr(reshape(Ar_flavor_block_update,nfdeg,chi*DD))[1]
        Wrf=Qr*Qr_update'
        Wr=zeros(elemtype,ndeg,ndeg)
        for i=1:size(Wrf,1),j=1:size(Wrf,2) 
            for z=1:Int(2s+1) Wr[(i-1)*Int(2s+1)+z,(j-1)*Int(2s+1)+z]=Wrf[i,j] end
        end

        #update spin s sector
        Al_block_update=permutedims(Us[iter][:,:,1:ndeg],[1,3,2])
        Al_block_update=jcontract([Al_block_update,Wl],[[-1,1,-3],[-2,1]])
        Al_update[npos,nind:nind+ndeg-1,:]=Al_block_update

        Ar_block_update=Vts[iter][1:ndeg,:,:]
        Ar_block_update=jcontract([Wr,Ar_block_update],[[-1,1],[1,-2,-3]])
        Ar_update[nind:nind+ndeg-1,npos,:]=Ar_block_update


        #test
        MA=spin_singlet_space_from_cg([chi_spin_final,chi_spin_final,T_spin,T_spin],[1,-1,1,-1]) 
        MA=reshape(MA,chi+dchi,chi+dchi,DD,size(MA)[end]) 
        @show vecnorm(Al_update),vecnorm(Al_update-sym_tensor_proj(Al_update,MA))
        @show vecnorm(Ar_update),vecnorm(Ar_update-sym_tensor_proj(Ar_update,MA))

        nind+=ndeg
        oind+=odeg
        iter+=1
    end
    =#

    @show vecnorm(Al_update[npos,npos,:]-Al[1])

    Al_update=[Al_update,Al_update]
    Ar_update=[Ar_update,Ar_update]
    return Al_update,Ar_update,chi+dchi,chi_spin_final
end
