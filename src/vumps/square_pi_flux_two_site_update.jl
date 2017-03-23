
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

returns(Al_update,Ar_update,chi+dchi,chi_spin_final)
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
    #Us=map(U->reshape(U,chi,DD,div(length(U),chi*DD)),Us)
    #Vts=map(Vt->reshape(Vt,div(length(Vt),chi*DD),chi,DD),Vts)

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

    #update Al and Ar
    Al_update=zeros(elemtype,chi+dchi,chi+dchi,D,D)
    Ar_update=zeros(elemtype,chi+dchi,chi+dchi,D,D)

    #get position of original indices in the increased bond
    ninds=[]
    ind=1
    for s in spin_list
        odeg=count(x->x==s,chi_spin)*Int(2s+1)
        ndeg=count(x->x==s,chi_spin_final)*Int(2s+1)
        append!(ninds,collect(ind:ind+odeg-1))
        ind=ind+ndeg
        @show odeg,ndeg,ninds
    end

    iter=ind=1
    for s in spin_list
        deg=count(x->x==s,chi_spin_final)*Int(2s+1)
        @show s,iter,ind,deg
        @show size(Us[iter]),size(Vts[iter])
        if deg==0 
            iter+=1
            continue 
        end
        Al_update[ninds,ind:ind+deg-1,:,:]=permutedims(Us[iter][:,:,:,1:deg],[1,4,2,3])
        Ar_update[ind:ind+deg-1,ninds,:,:]=Vts[iter][1:deg,:,:,:]

        #fixing gauge

        #test
        MA=spin_singlet_space_from_cg([chi_spin_final,chi_spin_final,T_spin,T_spin],[1,-1,1,-1]) 
        #MA=reshape(MA,chi+dchi,chi+dchi,DD,size(MA)[end]) 
        @show vecnorm(Al_update),vecnorm(Al_update-sym_tensor_proj(Al_update,MA))
        @show vecnorm(Ar_update),vecnorm(Ar_update-sym_tensor_proj(Ar_update,MA))

        ind+=deg
        iter+=1
    end

    Al_update=reshape(Al_update,chi+dchi,chi+dchi,DD)
    Ar_update=reshape(Ar_update,chi+dchi,chi+dchi,DD)

    #fixing gauge
    spin_list=unique(chi_spin_final)
    Ql_temp=posqr(reshape(permutedims(Al[1],[2,3,1]),chi,chi*DD))[1]
    oind=nind=1
    Ql=zeros(elemtype,chi+dchi,chi+dchi)
    for s in spin_list
        odeg=count(x->x==s,chi_spin)*Int(2s+1)
        ndeg=count(x->x==s,chi_spin_final)*Int(2s+1)
        @show odeg,ndeg
        Ql[nind:nind+odeg-1,nind:nind+odeg-1]=Ql_temp[oind:oind+odeg-1,oind:oind+odeg-1]
        Ql[nind+odeg:nind+ndeg-1,nind+odeg:nind+ndeg-1]=diagm(ones(ndeg-odeg))
        oind+=odeg
        nind+=ndeg
    end
    Ql_update=posqr(reshape(permutedims(Al_update,[2,3,1]),chi+dchi,(chi+dchi)*DD))[1]
    Wl=Ql*Ql_update'
    @show Ql_temp
    @show Ql
    @show Ql_update
    Al_update=jcontract([Al_update,Wl],[[-1,1,-3],[-2,1]])

    Qr_temp=posqr(reshape(Ar[1],chi,chi*DD))[1]
    oind=nind=1
    Qr=zeros(elemtype,chi+dchi,chi+dchi)
    for s in spin_list
        odeg=count(x->x==s,chi_spin)*Int(2s+1)
        ndeg=count(x->x==s,chi_spin_final)*Int(2s+1)
        Qr[nind:nind+odeg-1,nind:nind+odeg-1]=Qr_temp[oind:oind+odeg-1,oind:oind+odeg-1]
        Qr[nind+odeg:nind+ndeg-1,nind+odeg:nind+ndeg-1]=diagm(ones(ndeg-odeg))
        oind+=odeg
        nind+=ndeg
    end
    Qr_update=posqr(reshape(Al_update,chi+dchi,(chi+dchi)*DD))[1]
    Wr=Qr*Qr_update'
    Ar_update=jcontract([Wr,Ar_update],[[-1,1],[1,-2,-3]])


    Al_update=[Al_update,Al_update]
    Ar_update=[Ar_update,Ar_update]
    return Al_update,Ar_update,chi+dchi,chi_spin_final
end
