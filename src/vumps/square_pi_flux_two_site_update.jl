
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

returns(Al,Ar,chi+dchi,val_spin)
"""
function square_pi_flux_spin_sym_two_site_update(TT,Fl,Fr,dchi,T_spin,chi_spin; A2c=[],cut_ratio=0.8,e0=1e-12,elemtype=Complex128,ncv=20)
    chi=size(Al,1)
    DD=size(TT[1],1)
    D=Int(sqrt(DD))
    N=2
    @show chi,dchi,e0,cut_ratio

    if A2c==[] A2c=rand(elemtype,chi,DD,chi,DD) end
    #obtain updated A2c
    A2clm=LinearMap([Fl,A2c,TT[1],TT[2],Fr],[[1,2,-1],[1,3,6,5],[2,4,3,-2],[4,7,5,-4],[6,7,-3]],2,elemtype=elemtype)
    A2ceig_res=eigs(A2clm,nev=1,v0=A2c[:],tol=max(e0/100,1e-15),ncv=ncv)
    λA2c,A2c=A2ceig_res
    λA2c=λA2c[1]
    A2c=reshape(A2c,chi,D,D,chi,D,D)
    MA2c=spin_singlet_space_from_cg([chi_spin,T_spin,T_spin,chi_spin,T_spin,T_spin],[1,1,-1,-1,1,-1])
    A2c=sym_tensor_proj(A2c,MA2c)

    #get spin symmetric svd
    Us,Ss,Vts,val_spin=svd_spin_sym_tensor(A2c,[1,2,3],[chi_spin,T_spin,T_spin,chi_spin,T_spin,T_spin],[1,-1,-1,1,-1,1])
    svals=vcat(Ss...)
    map(U->reshape(U,chi,DD,chi*DD),Us)
    map(Vt->reshape(Vt,chi*DD,chi,DD),Vts)

    vals_order=sortperm(svals,rev=true)
    while svals[vals_order[chi+dchi+1]]/svals[vals_order[chi+dchi]]>cut_ratio dchi+=1 end
    vals_order=vals_order[1:chi+dchi]

    smin,smax=min(val_spin),max(val_spin)
    val_spin_final=[]
    for s=smin:0.5:smax
        flavor_deg=div(count(x->spin_qn_from_ind(x,val_spin)==s,vals_order),Int(2*s+1))
        append!(val_spin_final,s*ones(flavor_deg))
    end
    @show val_spin_final

    #update Al and Ar
    Al=zeros(chi+dchi,chi+dchi,DD)
    Ar=zeros(chi+dchi,chi+dchi,DD)

    iter=ind=1
    for s=smin:0.5:smax
        deg=count(s->s==val_spin)*Int(2s+1)
        if deg==0 continue end
        deg=count(s->s==val_spin_final)*Int(2s+1)
        Al[1:chi,ind:ind+deg-1,:]=Us[iter][:,:,1:deg]
        Ar[ind:ind+deg-1,1:chi,:]=Vts[iter][1:deg,:,:]
        iter+=1
        ind+=deg
    end

    return Al,Ar,chi+dchi,val_spin_final
end
