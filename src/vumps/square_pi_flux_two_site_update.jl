
"""
Increase cutoff bond dimension for square pi flux
Here, T,Al,Ar all contains two tensors
   
   ----A2c----
  /   /   \    \         ---A2c---
Fl--T[1]--T[2]--Fr  ~       | |
  \  |     |   /

legs_order for A2c: left,ld,right,rd
legs_order for Al,Ar: l,r,d

returns(Al,Ar,chi+dchi,vals_spin_rep)
"""
function square_pi_flux_spin_sym_two_site_update(T,Fl,Fr,dchi,T_spin_rep,chi_spin_rep; A2c=[],cut_ratio=0.8,e0=1e-12,elemtype=Complex128,ncv=20)
    chi=size(Al,1)
    D=size(T[1],1)
    N=2
    @show chi,dchi,e0,cut_ratio

    if A2c==[] A2c=rand(elemtype,chi,D,chi,D) end
    #obtain updated A2c
    A2clm=LinearMap([Fl,A2c,T[1],T[2],Fr],[[1,2,-1],[1,3,6,5],[2,4,3,-2],[4,7,5,-4],[6,7,-3]],2,elemtype=elemtype)
    A2ceig_res=eigs(A2clm,nev=1,v0=A2c[:],tol=max(e0/100,1e-15),ncv=ncv)
    λA2c,A2c=A2ceig_res
    λA2c=λA2c[1]
    A2c=reshape(A2c,chi,D,chi,D)

    #get spin symmetric svd
    Us,Ss,Vts,vals_spin_rep=svd_spin_sym_tensor(A2c,[1,2],[chi_spin_rep,T_spin_rep,chi_spin_rep,T_spin_rep],[-1,1,1,1])
    svals=vcat(Ss...)
    vals_order=sortperm(svals,rev=true)
    while svals[vals_order[chi+dchi+1]]/svals[vals_order[chi+dchi]]>cut_ratio dchi+=1 end
    vals_order=vals_order[1:chi+dchi]
    smin,smax=min(vals_spin_rep),max(vals_spin_rep)
    vals_spin_rep_final=[]
    for s=smin:0.5:smax
        flavor_deg=div(count(x->spin_qn_from_ind(x,vals_spin_rep)==s,vals_order),Int(2*s+1))
        append!(vals_spin_rep_final,s*ones(flavor_deg))
    end
    @show vals_spin_rep_final

    #update Al and Ar
    Al=zeros(chi+dchi,chi+dchi,D)
    Ar=zeros(chi+dchi,chi+dchi,D)

    iter=ind=1
    for s=smin:0.5:smax
        deg=count(s->s==vals_spin_rep)*Int(2s+1)
        if deg==0 continue end
        deg=count(s->s==vals_spin_rep_final)*Int(2s+1)
        Al[1:chi,ind:ind+deg-1,:]=Us[iter][:,:,1:deg]
        Ar[ind:ind+deg-1,1:chi,:]=Vts[iter][1:deg,:,:]
        iter+=1
        ind+=deg
    end

    return Al,Ar,chi+dchi,vals_spin_rep_final
end
