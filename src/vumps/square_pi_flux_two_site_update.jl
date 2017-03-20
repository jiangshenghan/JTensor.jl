
"""
Increase cutoff bond dimension for square pi flux
Here, T,Al,Ar all contains two tensors
   
   ----A2c----
  /   /   \    \         ---A2c---
Fl--T[1]--T[2]--Fr  ~       | |
  \  |     |   /

legs_order for A2c: left,ld,right,rd
spin_reps: for "phys" leg, for "virt" leg

returns(Al,Ar,free_energy,err)
"""
function square_pi_flux_spin_sym_two_site_update(T,Al,Ar,spin_reps,dchi,Fl,Fr,A2c=[];e0=1e-12,elemtype=Complex128,ncv=20)
    chi=size(Al,1)
    D=size(T[1],1)
    N=2
    @show chi,dchi,e0,D

    if A2c==[] A2c=rand(elemtype,chi,D,chi,D) end
    #obtain updated A2c
    A2clm=LinearMap([Fl,A2c,T[1],T[2],Fr],[[1,2,-1],[1,3,6,5],[2,4,3,-2],[4,7,5,-4],[6,7,-3]],2,elemtype=elemtype)
    A2ceig_res=eigs(A2clm,nev=1,v0=A2c[:],tol=max(e0/100,1e-15))
    λA2c,A2c=A2ceig_res
    λA2c=λA2c[1]
    A2c=reshape(A2c,chi,D,chi,D)

    #update Al and Ar
    svd_spin_sym_tensor(A2c,[1,2],[spin_reps[2],spin_reps[1],spin_reps[2],spin_reps[1]],[1,1,-1,1])

end
