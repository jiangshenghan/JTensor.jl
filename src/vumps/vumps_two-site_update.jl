#This file stores functions for two-site update

"""
Obtain updated A2c for magnetic translational symmetric systems

   --Jc--Al--Jc--Ac--Jc--
  /      |       |       \          ---A2c---
Fl-------T-------T-------Fr  =         / \
  \      |       |       /
   --                  --

legs order for A2c: left,right,dl,dr

return A2c
"""
function mag_trans_A2c(T,Fl,Fr,Al,Ac,Jc)
    return jcontract([Fl,Jc,Al,T,Jc,Ac,T,Jc,Fr],[[1,2,-1],[1,3],[3,5,4],[2,6,4,-3],[5,7],[7,9,8],[6,10,8,-4],[9,11],[11,10,-2]])
end


"""
For spin symmetric dl MPS, given Al,Ar,A2c, obtain updated Al',Ar',C' with increase bond dimension

   ----A2c----                           ---Al/r--
   |  /   \  |  = >-U--S--Vt->,  where   |  |       =0
   --Nl   Nr--                           ---Nl/r--
      |   |

where U/Vt is cut to chi*(chi+dchi)/(chi+dchi)*chi, and dchi should be chosen to perserve spin deg
We should keep track of spin_reps. Here pspin_rep stores spin reps for a single physical spin. For dlmps, one tensor contains two physical spins

updated:
updated_Al= Al (Nl^*)U    updated_Ar= Ar       0    updated_C= C 0
            0  0                      Vt(Nr^*) 0               0 0

return updated_Al,updated_Ar,updated_chi,updated_vspin_rep
"""
function spin_sym_dlmps_incD(Al,Ar,A2c,inc_spin_no,pspin_rep,vspin_rep,Aarrows)
    D=Int(sum(x->2x+1,pspin_rep))
    chi=Int(sum(x->2x+1,vspin_rep))
    Aspin_reps=[vspin_rep,vspin_rep,pspin_rep,pspin_rep]

    #obtain null space
    Nl,lspin_rep=spin_sym_tensor_nullspace(reshape(Al,chi,chi,D,D),[2],Aspin_reps,Aarrows,null_leg_arrow=-1)
    Nr,rspin_rep=spin_sym_tensor_nullspace(reshape(Ar,chi,chi,D,D),[1],Aspin_reps,Aarrows,null_leg_arrow=1)
    Nl=reshape(Nl,chi,D^2,size(Nl,4))
    Nl=reshape(Nr,chi,D^2,size(Nr,4))
    @show lspin_rep,rspin_rep
    if lspin!=rspin error("incorrect null spin reps") end

    #SVD & cutoff to get new spin reps 
    proj_A2c=jcontract([Nl,A2c,Nr],[[1,2,-1],[1,4,2,3],[4,3,-2]])
    Us,Ss,Vts,vals_spin_rep,spin_species=svd_spin_sym_tensor(proj_A2c,[1],[lspin_rep,rspin_rep],[-1,1],larrow=-1);
    vals=Float64[]
    vals=...
    U=...
    Vt=...
    inc_spin_rep=vals_spin_rep(sortperm(vals)[1:inc_spin_no])
    @show inc_spin_rep
    @show vecnorm(proj_A2c-U.diagm(vals).Vt)

    #update MPS
    updated_vspin_rep=append(vspin_rep,inc_spin_rep)
    updated_chi=Int(sum(x->2x+1,updated_vspin_rep))
    @show updated_vspin_rep,updated_chi
    updated_Al=zeros(eltype(Al),dchi,dchi,DD)
    updated_Al[1:chi,1:chi,;]=Al
    updated_Al[1:chi,chi+1:updated_chi,;]=jcontract([conj(Nl),U],[[-1,-3,1],[1,-2]])
    updated_Ar=zeros(eltype(Ar),dchi,dchi,DD)
    updated_Ar[1:chi,1:chi,;]=Ar
    updated_Ar[chi+1:updated_chi,1:chi,;]=jcontract([Vt,conj(Nr)],[[-1,1],[-2,-3,1]])

    return updated_Al,updated_Ar,updated_chi,updated_vspin_rep
end
