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
      \   |

where U/Vt is cut to dchi cols/rows, and dchi should be chosen to perserve spin deg
We should keep track of spin_reps. Here pspin stores spin reps for a single physical spin. For dlmps, one tensor contains two physical spins

updated:
updated_Al= Al (Nl^*)U    updated_Ar= Ar       0    updated_C= C 0
            0  0                      Vt(Nr^*) 0               0 0

return updated_Al,updated_Ar,updated_chi,updated_chi_spin
"""
function spin_sym_dlmps_inc_chi(Al,Ar,A2c,inc_spin_no,pspin,chi_spin,Aarrows)
    D=Int(sum(x->2x+1,pspin))
    chi=Int(sum(x->2x+1,chi_spin))
    Aspin_reps=[chi_spin,chi_spin,pspin,pspin]

    #obtain null space
    Nl,lspin_rep=spin_sym_tensor_nullspace(reshape(Al,chi,chi,D,D),[2],Aspin_reps,Aarrows,null_leg_arrow=-1)
    Nr,rspin_rep=spin_sym_tensor_nullspace(reshape(Ar,chi,chi,D,D),[1],Aspin_reps,Aarrows,null_leg_arrow=1)
    Nl=reshape(Nl,chi,D^2,size(Nl,4))
    Nr=reshape(Nr,chi,D^2,size(Nr,4))
    @show lspin_rep
    @show rspin_rep
    @show size(Nl),size(Nr)
    if lspin_rep!=rspin_rep error("incorrect null spin reps") end

    #SVD on the new basis & keep only largest svals to get updated spin reps 
    proj_A2c=jcontract([Nl,A2c,Nr],[[1,2,-1],[1,4,2,3],[4,3,-2]])
    Us,Ss,Vts,vals_spin_rep,spin_species=svd_spin_sym_tensor(proj_A2c,[1],[lspin_rep,rspin_rep],[-1,1],larrow=Aarrows[1])
    #svals_unique are singular value without spin deg
    svals_unique=Float64[]
    for i=1:length(spin_species)
        append!(svals_unique,Ss[i][1:Int(2*spin_species[i]+1):length(Ss[i])])
    end
    svals_ordered=sort(svals_unique,rev=true)
    n0=inc_spin_no
    while svals_ordered[inc_spin_no]*0.99<svals_ordered[inc_spin_no+1]
        println("add more spins due to small gaps of svals!")
        inc_spin_no+=1 
    end
    inc_spin_rep=sort(vals_spin_rep[sortperm(svals_unique,rev=true)[1:inc_spin_no]])
    inc_chi=Int(sum((x->2x+1),inc_spin_rep))
    #@show svd(reshape(permutedims(A2c,[1,3,2,4]),chi*D^2,chi*D^2))[2]
    @show svals_ordered[1:inc_spin_no]
    @show inc_spin_no
    @show inc_spin_rep,inc_chi

    #get new spin basis stored in U and Vt
    U=zeros(eltype(Us[1]),size(Us[1],1),inc_chi)
    Vt=zeros(eltype(Vts[1]),inc_chi,size(Vts[1],2))
    svals_remain=zeros(eltype(Ss[1]),inc_chi)
    ind=1
    for i=1:length(spin_species)
        ni=count(x->x==spin_species[i],inc_spin_rep)
        if ni==0 continue end
        vec_range=1:Int(ni*(2*spin_species[i]+1))
        U[:,ind+vec_range-1]=Us[i][:,vec_range]
        Vt[ind+vec_range-1,:]=Vts[i][vec_range,:]
        svals_remain[ind+vec_range-1,:]=Ss[i][vec_range]
        ind+=vec_range[end]
    end

    #=
    #check U,Vt spin sym
    MU=spin_singlet_space_from_cg([lspin_rep,inc_spin_rep],[-1,-Aarrows[1]])
    MV=spin_singlet_space_from_cg([inc_spin_rep,rspin_rep],[Aarrows[1],1])
    @show vecnorm(sym_tensor_proj(U,MU)-U)
    @show vecnorm(sym_tensor_proj(Vt,MV)-Vt)
    =#

    #update MPS
    updated_chi_spin=append!(chi_spin,inc_spin_rep)
    updated_chi=Int(sum(x->2x+1,updated_chi_spin))
    @show updated_chi_spin,updated_chi

    updated_Al=zeros(eltype(Al),updated_chi,updated_chi,D^2)
    updated_Al[1:chi,1:chi,:]=Al
    updated_Al[1:chi,chi+1:updated_chi,:]=jcontract([conj(Nl),U],[[-1,-3,1],[1,-2]])
    updated_Ar=zeros(eltype(Ar),updated_chi,updated_chi,D^2)
    updated_Ar[1:chi,1:chi,:]=Ar
    updated_Ar[chi+1:updated_chi,1:chi,:]=jcontract([Vt,conj(Nr)],[[-1,1],[-2,-3,1]])

    #=
    #check updated_Al/r spin symmetric
    MA=spin_singlet_space_from_cg([updated_chi_spin,updated_chi_spin,pspin,pspin],Aarrows)
    MA=reshape(MA,updated_chi,updated_chi,D^2,size(MA)[end])
    @show vecnorm(sym_tensor_proj(updated_Al,MA)-updated_Al)
    @show vecnorm(sym_tensor_proj(updated_Ar,MA)-updated_Ar)
    =#

    println()

    return updated_Al,updated_Ar,updated_chi,updated_chi_spin
end


"""
Increase bond dimension of uMPS from chi_old to chi with one site per uc

first generate A2c
   --Al--C--Ar--
  /  |      |   \          ---A2c---
Fl---Wh-----Wh---Fr  =        / \
  \  |      |   /
     |      |   
legs order for A2c: left,right,dl,dr

Then, SVD and truncate 
   ----A2c----                           ---Al--         --Ar---
   |  /   \  |  = --U--S--Vt--,  where   |  |     =0,      |   | =0
   --Nl   Nr--                           ---Nl--         --Nr---
     |    |

where U/Vt is cut to chi cols/rows

updated:
updated_Al= Al (Nl^*)U    updated_Ar= Ar       0    updated_C= C 0
            0  0                      Vt(Nr^*) 0               0 0

return updated_Al, updated_Ar, updated_C
"""
function one_site_vumps_inc_chi(Wh,chi,Al,Ar,C,Fl,Fr)
    dp,chi_old=size(Al,3,1)

    #obtain A2c
    A2c=jcontract([Fl,Al,Wh,C,Ar,Wh,Fr],[[1,2,-1],[1,4,3],[2,6,3,-3],[4,5],[5,8,7],[6,9,7,-4],[8,9,-2]])

    #obtain null space
    Nl=nullspace(reshape(permutedims(Al,[2,1,3]),chi_old,chi_old*dp))
    Nr=nullspace(reshape(Ar,chi_old,chi_old*dp))
    Nl=reshape(Nl,chi_old,dp,(dp-1)*chi_old)
    Nr=reshape(Nr,chi_old,dp,(dp-1)*chi_old)
    #check null condition and normalization
    @show vecnorm(jcontract([Al,Nl],[[1,-1,2],[1,2,-2]])) 
    @show vecnorm(jcontract([Ar,Nr],[[-1,1,2],[1,2,-2]]))
    @show diag(jcontract([Nl,conj(Nl)],[[1,2,-1],[1,2,-2]]))
    @show diag(jcontract([Nr,conj(Nr)],[[1,2,-1],[1,2,-2]]))

    #SVD on the new basis
    proj_A2c=jcontract([Nl,A2c,Nr],[[1,2,-1],[1,4,2,3],[4,3,-2]])
    svd_res=svdfact(proj_A2c)
    dchi=chi-chi_old
    U=svd_res[:U][:,1:dchi]
    S=svd_res[:S][1:dchi]
    Vt=svd_res[:Vt][1:dchi,:]

    #update tensor
    updated_Al=zeros(eltype(Al),chi,chi,dp)
    updated_Al[1:chi_old,1:chi_old,:]=Al
    updated_Al[1:chi_old,chi_old+1:chi,:]=jcontract([conj(Nl),U],[[-1,-3,1],[1,-2]])

    updated_Ar=zeros(eltype(Ar),chi,chi,dp)
    updated_Ar[1:chi_old,1:chi_old,:]=Ar
    updated_Ar[chi_old+1:chi,1:chi_old,:]=jcontract([Vt,conj(Nr)],[[-1,1],[-2,-3,1]])

    updated_C=zeros(eltype(C),chi,chi)
    updated_C[1:chi_old,1:chi_old]=C

    return updated_Al,updated_Ar, updated_C
end
