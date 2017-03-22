# this file stores multiple methods for matrix decomposition

"""
polar decomposition X=U*A with A positive semidefinite
return U,A
"""
function polardecomp(X)
    U,S,V=svd(X)
    return U*V',V*diagm(S)*V'
end


"""
svd decomposition for spin symmetric tensor

 \ /      \       /
--T-- =  --U--S--V--  
 / \      /       \

pay attention to legs_order, may differ by a permutation

U=oplus_{si}(U^{si})=oplus_{si}(U'^{si}\otimes I_{2si+1})
S=oplus_{si}(S^{si})=oplus_{si}(S'^{si}\otimes I_{2si+1})
Vt=oplus_{si}(Vt^{si}=oplus_{si}(Vt'^{si}\otimes I_{2si+1})

return {U^{si}},{S^{si}},{Vt^{si}},{vals_spin_rep}
"""
#TODO:check the case for left_legs!=[1,2,...,]
function svd_spin_sym_tensor(T,left_legs,spin_reps,arrows)
    right_legs=setdiff(collect(1:ndims(T)),left_legs)
    rl=1:length(left_legs)
    rr=1:length(right_legs)
    leg_order=zeros(Int,ndims(T))
    for i=rl leg_order[left_legs[i]]=i end
    for i=rr leg_order[right_legs[i]]=i+length(left_legs) end
    lmax_spin=sum(i->max(spin_reps[left_legs[i]]...),rl)
    rmax_spin=sum(i->max(spin_reps[right_legs[i]]...),rr)
    max_spin=min(lmax_spin,rmax_spin)

    Us=[]
    Ss=[]
    Vts=[]
    vals_spin_rep=[]

    for si=0:0.5:max_spin
        ML=spin_singlet_space_from_cg([spin_reps[left_legs]...,[si]],[-arrows[left_legs]...,-1])*sqrt(2*si+1)
        MR=spin_singlet_space_from_cg([spin_reps[right_legs]...,[si]],[-arrows[right_legs]...,1])*sqrt(2*si+1)
        if ML==[] || MR==[] continue end

        Tsi=jcontract([ML[[Colon() for i=rl]...,1,:],T,MR[[Colon() for i=rr]...,1,:]],[[rl...,-1],leg_order,[(length(left_legs)+rr)...,-2]])
        svd_res=svdfact(Tsi)
        Usi=reshape(jcontract([eye(Int(2*si+1)),svd_res[:U]],[[-1,-3],[-2,-4]]),Int(2*si+1),size(svd_res[:U],1),Int(2*si+1)*size(svd_res[:U],2))
        Vtsi=reshape(jcontract([eye(Int(2*si+1)),svd_res[:Vt]],[[-1,-3],[-2,-4]]),Int(2*si+1)*size(svd_res[:Vt],1),Int(2*si+1),size(svd_res[:Vt],2))

        push!(Ss,repeat(svd_res[:S],inner=Int(2*si+1)))
        push!(Us,jcontract([conj(ML),Usi],[[-rl...,1,2],[1,2,-rl[end]-1]]))
        push!(Vts,jcontract([Vtsi,conj(MR)],[[-1,1,2],[(-rr-1)...,1,2]]))
        append!(vals_spin_rep,si*ones(length(svd_res[:S])))
    end

    #test svd
    #U=zeros(size(T)[left_legs]...,sum(i->size(Us[i])[end],1:length(Us)))
    #Vt=zeros(sum(i->size(Vts[i],1),1:length(Vts)),size(T)[right_legs]...)
    #ind=1
    #for i=1:length(Us)
    #    U[[Colon() for k=rl]...,ind:ind+size(Us[i])[end]-1]=Us[i] 
    #    ind+=size(Us[i])[end]
    #end
    #ind=1
    #for i=1:length(Vts)
    #    Vt[ind:ind+size(Vts[i],1)-1,[Colon() for k=rr]...]=Vts[i] 
    #    ind+=size(Vts[i],1)
    #end
    #S=vcat([Ss[i] for i=1:length(Ss)]...)
    #@show size(jcontract([U,diagm(S),Vt],[[-rl...,1],[1,2],[2,(-rr-rl[end])...]]))
    #@show vecnorm(jcontract([U,diagm(S),Vt],[[-rl...,1],[1,2],[2,(-rr-rl[end])...]])-permutedims(T,[left_legs...,right_legs...]))

    @show vals_spin_rep,Ss

    return Us,Ss,Vts,vals_spin_rep
end


function svd_spin_sym_tensor(T,left_legs,spin_reps)
    right_legs=setdiff(collect(1:ndims(T)),left_legs)
    rl=1:length(left_legs)
    rr=1:length(right_legs)
    leg_order=zeros(Int,ndims(T))
    for i=rl leg_order[left_legs[i]]=i end
    for i=rr leg_order[right_legs[i]]=i+length(left_legs) end
    lmax_spin=sum(i->max(abs(spin_reps[left_legs[i]])...),rl)
    rmax_spin=sum(i->max(abs(spin_reps[right_legs[i]])...),rr)
    max_spin=min(lmax_spin,rmax_spin)

    Us=[]
    Ss=[]
    Vts=[]
    vals_spin_rep=[]

    for si=0:0.5:max_spin
        ML=spin_singlet_space_from_cg([-spin_reps[left_legs]...,[-si]])*sqrt(2*si+1)
        MR=spin_singlet_space_from_cg([-spin_reps[right_legs]...,[si]])*sqrt(2*si+1)
        if ML==[] || MR==[] continue end

        Tsi=jcontract([ML[[Colon() for i=rl]...,1,:],T,MR[[Colon() for i=rr]...,1,:]],[[rl...,-1],leg_order,[(length(left_legs)+rr)...,-2]])
        svd_res=svdfact(Tsi)
        Usi=reshape(jcontract([eye(Int(2*si+1)),svd_res[:U]],[[-1,-3],[-2,-4]]),Int(2*si+1),size(svd_res[:U],1),Int(2*si+1)*size(svd_res[:U],2))
        Vtsi=reshape(jcontract([eye(Int(2*si+1)),svd_res[:Vt]],[[-1,-3],[-2,-4]]),Int(2*si+1)*size(svd_res[:Vt],1),Int(2*si+1),size(svd_res[:Vt],2))

        push!(Ss,repeat(svd_res[:S],inner=Int(2*si+1)))
        push!(Us,jcontract([conj(ML),Usi],[[-rl...,1,2],[1,2,-rl[end]-1]]))
        push!(Vts,jcontract([Vtsi,conj(MR)],[[-1,1,2],[(-rr-1)...,1,2]]))
        append!(vals_spin_rep,si*ones(length(svd_res[:S])))
    end

    return Us,Ss,Vts,vals_spin_rep
end
