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

return {U^{si}},{S^{si}},{Vt^{si}},{spins}
"""
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
    @show rl,rr,leg_order
    @show lmax_spin,rmax_spin,max_spin


    Us=[]
    Ss=[]
    Vts=[]
    spins=[]

    for si=0:0.5:max_spin
        ML=spin_singlet_space_from_cg([spin_reps[left_legs]...,[si]],[arrows[left_legs]...,-1])*sqrt(2*si+1)
        MR=spin_singlet_space_from_cg([spin_reps[right_legs]...,[si]],[arrows[right_legs]...,1])*sqrt(2*si+1)
        if ML==[] || MR==[] continue end

        @show si
        @show reshape(jcontract([ML,conj(ML)],[[rl...,-1,-2],[rl...,-3,-4]]),size(ML)[end]*Int(2*si+1),size(ML)[end]*Int(2*si+1))
        @show reshape(jcontract([MR,conj(MR)],[[rr...,-1,-2],[rr...,-3,-4]]),size(MR)[end]*Int(2*si+1),size(MR)[end]*Int(2*si+1))

        Tsi=jcontract([ML[[Colon() for i=rl]...,1,:],T,MR[[Colon() for i=rr]...,1,:]],[[rl...,-1],leg_order,[(length(left_legs)+rr)...,-2]])
        @show Tsi

        svd_res=svdfact(Tsi)
        Usi=reshape(jcontract([svd_res[:U],eye(Int(2*si+1))],[[-1,-3],[-2,-4]]),size(svd_res[:U],1),Int(2*si+1),size(svd_res[:U],2)*Int(2*si+1))
        Vtsi=reshape(jcontract([svd_res[:Vt],eye(Int(2*si+1))],[[-1,-3],[-2,-4]]),size(svd_res[:Vt],1)*Int(2*si+1),size(svd_res[:Vt],2),Int(2*si+1))

        push!(Ss,repeat(svd_res[:S],inner=Int(2*si+1)))
        push!(Us,jcontract([conj(ML),Usi],[[-rl...,1,2],[2,1,-rl[end]-1]]))
        push!(Vts,jcontract([Vtsi,conj(MR)],[[-1,2,1],[(-rr-1)...,1,2]]))
        push!(spins,si)
    end

    return Us,Ss,Vts,spins
end
