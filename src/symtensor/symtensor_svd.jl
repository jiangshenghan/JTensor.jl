
"""
svd decomposition for spin symmetric tensor

 \ /      \       /
--T-- =  --U--S--V--  
 / \      /       \

pay attention to legs_order, may differ by a permutation

U=oplus_{si}(U^{si})=oplus_{si}(U'^{si}\otimes I_{2si+1})
S=oplus_{si}(S^{si})=oplus_{si}(S'^{si}\otimes I_{2si+1})
Vt=oplus_{si}(Vt^{si}=oplus_{si}(Vt'^{si}\otimes I_{2si+1})

leg order for U: left_legs...,Sl_leg
leg order for V: Sr_leg, right_legs...

vals_spin_rep is spin reps for singular values
spin_species[i]=s[i]

return {U^{si}},{S^{si}},{Vt^{si}},vals_spin_rep,spin_species
"""
function svd_spin_sym_tensor(T,left_legs,spin_reps,arrows;larrow=-1,thin=true)
    right_legs=setdiff(collect(1:ndims(T)),left_legs)
    rl=1:length(left_legs)
    rr=1:length(right_legs)
    leg_order=zeros(Int,ndims(T))
    for i=rl leg_order[left_legs[i]]=i end
    for i=rr leg_order[right_legs[i]]=i+length(left_legs) end
    lmax_spin=sum(i->max(spin_reps[left_legs[i]]...),rl)
    rmax_spin=sum(i->max(spin_reps[right_legs[i]]...),rr)

    if thin max_spin=min(lmax_spin,rmax_spin)
    else max_spin=max(lmax_spin,rmax_spin) end

    Us=[]
    Ss=[]
    Vts=[]
    vals_spin_rep=Float64[]
    spin_species=Float64[]

    for si=0:0.5:max_spin
        ML=spin_singlet_space_from_cg([spin_reps[left_legs]...,[si]],[-arrows[left_legs]...,larrow])*sqrt(2*si+1)
        MR=spin_singlet_space_from_cg([spin_reps[right_legs]...,[si]],[-arrows[right_legs]...,-larrow])*sqrt(2*si+1)
        if ML==[] && MR==[] continue end
        if ML==[] || MR==[] 
            if thin==false
                append!(spin_species,si)
                push!(Ss,[])
                if ML==[] 
                    push!(Us,ML)
                else 
                    Usi=reshape(ML,size(ML,(1:ndims(ML)-2)...)...,prod(size(ML,ndims(ML),ndims(ML)-1)))
                    Usi=conj(Usi)
                    push!(Us,Usi)
                    #=
                    #check spin symmetric
                    MU=spin_singlet_space_from_cg([spin_reps[left_legs]...,si*ones(size(ML,ndims(ML)))],[arrows[left_legs]...,-larrow])
                    @show si,vecnorm(Usi-sym_tensor_proj(Usi,MU))
                    =#
                end
                if MR==[] 
                    push!(Vts,MR)
                else 
                    Vsi=reshape(MR,size(MR,(1:ndims(MR)-2)...)...,prod(size(MR,ndims(MR),ndims(MR)-1)))
                    Vsi=permutedims(Vsi,[ndims(Vsi),(1:ndims(Vsi)-1)...])
                    Vsi=conj(Vsi)
                    push!(Vts,Vsi)
                    #=
                    #check spin symmetric
                    MV=spin_singlet_space_from_cg([si*ones(size(MR,ndims(MR))),spin_reps[right_legs]...],[larrow,arrows[right_legs]...])
                    @show si,vecnorm(Vsi-sym_tensor_proj(Vsi,MV))
                    =#
                end
            end
            continue
        end
        #for thin==false, we should include
        if ML==[] && MR!=[]
        end

        Tsi=jcontract([ML[[Colon() for i=rl]...,1,:],T,MR[[Colon() for i=rr]...,1,:]],[[rl...,-1],leg_order,[(length(left_legs)+rr)...,-2]])
        svd_res=svdfact(Tsi,thin=thin)
        Usi=reshape(jcontract([eye(Int(2*si+1)),svd_res[:U]],[[-1,-3],[-2,-4]]),Int(2*si+1),size(svd_res[:U],1),Int(2*si+1)*size(svd_res[:U],2))
        Vtsi=reshape(jcontract([eye(Int(2*si+1)),svd_res[:Vt]],[[-1,-3],[-2,-4]]),Int(2*si+1)*size(svd_res[:Vt],1),Int(2*si+1),size(svd_res[:Vt],2))

        push!(Ss,repeat(svd_res[:S],inner=Int(2*si+1)))
        push!(Us,jcontract([conj(ML),Usi],[[-rl...,1,2],[1,2,-rl[end]-1]]))
        push!(Vts,jcontract([Vtsi,conj(MR)],[[-1,1,2],[(-rr-1)...,1,2]]))
        append!(vals_spin_rep,si*ones(length(svd_res[:S])))
        append!(spin_species,si)

        #=
        #check spin symmetric
        MU=spin_singlet_space_from_cg([spin_reps[left_legs]...,si*ones(size(svd_res[:U],2))],[arrows[left_legs]...,-larrow])
        @show si,vecnorm(Us[end]-sym_tensor_proj(Us[end],MU))
        =#
    end

    #test svd
    #check T if spin symmetric
    #MT=spin_singlet_space_from_cg(spin_reps,arrows)
    #@show vecnorm(T-sym_tensor_proj(T,MT))
    #U=zeros(Complex128,size(T)[left_legs]...,sum(i->size(Us[i])[end],1:length(Us)))
    #Vt=zeros(Complex128,sum(i->size(Vts[i],1),1:length(Vts)),size(T)[right_legs]...)
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
    #@show vecnorm(jcontract([U,diagm(S),Vt],[[-rl...,1],[1,2],[2,(-rr-rl[end])...]])-permutedims(T,[left_legs...,right_legs...]))

    #@show Ss,vals_spin_rep
    #println()

    return Us,Ss,Vts,vals_spin_rep,spin_species
end

#=
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
=#
