#this file stores functions generating spin symmetric tensor basis using CG coeffiecient

"""
get spin quantum number from ind and spin_rep
"""
function spin_qn_from_ind(ind,spin_rep)
    for spin in spin_rep
        if 2*spin+1>=ind
            return spin,spin-ind+1
        else
            ind=ind-2*spin-1
        end
    end
    return "ind overflow"
end

"""
given three spin irreps, get fusion tensors CG_tens 
return CG_tens
"""
function CG_tensor(js,arrows)
    js*=1.
    legs_dims=[Int(2*js[i]+1) for i=1:3]
    CG_tens=zeros(legs_dims...)
    if abs(js[1]-js[2])>js[3] || js[1]+js[2]<js[3] return CG_tens end

    for inds in CartesianRange(size(CG_tens))
        ms=[js[i]-inds[i]+1 for i=1:3]
        if abs(sum(ms.*arrows))>eps(Float64) continue end

        fac=1.
        if arrows[1]==arrows[2]!=arrows[3] 
            fac=(-1)^(js[1]-js[2]+ms[3])*sqrt(2*js[3]+1)
        elseif arrows[1]==arrows[3]!=arrows[2]
            fac=(-1)^(js[3]-js[1]+ms[2])*sqrt(2*js[2]+1)
        elseif arrows[2]==arrows[3]!=arrows[1]
            fac=(-1)^(js[2]-js[3]+ms[1])*sqrt(2*js[1]+1)
        end

        CG_tens[inds]=fac*GSL.sf_coupling_3j(Int.(2*js)...,Int.(2*ms.*arrows)...)
    end

    return CG_tens;
end

"""
input THREE spin reps, and their arrows, obtain all possible fusion channels

return fusion_tens
"""
function three_spin_fusion_tensors(spin_reps,arrows)
    spin_reps*=1.
    fusion_tens=[]
    legs_dims=[sum(x->Int(2x+1),spin_reps[i]) for i=1:size(spin_reps,1)]

    ia=1
    for sa in spin_reps[1]
        ra=ia:ia+Int(2*sa)
        ib=1
        for sb in spin_reps[2]
            rb=ib:ib+Int(2*sb)
            ic=1
            for sc in spin_reps[3]
                rc=ic:ic+Int(2*sc)
                if abs(sa-sb)<=sc<=sa+sb && abs((sa+sb+sc)%1)<eps(Float64)
                    CG_tens=zeros(legs_dims...)
                    CG_tens[ra,rb,rc]=CG_tensor([sa,sb,sc],arrows)
                    @show sa,sb,sc
                    push!(fusion_tens,CG_tens)
                end
                ic=rc[end]+1
            end
            ib=rb[end]+1
        end
        ia=ra[end]+1
    end

    return fusion_tens
end


"""
input spin_reps and arrows
generating spin singlet tensor basis M
return M
"""
function spin_singlet_space_from_cg(spin_reps,arrows)
    nlegs=size(spin_reps,1)
    legs_dims=[sum(x->Int(2x+1),spin_reps[i]) for i=1:nlegs]
    M=[]
    c_irreps=[]

    if nlegs==1 return 0 end
    if nlegs==2
        M=three_spin_fusion_tensors([spin_reps[0],spin_reps[1],[0]],[arrows[0],arrows[1],1])
        println("singlet subspace dims:",size(M,1))
        M=reshape(hcat(M...),legs_dims...,size(M,1))
        return M
    end

    #initialize for the first two spins
    for sc=0:max(spin_reps[1])+max(spin_reps[2])
        fusion_tens=three_spin_fusion_tensors([spin_reps[1],spin_reps[2],[sc]],[arrows[1],arrows[2],1])
        append!(M,fusion_tens)
        append!(c_irreps,[sc for i=1:size(fusion_tens,1)])
    end

    #middle legs
    for legi=3:nlegs-2
        M_next=[]
        c_irreps_next=[]

        for basei=1:size(M,1)
            sc_min=min(abs(c_irreps[basei]-min(spin_reps[legi])),abs(max(spin_reps[legi])-c_irreps[basei]))
            sc_max=c_irreps[basei]+max(spin_reps[legi])
            
            for sc=sc_min:sc_max
                fusion_tens=three_spin_fusion_tensors([[c_irreps[basei]],spin_reps[legi],[sc]],[-1,arrows[legi],1])
                append!(M_next,map(tens->jcontract([M[basei],tens],[[(-1:-1:legi+1)...,1],[1,-legi,-legi-1]]),fusion_tens))
                append!(c_irreps_next,[sc for i=1:size(fusion_tens,1)])
            end

        end

        M=M_next
        c_irreps=c_irreps_next
    end

    #the last two legs
    M_final=[]
    for basei=1:size(M,1)
        fusion_tens=three_spin_fusion_tensors([[c_irreps[basei]],spin_reps[end-1],spin_reps[end]],[-1,arrows[end-1],arrows[end]])
        append!(M_final,map(tens->jcontract([M[basei],tens],[[(-1:-1:-nlegs+2)...,1],[1,-nlegs+1,-nlegs]])),fusion_tens)
    end
    M=M_final

    println("singlet subspace dims:",size(M,1))
    M=reshape(hcat(M...),legs_dims...,size(M,1))
    return M
end
