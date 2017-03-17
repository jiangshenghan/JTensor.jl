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
input spin reps, and arrows, obtain all possible fusion channels

return fusion_tens
"""
function spin_fusion_tensors(spin_reps,arrows)
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
                if abs(sa-sb)<=sc<=sa+sb 
                    CG_tens=zeros(leg_dims...)
                    CG_tens[ra,rb,rc]=CG_tensor([sa,sb,sc],arrows)
                    push!(fusion_tens,CG_tens)
                end
                ic=rc[2]+1
            end
            ib=rb[2]+1
        end
        ia=ra[2]+1
    end

    return fusion_tens
end


"""
given a spin singlet basis T, its last leg "a" and a new spin rep (leg "b"), generate all possible basis S defined as
S[i]=T_{a...}*(CG[i]_{ab}^c)^{(i)}
leg "c" is some irrep depending on i
Notice that arrows[1] should be inverse of direction of leg "a" of T

return S,c_irrep
"""
function spin_basis_add_leg(spin_reps,arrows,T=[]) 
    c_irrep=[]
    S=[]
    ia=1

    for sa in spin_reps[1]
        ra=ia:ia+Int(2*sa)
        ib=1

        for sb in spin_reps[2]
            rb=ib:ib+Int(2*sb)

            for sc=abs(sa-sb):(sa+sb)

                CG=zeros(Int(2*spin_reps[1]+1),Int(2*spin_reps[2]+1),Int(2*sc+1))
                CG[ra,rb,:]=spin_fusion_tensor([sa,sb,sc],[arrows[1],arrows[2],1])

                if T==[]
                    push!(S,CG)
                else
                    push!(S,jcontract([T,CG],[[-1:-1:(-ndims(T)+1)...,1],[1,-ndims(T),-ndims(T)-1]]))
                end
            end
            ib=rb[2]+1
        end
        ia=ra[2]+1
    end
    return S,c_irrep
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

    for sc=0:max(spin_reps[1])+max(spin_reps[2])
        fusion_tens=spin_fusion_tensors([spin_reps[1],spin_reps[2],[sc]],[arrows[1],arrows[2],1])
        append!(M,fusion_tens)
        push!(c_irreps,[sc,size(fusion_tens,1)])
    end

    for legi=3:nlegs-1
        M_next=[]
        leg_irrep_next=[]
        for basei=1:size(M_last,1)
            S,c_irrep=spin_basis_add_leg([[leg_irrep[basei]],spin_reps[legi]],[-1,arrows[legi],1],M[basei])
            append!(M_next,S)
            append!(leg_irrep_next,c_irrep)
        end
        M=M_next
        leg_irrep=leg_irrep_next
    end
    println("singlet subspace dims:",size(M,1))
    M=reshape(hcat(M...),leg_dims...,size(M,1))
    return M
end
