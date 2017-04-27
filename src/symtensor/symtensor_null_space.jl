
"""
find null space for spin symmetric tensor

    __
 \ /  \
--T----NT--- = 0
 / \__/ 

NT=oplus_{si}(NT^{si}), where si is spin quantum number
legs order of NT: right_legs...,null_leg
    
return NT,null_spin_reps
"""
function spin_sym_tensor_nullspace(T,left_legs,spin_reps,arrows;null_arrow=1)
    NT=[]
    null_spin_reps=[]

    #perform svd on T
    Us,Ss,Vts,_,spin_species=svd_spin_sym_tensor(T,left_legs,spin_reps,arrows,larrow=-null_arrow,thin=false)

    for i=1:length(spin_species)
        if Vts[i]==[] continue end
        if Ss[i]==[]
            NTi=permutedims(conj(Vts[i]),[2:ndims(Vts[i])...,1])
            append!(NT,NTi)
            nflavor=div(size(Vts[i],1),Int(2*spin_species[i]+1))
            append!(null_spin_reps,spin_species[i]*ones(nflavor))
            continue
        end
        indstart=sum(Ss[i].>max(size(Us[i],ndims(Us[i])),size(Vts[i],1))*maximum(Ss[i])*eps(eltype(Ss[i])))+1
        NTi=permutedims(conj(Vts[i][indstart:end,[Colon() for i=2:ndims(Vts[i])]]),[2:ndims(Vts[i])...,1])
        append!(NT,NTi)
        nflavor=div(size(Vts[i],1)-indstart+1,Int(2*spin_species[i]+1))
        append!(null_spin_reps,spin_species[i]*ones(nflavor))
    end
    return NT,null_spin_reps
end
