
"""
find null space for spin symmetric tensor

    __
 \ /  \
--T----N--- = 0
 / \__/ 

NT=oplus_{si}(NT^{si}), where si is spin quantum number
    
return NT,null_spin_reps
"""
function spin_sym_tensor_nullspace(T,left_legs,spin_reps,arrows;larrow=-1)
    m=prod(size(T,left_legs...))
    n=div(length(T),m)

    #perform svd on T
    Us,Ss,Vts,vals_spin_rep=svd_spin_sym_tensor(T,left_legs,spin_reps,arrows,larrow=larrow,thin=false)

    null_spin_reps=

    for i=1:size(null_spin_rep,1)
        indstart=sum(Ss[i].>max(m,n)*maximum(Ss[i])*eps(eltype(Ss[i])))+1
        Ni=permutedims(conj(Vts[i][indstart:end,[Colon() for i=2:ndims(Vts[i])]]),[2:ndims(Vts[i])...,1])
        push!(Ns,Ni)
    end
    return Ns,null_spin_rep
end
