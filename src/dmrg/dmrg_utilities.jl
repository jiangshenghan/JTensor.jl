###This file stores some useful function for finite size dmrg

"""
Turn MPS A to canonical form with center site at jc

A legs order: lrd

A[1] and A[L] are also three leg tensors, where the boundary leg has dim 1
"""
function turn_finite_mps_to_canonical!(A,jc=1)
    L=length(A)

    #tensors on left of jc become left canonical
    for j=1:jc-1
        Dl,Dr,d=size(A[j])
        svd_res=svdfact(reshape(permutedims(A[j],[1,3,2]),Dl*d,Dr),thin=true)
        Dr=length(svd_res[:S])
        A[j]=permutedims(reshape(svd_res[:U],Dl,d,Dr),[1,3,2])
        A[j+1]=jcontract([diagm(svd_res[:S])*svd_res[:Vt],A[j+1]],[[-1,1],[1,-2,-3]])
    end

    #tensors on right of jc become right canonical
    for j=L:-1:jc+1
        Dl,Dr,d=size(A[j])
        svd_res=svdfact(reshape(A[j],Dl,Dr*d),thin=true)
        Dl=length(svd_res[:S])
        A[j]=reshape(svd_res[:Vt],Dl,Dr,d)
        A[j-1]=jcontract([A[j-1],svd_res[:U]*diagm(svd_res[:S])],[[-1,1,-3],[1,-2]])
    end
end
