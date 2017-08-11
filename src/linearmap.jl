
# type LinearMap
"""
LinearMap provides an interface for the multiplication of a general tensor network to a "vector".
This is very useful when we try to compute eig problems for tensors. tensor_list and legs_list follows definition in jcontract. We require tensor_list to have the same element type. Notice that on the position of "vector", we also input a random tensor with the same dims and eltype as "vector"
"""

type LinearMap
    _tensor_list
    _legs_list
    _vecpos
    _issym::Bool
    _elemtype::DataType
end
LinearMap(tensor_list,legs_list,vecpos;issym=false,elemtype=Complex128)=LinearMap(tensor_list,legs_list,vecpos,issym,elemtype)

function Base.size(lm::LinearMap)
    insize=prod(size(lm._tensor_list[lm._vecpos]))
    outsize=1
    for (i,legs) in enumerate(lm._legs_list)
        outsize*=prod(size(lm._tensor_list[i])[find(x->x<0,legs)])
    end
    return (outsize,insize)
end

Base.size(lm::JTensor.LinearMap,d)=d<=2?size(lm)[d]:1

Base.eltype(lm::LinearMap)=lm._elemtype

Base.issymmetric(lm::LinearMap)=lm._issym

function Base.A_mul_B!(y::AbstractVector,lm::LinearMap,x::AbstractVector)
    lm._tensor_list[lm._vecpos]=reshape(x,size(lm._tensor_list[lm._vecpos]))
    copy!(y,reshape(jcontract(lm._tensor_list,lm._legs_list),size(lm)[1]))
end
