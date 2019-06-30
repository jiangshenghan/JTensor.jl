
#type MultLinearMap
"""
MultLinearMap provides an interface for the multiplication of multiple general tensor networks to a "vector":
(a*A+b*B+c*C+...).v
where a,b,c,... are scalars and A,B,C,... are linearmaps

tensor_lists and legs_lists follows definition in jcontract. 
Notice that on the position of "vector", we also input a random tensor with the same dims and eltype as "vector"
"""

type MultLinearMap
    _len
    _tensor_lists
    _legs_lists
    _vecpositions
    _scalar_lists
    _issym::Bool
    _elemtype::DataType
end
MultLinearMap(tensor_lists,legs_lists,vecpositions;scalar_lists=[],issym=false,elemtype=Complex128)=MultLinearMap(length(tensor_lists),tensor_lists,legs_lists,vecpositions,scalar_lists,issym,elemtype)

function Base.size(mlm::MultLinearMap)
    insize=prod(size(mlm._tensor_lists[1][mlm._vecpositions[1]]))
    outsize=1
    for (i,legs) in enumerate(mlm._legs_lists[1])
        outsize*=prod(size(mlm._tensor_lists[1][i])[find(x->x<0,legs)])
    end
    return (outsize,insize)
end

function Base.size(mlm::MultLinearMap,n)
    return size(mlm)[n]
end

Base.eltype(mlm::MultLinearMap)=mlm._elemtype

Base.issymmetric(mlm::MultLinearMap)=mlm._issym

#TODO:check scalar_lists
function Base.A_mul_B!(y::AbstractVector,mlm::MultLinearMap,x::AbstractVector)
    res=zeros(mlm._elemtype,size(mlm,1))
    scalar_lists=[]
    if length(mlm._scalar_lists)==0
        scalar_lists=ones(mlm._len)
    else
        append!(scalar_lists,mlm._scalar_lists)
    end
    for i=1:mlm._len
        mlm._tensor_lists[i][mlm._vecpositions[i]]=reshape(x,size(mlm._tensor_lists[i][mlm._vecpositions[i]]))
        res+=scalar_lists[i]*jcontract(mlm._tensor_lists[i],mlm._legs_lists[i])[:]
    end
    copy!(y,res)
end

