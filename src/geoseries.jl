
#type GeoSeries
"""
GeoSeries provides an interface to obtain
(I+exp(ip)*M+exp(2ip)*M^2+...).b = (I-exp(ip)*M)^{-1}.b
with p momentum. We normalize matrix M, with radius one. So inverse here means pseudo inverse, which substract largest eigenvector. This equation holds when b is orthogonal to largest eigenvector

We obtain the result by solving linear equation
(I-exp(ip)*M).x=b
with M substract the dominant eigenvector space r⊗ l with tr(lr)=1

So A_mult_B! ~ x-exp(ip)*M.x+exp(ip)*(l.x)*r


To solve linear equation using function in IterativeSolvers, one needs function size and A_mul_B!
"""
type GeoSeries
    _tensor_list
    _legs_list
    _vecpos
    _l
    _r
    _p
end

function Base.size(gs::GeoSeries)
    insize=prod(size(gs._tensor_list[gs._vecpos]))
    outsize=1
    for (i,legs) in enumerate(gs._legs_list)
        outsize*=prod(size(gs._tensor_list[i])[find(x->x<0,legs)])
    end
    return (outsize,insize)
end

Base.size(gs::GeoSeries,d)=d<=2?size(gs)[d]:1

function Base.A_mul_B!(y::AbstractVector,gs:GeoSeries,x::AbstractVector)
    vec=reshape(x,size(gs._tensor_list[gs._vecpos]))
    gs._tensor_list[gs._vecpos]=vec
    res=vec-exp(im*gs._p)*jcontract(gs._tensor_list,gs._legs_list)+exp(im*gs._p)*vecdot(conj(gs._l),vec)*gs._r
    copy!(y,res[:])
end
