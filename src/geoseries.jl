
#type GeoSeries
"""
GeoSeries provides an interface to obtain
(I+exp(ip)*M+exp(2ip)*M^2+...).b = (I-exp(ip)*M)^{-1}.b
with p momentum. 

Matrix M is normalized with largest eigenvalue one. Here inverse means pseudo inverse, which substract largest eigenvector space. This equation holds when b is orthogonal to largest eigenvector

We obtain the result by solving linear equation (using function such as bicgstabl)
(I-exp(ip)*M).x=b
with M substract the dominant eigenvector space r⊗ l with tr(lr)=1. x is always multiplied on the right

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
    len=length(gs._tensor_list[gs._vecpos])
    return (len,len)
end

Base.size(gs::GeoSeries,d)=d<=2?size(gs)[d]:1

function Base.A_mul_B!(y::AbstractVector,gs::GeoSeries,x::AbstractVector)
    vec=reshape(x,size(gs._tensor_list[gs._vecpos]))
    gs._tensor_list[gs._vecpos]=vec
    res=vec-exp(im*gs._p)*jcontract(gs._tensor_list,gs._legs_list)+exp(im*gs._p)*jcontract([gs._l,vec],[[1:ndims(vec)...],[1:ndims(vec)...]])*gs._r
    copy!(y,res[:])
end
