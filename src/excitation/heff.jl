
"""
This file provides interface to obtain Heff for excited state
"""

"""
MPO as original Hamiltonian
We use the left gauge of B 
We assume normalized condition for MPO T, such that free energy equal to one
_tensor_list are ordered as (T,Al,Ar,C,Fl,Fr)
"""
type MPO_Heff
    _tensor_list
    _p
    _issym::Bool
    _elemtype::DataType
    function MPO_Heff(tensor_list,p,issym,elemtype)
        #Normalization of Fl and Fr, s.t. Fl.C.C*.Fr=1
        lr_norm=jcontract([tensor_list[2],tensor_list[4],conj(tensor_list[4]),tensor_list[3]],[[1,2,3],[1,4],[3,5],[4,2,5]])
        tensor_list_init=copy(tensor_list)
        tensor_list_init=tensor_list[3]/lr_norm
        return new(tensor_list_init,p,issym,elemtype)
    end
end
MPO_Heff(tensor_list,p;issym=false,elemtype=Complex128)=MPO_Heff(tensor_list,p,issym,elemtype)

function Base.size(heff::MPO_Heff)
    len=length(_tensor_list[2])
    return (len,len)
end

Base.size(heff::MPO_Heff,d)=d<=2?size(heff)[d]:1

Base.eltype(heff::MPO_Heff)=heff._elemtype

Base.issymmetric(heff::MPO_Heff)=heff._issym

function Base.A_mul_B!(y::AbstractVector,heff:MPO_Heff,x::AbstractVector)
    T,Al,Ar,Fl,Fr=heff._tensor_list
    B=reshape(x,size(Al))
    tol=1e-12
    LB=jcontract([Fl,B,T,conj(Al)],[[1,2,3],[1,-1,4],[2,-2,4,5],[3,-3,5]])
    @show vecdot(conj(Fr),LB)
    LB=LB-vecdot(conj(Fr),LB)*Fl
    LB=bicstabl(GeoSeries([LB,Ar,T,conj(Al)],[[1,2,3],[1,-1,4],[2,-2,4,5],[3,-3,5]],1,Fr,Fl,-heff._p),LB,tol=tol,log=true)
    RB=jcontract([Fr,B,T,conj(Ar)],[[1,2,3],[-1,1,4],[-2,2,4,5],[-3,3,5]])
    @show vecdot(conj(Fl,RB))
    RB=RB-vecdot(conj(Fl),RB)*Fr
    RB=bicstabl(GeoSeries([RB,Al,T,conj(Ar)],[[1,2,3],[-1,1,4],[-2,2,4,5],[-3,3,5]],1,Fl,Fr,heff._p),RB,tol=tol,log=true)

    res=jcontract([LB,Ar,T,Fr],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]])*exp(-im*heff._p)+jcontract([Fl,Al,T,RB],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]])*exp(im*heff._p)+jcontract([Fl,B,T,Fr],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]])
    copy!(y,res[:])
end


