
"""
This file provides interface to obtain Heff for excited state
"""

"""
MPO as original Hamiltonian
We use the left gauge of B and assume normalized condition for MPO T
_tensor_list are ordered as (T,Al,Ar,Fl,Fr)
"""
type MPO_Heff
    _tensor_list
    _p
    _issym::Bool
    _elemtype::DataType
end
MPO_Heff(tensor_list,p;issym=false,elemtype=Complex128)=MPO_Heff(tensor_list,p,issym,elemtype)

function Base.size(heff::MPO_Heff)
    len=length(_tensor_list[2])
    return (len,len)
end

Base.eltype(heff::MPO_Heff)=heff._elemtype

Base.issymmetric(heff::MPO_Heff)=heff._issym

function Base.A_mul_B!(y::AbstractVector,heff:MPO_Heff,x::AbstractVector)
    T,Al,Ar,Fl,Fr=heff._tensor_list
    B=reshape(x,size(Al))
    #TODO: Use bicgstab in IterativeSolver.jl to obtain LB and RB
    LB=jcontract([Fl,B,T,conj(Al)],[[1,2,3],[1,-1,4],[2,-2,4,5],[3,-3,5]])
    LB=
    RB=jcontract([Fr,B,T,conj(Ar)],[[1,2,3],[-1,1,4],[-2,2,4,5],[-3,3,5]])
    RB=
    res=jcontract([LB,Ar,T,Fr],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]])*exp(-im*heff._p)+jcontract([Fl,Al,T,RB],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]])*exp(im*heff._p)+jcontract([Fl,B,T,Fr],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]])
    copy!(y,res[:])
end


