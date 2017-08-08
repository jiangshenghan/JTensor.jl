
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
#TODO:constructor

function Base.size(heff::MPO_Heff)
    len=length(_tensor_list[2])
    return (len,len)
end

Base.eltype(heff::MPO_Heff)=heff._elemtype

Base.issymmetric(heff::MPO_Heff)=heff._issym

function Base.A_mul_B!(y::AbstractVector,heff:MPO_Heff,x::AbstractVector)
    #=TODO: 
    1. Use bicgstab in IterativeSolver.jl to obtain LB and RB
    =#
    LB=
    RB=
    res=jcontract([LB,Ar,T,Fr],[])*exp(-im*heff._p)
    copy!(y,res[:])
end


