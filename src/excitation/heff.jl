
#This file provides interface to obtain Heff for excited state

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

function Base.A_mul_B!(y::AbstractVector,heff::MPO_Heff,x::AbstractVector)
    T,Al,Ar,C,Fl,Fr=heff._tensor_list
    B=reshape(x,size(Al))
    tol=1e-10
    n_mv=2000

    bl=jcontract([Fl,B,T,conj(Al)],[[1,2,3],[1,-1,4],[2,-2,4,5],[3,-3,5]])
    @show jcontract([bl,conj(C),Fr],[[1,2,3],[3,4],[1,2,4]])
    FlC=jcontract([Fl,C],[[1,-2,-3],[1,-1]])
    bl=bl-jcontract([bl,conj(C),Fr],[[1,2,3],[3,4],[1,2,4]])*FlC

    br=jcontract([Fr,B,T,conj(Ar)],[[1,2,3],[-1,1,4],[-2,2,4,5],[-3,3,5]])
    @show jcontract([br,conj(C),Fl],[[1,2,3],[3,4],[1,2,4]])
    FrC=jcontract([Fr,C],[[1,-2,-3],[1,-1]])
    br=br-jcontract([br,conj(C),Fl],[[1,2,3],[3,4],[1,2,4]])*FrC

    LB=IterativeSolvers.bicgstabl(GeoSeries([bl,Ar,T,conj(Al)],[[1,2,3],[1,-1,4],[2,-2,4,5],[3,-3,5]],1,jcontract([Fr,conj(C)],[[-1,-2,1],[-3,1]]),FlC,-heff._p),bl,tol=tol,log=true,max_mv_products=n_mv)
    RB=IterativeSolvers.bicgstabl(GeoSeries([br,Al,T,conj(Ar)],[[1,2,3],[-1,1,4],[-2,2,4,5],[-3,3,5]],1,jcontract([Fl,conj(C)],[[-1,-2,1],[1,-3]]),FrC,heff._p),br,tol=tol,log=true,max_mv_products=n_mv)

    res=exp(-im*heff._p)*jcontract([LB,Ar,T,Fr],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]])+exp(im*heff._p)*jcontract([Fl,Al,T,RB],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]])+jcontract([Fl,B,T,Fr],[[1,2,-1],[1,4,3],[2,5,3,-3],[4,5,-2]])
    copy!(y,res[:])
end


