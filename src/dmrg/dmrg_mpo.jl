"""
Finite two-site DMRG algorithm with MPO Hamiltonian

---A[i-1]---A[i]---A[i+1]---
   |        |      |
---T[i-1]---T[i]---T[i+1]---   
   |        |      |


Fl[j] contains A[1],...,A[j-1] (also for T), define Fl[1] as one entry three leg tensor for convenient, so j=1,...,L 
Fr[j] contains A[j+1],...,A[L] (also for T), define Fr[L] as one entry three leg tensor for convenient, so j=1,...,L
(for two site algorithm, Fl[L] and Fr[1] will not be used)

For tensor T, legs order lrud
For tensor A, legs order lrd
For tenso rFl/Fr, legs order umd

A[1] and A[L] are also three leg tensors, where the boundary leg has dim 1
Similar for T[1] and T[L]

nsweep is total number of sweeps
χs is a nsweep-dim array of integer entries, which stores cutoff bond dim for each sweep

"""
function dmrg_mpo!(nsweep,χs,T,A=[];elemtype=Complex128)
    @show nsweep,χs

    L=length(T)
    d=size(T[1])[3]

    #init right canonical MPS A
    if A==[]
        D=div.(χs[1],2)
        push!(A,elemtype.(rand(1,D,d)))
        for j=2:L-1 push!(A,elemtype.(rand(D,D,d))) end
        push!(A,elemtype.(rand(D,1,d)))
    end
    turn_finite_mps_to_canonical!(A)

    #init Fl and Fr, which are used to generate effective Hamiltonian
    Fl=[]
    push!(Fl,ones(1,1,1))
    for j=2:L push!(Fl,jcontract([Fl[j-1],A[j-1],T[j-1],conj(A[j-1])],[[1,2,3],[1,-1,4],[2,-2,4,5],[3,-3,5]])) end

    Fr=copy(Fl)
    Fr[L]=ones(1,1,1)
    for j=L-1:-1:1 Fr[j]=jcontract([Fr[j+1],A[j+1],T[j+1],conj(A[j+1])],[[1,2,3],[-1,1,4],[-2,2,4,5],[-3,3,5]]) end

    #perform dmrg sweep
    for isweep=1:nsweep
        @show isweep,χs[isweep]
        sweep_dmrg!(A,T,Fl,Fr,χs[isweep],dir=1,elemtype=elemtype)
        sweep_dmrg!(A,T,Fl,Fr,χs[isweep],dir=-1,elemtype=elemtype)
    end

    #TODO: error analysis!
end
