include("../src/JTensor.jl")
using JTensor

chi_spin=[0,0,0,0,0.5,0.5,0.5,0.5]
chi=Int(sum(x->2x+1,chi_spin))
maxiter=500
println("chi=",chi)
println("chi spins: ",chi_spin)
println("maxiter=",maxiter)
println()
flush(STDOUT)


#pi srvb
T=[zeros(2,3,3,3,3) for i=1:2]
T[1][1,2,3,3,3]=1
T[1][2,1,3,3,3]=-1
T[1][1,3,1,3,3]=-1
T[1][2,3,2,3,3]=-1
T[1][1,3,3,1,3]=-1
T[1][2,3,3,2,3]=-1
T[1][1,3,3,3,2]=1
T[1][2,3,3,3,1]=-1

T[2][1,2,3,3,3]=1
T[2][2,1,3,3,3]=-1
T[2][1,3,1,3,3]=1
T[2][2,3,2,3,3]=1
T[2][1,3,3,1,3]=-1
T[2][2,3,3,2,3]=-1
T[2][1,3,3,3,2]=1
T[2][2,3,3,3,1]=-1

virt_spin=[0.5,0]


#initialize
D=size(T[1],2)
DD=D^2
TTu=[permutedims(reshape(jcontract([T[i],conj(T[i])],[[1,-1,-3,-5,-7],[1,-2,-4,-6,-8]]),DD,DD,DD,DD),[1,3,2,4]) for i=1:2]

#spin symmetric subspace
#MA=spin_sym_space([chi_spin,chi_spin,virt_spin,virt_spin],[1,-1,1,-1])
#MC=spin_sym_space([chi_spin,chi_spin],[1,-1])
MA=spin_singlet_space_from_cg([chi_spin,chi_spin,virt_spin,virt_spin],[1,-1,1,-1])
MC=spin_singlet_space_from_cg([chi_spin,chi_spin],[1,-1])
MA=reshape(MA,chi,chi,DD,size(MA)[end])

#symmetry transformation
#D=3 case
if D==3
    W=[0 1 0; -1 0 0; 0 0 1]
end
WW=reshape(jcontract([W,W],[[-1,-3],[-2,-4]]),DD,DD)
