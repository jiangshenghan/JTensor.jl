include("../src/JTensor.jl")
using JTensor

chi_spin=[0,0,0,0.5,0.5,0.5,1,1]
chi=Int(sum(x->2x+1,chi_spin))
maxiter=300
println("chi=",chi)
println("chi spins: ",chi_spin)
println("maxiter=",maxiter)
println()
flush(STDOUT)


#pi srvb
#T=[zeros(Complex128,2,3,3,3,3) for i=1:2]
#T[1][1,2,3,3,3]=1
#T[1][2,1,3,3,3]=-1
#T[1][1,3,1,3,3]=-1
#T[1][2,3,2,3,3]=-1
#T[1][1,3,3,1,3]=-1
#T[1][2,3,3,2,3]=-1
#T[1][1,3,3,3,2]=1
#T[1][2,3,3,3,1]=-1
#
#T[2][1,2,3,3,3]=1
#T[2][2,1,3,3,3]=-1
#T[2][1,3,1,3,3]=1
#T[2][2,3,2,3,3]=1
#T[2][1,3,3,1,3]=-1
#T[2][2,3,3,2,3]=-1
#T[2][1,3,3,3,2]=1
#T[2][2,3,3,3,1]=-1
#
#virt_spin=[0.5,0]

#pi rvb D=6
T=readdlm("/home/jiangsb/code/JTensor.jl/tensor_data/square_pi_flux")
T=[T[:,1],T[:,2]]
T=[reshape(T[i],2,6,6,6,6) for i=1:2]
virt_spin=[0,0.5,1]


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
#D=6(0+1/2+1) case
if D==6
    W=zeros(6,6)
    W[1,1]=W[2,3]=W[4,6]=W[6,4]=1
    W[3,2]=W[5,5]=-1
end
WW=reshape(jcontract([W,W],[[-1,-3],[-2,-4]]),DD,DD)

Alu=rand(Complex128,chi,chi,DD)
Alu=[Alu,Alu]
Aru=Alu
Acu=Alu
Cu=rand(Complex128,chi,chi)
Cu=[Cu,Cu]
err=1.

for iter=1:maxiter
    Alu,Aru,Acu,Cu,_,_,_,err=sl_mult_vumps_par(TTu,chi,Alu,Aru,Acu,Cu,e0=err/10,maxiter=1,ncv=20)
    Alu[1]=sym_tensor_proj(Alu[1],MA)
    Aru[1]=sym_tensor_proj(Aru[1],MA)
    Acu[1]=sym_tensor_proj(Acu[1],MA)
    Cu[1]=sym_tensor_proj(Cu[1],MC)
    Alu=[Alu[1],Alu[1]]
    Aru=[Aru[1],Aru[1]]
    Acu=[Acu[1],Acu[1]]
    Cu=[Cu[1],Cu[1]]

    #lower imps by symmetry
    Ald=[jcontract([Alu[i],WW],[[-1,-2,1],[1,-3]]) for i=1:2]

    @printf("iter=%d\n",iter)
    square_heisenberg(Alu,Ald,T)
    println()
end
