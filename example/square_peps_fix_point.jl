
include("../src/JTensor.jl")
using JTensor

chi_spin=[0,0,0,0,0.5,0.5,0.5,0.5]
chi=Int(sum(x->2x+1,chi_spin))
maxiter=100

Jc=mapreduce(x->[1-4*mod(x,1) for i=1:2x+1],append!,chi_spin)
Jc=diagm(Jc)

@show chi
@show chi_spin
@show maxiter
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


D=size(T[1],2)
DD=D^2
TTu=[permutedims(reshape(jcontract([T[i],conj(T[i])],[[1,-1,-3,-5,-7],[1,-2,-4,-6,-8]]),DD,DD,DD,DD),[1,3,2,4]) for i=1:2]


#spin symmetric subspace
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

#init MPS
srand()
Alu=rand(Complex128,chi,chi,DD)
Aru=rand(Complex128,chi,chi,DD)
Acu=rand(Complex128,chi,chi,DD)
Cu=rand(Complex128,chi,chi)

Alu=sym_tensor_proj(Alu,MA)
Aru=sym_tensor_proj(Aru,MA)
Acu=sym_tensor_proj(Acu,MA)
Cu=sym_tensor_proj(Cu,MC)

Fl=[]
Fr=[]

err=1e-12

for iter=1:maxiter
    Alu,Aru,Acu,Cu,Fl,Fr=sl_mag_trans_vumps(TTu[1],chi,Jc,Alu,Aru,Acu,Cu,Fl,Fr,e0=err/10,maxiter=1,ncv=20)
    Alu=sym_tensor_proj(Alu,MA)
    Aru=sym_tensor_proj(Aru,MA)
    Acu=sym_tensor_proj(Acu,MA)
    Cu=sym_tensor_proj(Cu,MC)

    #lower imps by symmetry
    Ald=jcontract([Alu,WW],[[-1,-2,1],[1,-3]])

    @show iter
    square_heisenberg([Alu,Alu],[Ald,Ald],T)
    println()
    flush(STDOUT)
end

