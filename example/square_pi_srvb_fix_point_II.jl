
include("../src/JTensor.jl")
using JTensor

# #=
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
# =#

#=
#pi rvb D=6
T=readdlm("/home/jiangsb/code/JTensor.jl/tensor_data/square_pi_flux")
T=[T[:,1],T[:,2]]
T=[reshape(T[i],2,6,6,6,6) for i=1:2]
virt_spin=[0,0.5,1]
# =#

D=size(T[1],2)
DD=D^2
TTu=permutedims(reshape(jcontract([T[1],conj(T[1])],[[1,-1,-3,-5,-7],[1,-2,-4,-6,-8]]),DD,DD,DD,DD),[1,3,2,4])
TTd=permutedims(TTu,[2,1,4,3])

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


# #=
#random init MPS Alu,Aru
chi_spin=[0.0,0.0,0.0,0.0,0.5,0.5,0.5,0.5]
chi=Int(sum(x->2x+1,chi_spin))

#spin symmetric subspace
MA=spin_singlet_space_from_cg([chi_spin,chi_spin,virt_spin,virt_spin],[1,-1,1,-1])
MA=reshape(MA,chi,chi,DD,size(MA)[end])
MC=spin_singlet_space_from_cg([chi_spin,chi_spin],[1,-1])
MF=spin_singlet_space_from_cg([chi_spin,virt_spin,virt_spin,chi_spin],[-1,-1,1,1])
MF=reshape(MF,chi,DD,chi,size(MF)[end])

srand()
Alu=rand(Complex128,chi,chi,DD)
Aru=rand(Complex128,chi,chi,DD)
Alu=sym_tensor_proj(Alu,MA)
Aru=sym_tensor_proj(Aru,MA)

Acu=rand(Complex128,chi,chi,DD)
Acu=sym_tensor_proj(Acu,MA)
Cu=rand(Complex128,chi,chi)
Cu=sym_tensor_proj(Cu,MC)


Ald=rand(Complex128,chi,chi,DD)
Ard=rand(Complex128,chi,chi,DD)
Ald=sym_tensor_proj(Ald,MA)
Ard=sym_tensor_proj(Ard,MA)

Acd=rand(Complex128,chi,chi,DD)
Acd=sym_tensor_proj(Acd,MA)
Cd=rand(Complex128,chi,chi)
Cd=sym_tensor_proj(Cd,MC)
# =#


Fl=[]
Gl=[]
Fr=[]
Gr=[]


maxiter=300
@show maxiter
f0=0

λ0=[0,0,0,0]
μ0=[0,0,0,0]
nev=4
erru=errd=1
Jc=mapreduce(x->[1-4*mod(x,1) for i=1:2x+1],append!,chi_spin)
Jc=diagm(Jc)
for iter=1:maxiter
    Alu,Aru,Acu,Cu,Fl,Fr,_,erru,λ0=JTensor.sl_mag_trans_vumps_test(TTu,chi,Jc,Alu,Aru,Acu,Cu,Fl,Fr,e0=erru/10,maxiter=1,ncv=30,nev=nev,λ0=λ0)
    Ald,Ard,Acd,Cd,Gl,Gr,_,errd,λ0=JTensor.sl_mag_trans_vumps_test(TTd,chi,Jc,Ald,Ard,Acd,Cd,Gl,Gr,e0=errd/10,maxiter=1,ncv=30,nev=nev,λ0=μ0)

    Alu=sym_tensor_proj(Alu,MA)
    Aru=sym_tensor_proj(Aru,MA)
    Acu=sym_tensor_proj(Acu,MA)
    Cu=sym_tensor_proj(Cu,MC)
    @show svd_spin_sym_tensor(Cu,[1],[chi_spin,chi_spin],[1,-1])[[2,4]]

    Ald=sym_tensor_proj(Ald,MA)
    Ard=sym_tensor_proj(Ard,MA)
    Acd=sym_tensor_proj(Acd,MA)
    Cd=sym_tensor_proj(Cd,MC)
    @show svd_spin_sym_tensor(Cd,[1],[chi_spin,chi_spin],[1,-1])[[2,4]]

    #@show jcontract([Alu,conj(MA)],[[1,2,3],[1,2,3,-1]])
    #@show jcontract([Aru,conj(MA)],[[1,2,3],[1,2,3,-1]])

    #lower imps by symmetry
    #Ald=jcontract([Alu,WW],[[-1,-2,1],[1,-3]])

    @show iter
    square_heisenberg([Alu,Alu],[Ald,Ald],T)
    println()
    flush(STDOUT)

    if erru<1e-10 && errd<1e-10 break end

end


