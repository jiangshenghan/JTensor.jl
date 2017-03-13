include("../src/JTensor.jl")
using JTensor

#running command
#julia square_pi_rvb_energy_sl_vumps.jl chi

#pi srvb
T=[zeros(Complex128,2,3,3,3,3) for i=1:2]
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

#pi rvb D=6
#T=readdlm("/home/jiangsb/code/JTensor.jl/tensor_data/square_pi_flux")
#T=[T[:,1],T[:,2]]
#T=[reshape(T[i],2,6,6,6,6) for i=1:2]


D=size(T[1],2)
DD=D^2
TT=[reshape(jcontract([T[i],conj(T[i])],[[1,-1,-3,-5,-7],[1,-2,-4,-6,-8]]),DD,DD,DD,DD) for i=1:2]

HS=zeros(Complex128,2,2,3)
HS[:,:,1]=0.5*[0 1; 1 0]
HS[:,:,2]=0.5*[0 -im; im 0]
HS[:,:,3]=0.5*[1 0; 0 -1]

THl=reshape(jcontract([T[1],HS,conj(T[1])],[[1,-1,-3,-5,-8],[1,2,-6],[2,-2,-4,-7,-9]]),DD,DD,DD*3,DD)
THr=reshape(jcontract([T[2],HS,conj(T[2])],[[1,-1,-4,-6,-8],[1,2,-2],[2,-3,-5,-7,-9]]),DD*3,DD,DD,DD)

TTu=[permutedims(TT[i],[1,3,2,4]) for i=1:2]
TTd=[permutedims(TT[i],[3,1,4,2]) for i=1:2]

chi=chi=parse(Int,ARGS[1])

#symmetry transformation
#D=3 case
if D==3
    W=[0 1 0; -1 0 0; 0 0 1]
end
#D=6(0+1/2+1) case
if D==6
    W=zeros(6,6)
    W[6,1]=W[3,3]=W[2,5]=W[1,6]=1
    W[5,2]=W[4,4]=-1
end
WW=reshape(jcontract([W,W],[[-1,-3],[-2,-4]]),DD,DD)

#using iTEBD to get initial state
#Alu=[reshape(jcontract([T[i],conj(T[i])],[[1,-1,2,-3,-5],[1,-2,2,-4,-6]]),DD,DD,DD) for i=1:2]
#Alu=[reshape(jcontract([eye(Complex128,chi),eye(Complex128,D)],[[-1,-2],[-3,-4]]),chi,chi,DD) for i=1:2]
Alu=[reshape(eye(Complex128,D),1,1,DD) for i=1:2]
Bu=Cu=[]
Flu=Fru=jcontract([eye(Complex128,size(Alu[1],1)),eye(Complex128,DD)],[[-1,-3],[-2,-4]])
for iter=1:3
    Bu,Cu,Flu,Fru=sl_mult_mpo_mps(Alu,TTu,chi,Flu,Fru)
    Alu=[jcontract([diagm(Cu[3-i]),Bu[i]],[[-1,1],[1,-2,-3]]) for i=1:2]
    Flu=Fru=jcontract([eye(Complex128,size(Cu[1],1)),eye(Complex128,DD)],[[-1,-3],[-2,-4]])
end
chi=size(Cu[1],1)
Cu=[complex(diagm(Cu[i])) for i=1:2]
Aru=[jcontract([Bu[i],Cu[i]],[[-1,1,-3],[1,-2]]) for i=1:2]
Acu=[jcontract([Alu[i],Cu[i]],[[-1,1,-3],[1,-2]]) for i=1:2]

#using radom intial state
#Alu=[permutedims(reshape(qr(rand(Complex128,chi*DD,chi))[1],chi,DD,chi),[1,3,2]) for i=1:2]
#Aru=[permutedims(Alu[i],[2,1,3]) for i=1:2]
#Acu=Cu=[]

Flu=Fru=[reshape(jcontract([eye(Complex128,chi),eye(Complex128,D)],[[-1,-4],[-2,-3]]),chi,DD,chi) for i=1:2]
err=1.

#VUMPS to obtain energy
for iter=1:300
    Alu,Aru,Acu,Cu,Flu,Fru,_,err=sl_mult_vumps_par(TTu,chi,Alu,Aru,Acu,Cu,Flu,Fru,e0=err,maxiter=1,ncv=40)

    #lower imps by symmetry
    Ald=[jcontract([Alu[i],WW],[[-1,-2,1],[1,-3]]) for i=1:2]

    @printf("iter=%d\n",iter)
    square_heisenberg(Aru,Ald,T)
end



