include("../src/JTensor.jl")
using JTensor

#running command
#julia square_pi_rvb_energy_sl_vumps.jl chi

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

#pi rvb D=6
T=readdlm("/home/jiangsb/code/JTensor.jl/tensor_data/square_pi_flux")
T=[T[:,1],T[:,2]]
T=[reshape(T[i],2,6,6,6,6) for i=1:2]


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
Alu=[reshape(jcontract([T[i],conj(T[i])],[[1,-1,2,-3,-5],[1,-2,2,-4,-6]]),DD,DD,DD) for i=1:2]
Bu=Cu=[]
Flu=Fru=jcontract([eye(Complex128,DD),eye(Complex128,DD)],[[-1,-3],[-2,-4]])
for iter=1:2
    Bu,Cu,Flu,Fru=sl_mult_mpo_mps(Alu,TTu,chi,Flu,Fru)
    Alu=[jcontract([diagm(Cu[3-i]),Bu[i]],[[-1,1],[1,-2,-3]]) for i=1:2]
    chi=size(Cu[1],1)
    Flu=Fru=jcontract([eye(Complex128,chi),eye(Complex128,DD)],[[-1,-3],[-2,-4]])
end
Cu=[complex(diagm(Cu[i])) for i=1:2]
Aru=[jcontract([Bu[i],Cu[i]],[[-1,1,-3],[1,-2]]) for i=1:2]
Acu=[jcontract([Alu[i],Cu[i]],[[-1,1,-3],[1,-2]]) for i=1:2]

#using radom intial state
#Alu=[permutedims(reshape(qr(rand(Complex128,chi*DD,chi))[1],chi,DD,chi),[1,3,2]) for i=1:2]
#Aru=[permutedims(Alu[i],[2,1,3]) for i=1:2]
#Acu=Cu=[]

Flu=Fru=[reshape(jcontract([eye(Complex128,chi),eye(Complex128,D)],[[-1,-4],[-2,-3]]),chi,DD,chi) for i=1:2]
err=1.

Gl=Gr=reshape(jcontract([eye(Complex128,chi,chi),eye(Complex128,D)],[[-1,-4],[-2,-3]]),chi,DD,chi)

#VUMPS to obtain energy
for iter=1:300
    Alu,Aru,Acu,Cu,Flu,Fru,_,err=sl_mult_vumps_par(TTu,chi,Alu,Aru,Acu,Cu,Flu,Fru,e0=err,maxiter=1,ncv=40)

    #lower imps by symmetry
    Ald=[jcontract([Alu[i],WW],[[-1,-2,1],[1,-3]]) for i=1:2]

    leftlm=LinearMap([Gl,Alu[1],TT[1],Ald[1],Alu[2],TT[2],Ald[2]],[[1,2,3],[1,6,4],[2,4,7,5],[8,3,5],[6,-1,9],[7,9,-2,10],[-3,8,10]],1)
    leig_res=eigs(leftlm,nev=1,v0=Gl[:],ncv=20,tol=1e-8)
    λl,Gl=leig_res
    λl=λl[1]
    Gl=reshape(Gl,chi,DD,chi)

    rightlm=LinearMap([Gr,Alu[2],TT[2],Ald[2],Alu[1],TT[1],Ald[1]],[[1,2,3],[6,1,4],[7,4,2,5],[3,8,5],[-1,6,9],[-2,9,7,10],[8,-3,10]],1)
    reig_res=eigs(rightlm,nev=1,v0=Gr[:],ncv=20,tol=1e-8)
    λr,Gr=reig_res
    λr=λr[1]
    Gr=reshape(Gr,chi,DD,chi)

    @printf("eig info: \n λl=%f+i%f \n λr=%f+i%f \n lniter=%d, lnmult=%d \n rniter=%d, rnmult=%d \n",real(λl),imag(λl),real(λr),imag(λr),leig_res[4],leig_res[5],reig_res[4],reig_res[5])

    wf_norm=λl*jcontract([Gl,Gr],[[1,2,3],[1,2,3]])
    EL=jcontract([Gl,Alu[1],THl,Ald[1]],[[1,2,3],[1,-1,4],[2,4,-2,5],[-3,3,5]])
    ER=jcontract([Gr,Alu[2],THr,Ald[2]],[[1,2,3],[-1,1,4],[-2,4,2,5],[3,-3,5]])
    energy=jcontract([EL,ER],[[1,2,3],[1,2,3]])/wf_norm

    @printf("iter=%d\n wf_norm=%f+i%f\n energy=%.16f+i%e\n\n",iter,real(wf_norm),imag(wf_norm),real(energy),imag(energy))
     flush(STDOUT)

     if iter%10==0
         Hlu=Hru=jcontract([eye(Complex128,chi),eye(Complex128,DD)],[[-1,-3],[-2,-4]])
         Bu,Cu=sl_mult_mpo_mps(Alu,TTu,chi,Hlu,Hru)
         Alu=[jcontract([diagm(Cu[3-i]),Bu[i]],[[-1,1],[1,-2,-3]]) for i=1:2]
         chi=size(Cu[1],1)
         Cu=[complex(diagm(Cu[i])) for i=1:2]
         Aru=[jcontract([Bu[i],Cu[i]],[[-1,1,-3],[1,-2]]) for i=1:2]
         Acu=[jcontract([Alu[i],Cu[i]],[[-1,1,-3],[1,-2]]) for i=1:2]
         Flu=Fru=[reshape(jcontract([eye(Complex128,chi),eye(Complex128,D)],[[-1,-4],[-2,-3]]),chi,DD,chi) for i=1:2]
         Gl=Gr=reshape(jcontract([eye(Complex128,chi,chi),eye(Complex128,D)],[[-1,-4],[-2,-3]]),chi,DD,chi)
     end
end



