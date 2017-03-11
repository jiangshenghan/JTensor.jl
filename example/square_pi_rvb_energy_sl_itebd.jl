include("../src/JTensor.jl")
using JTensor

#running command
#julia square_pi_rvb_energy_sl_itebd.jl chi

#pi srvb
#T=[zeros(2,3,3,3,3) for i=1:2]
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
T=readdlm("/home/jiangsb/code/JTensor.jl/tensor_data/square_pi_flux_II")
T=[T[:,1],T[:,2]]
T=[reshape(T[i],2,6,6,6,6) for i=1:2]

#zero rvb D=6, but measure using two sites per uc
#T=readdlm("/home/jiangsb/code/JTensor.jl/tensor_data/square_zero_flux")
#T=reshape(T,2,6,6,6,6)
#T=[T,T]


D=size(T[1],2)
DD=D^2
TT=[reshape(jcontract([T[i],conj(T[i])],[[1,-1,-3,-5,-7],[1,-2,-4,-6,-8]]),DD,DD,DD,DD) for i=1:2]

HS=zeros(Complex128,2,2,3)
HS[:,:,1]=0.5*[0 1; 1 0]
HS[:,:,2]=0.5*[0 -im; im 0]
HS[:,:,3]=0.5*[1 0; 0 -1]

THl=reshape(jcontract([T[1],HS,conj(T[1])],[[1,-1,-3,-5,-8],[1,2,-6],[2,-2,-4,-7,-9]]),DD,DD,DD*3,DD)
THr=reshape(jcontract([T[2],HS,conj(T[2])],[[1,-1,-4,-6,-8],[1,2,-2],[2,-3,-5,-7,-9]]),DD*3,DD,DD,DD)

Au=[reshape(jcontract([T[i],conj(T[i])],[[1,-1,2,-3,-5],[1,-2,2,-4,-6]]),DD,DD,DD) for i=1:2]
Ad=[reshape(jcontract([T[i],conj(T[i])],[[1,-3,-5,-1,2],[1,-4,-6,-2,2]]),DD,DD,DD) for i=1:2]
Cu=Cd=[eye(DD) for i=1:2]
TTu=[permutedims(TT[i],[1,3,2,4]) for i=1:2]
TTd=[permutedims(TT[i],[3,1,4,2]) for i=1:2]
Flu=Fru=Fld=Frd=jcontract([eye(Complex128,DD),eye(Complex128,DD)],[[-1,-3],[-2,-4]])
chiu=chid=parse(Int,ARGS[1])

Gl=Gr=reshape(jcontract([eye(Complex128,chiu,chid),eye(Complex128,D)],[[-1,-4],[-2,-3]]),chiu,DD,chid)

for iter=1:100
    Bu,Cu,Flu,Fru=sl_mult_mpo_mps(Au,TTu,chiu,Flu,Fru)
    #Bd,Cd,Fld,Frd=sl_mult_mpo_mps(Ad,TTd,chid,Fld,Frd)
    Au=[jcontract([diagm(Cu[3-i]),Bu[i]],[[-1,1],[1,-2,-3]]) for i=1:2]
    #Ad=[jcontract([diagm(Cd[3-i]),Bd[i]],[[-1,1],[1,-2,-3]]) for i=1:2]

    #generate the fixed point tensor from lower half-plane by symmetry, where we assume the symmetry transforms trivially
    println("symmetry!")
    println()
    #D=3 case
    #if D==3
    #    W=[0 1 0; -1 0 0; 0 0 1]
    #end
    #D=6(0+1/2+1) case
    if D==6
        W=zeros(6,6)
        W[6,1]=W[3,3]=W[2,5]=W[1,6]=1
        W[5,2]=W[4,4]=-1
    end
    WW=reshape(jcontract([W,W],[[-1,-3],[-2,-4]]),DD,DD)
    #for pi flux rvb
    Ad=[jcontract([Au[i],WW],[[-1,-2,1],[1,-3]]) for i=1:2]
    Cd=[Cu[i] for i=1:2]

    if (chiu!=size(Au[1],1) || chid!=size(Ad[1],1) || iter==1)
        chiu,chid=size(Au[1],1),size(Ad[1],1)
        Flu=Fru=jcontract([eye(Complex128,chiu),eye(Complex128,DD)],[[-1,-3],[-2,-4]])
        Fld=Frd=jcontract([eye(Complex128,chid),eye(Complex128,DD)],[[-1,-3],[-2,-4]])
        Gl=Gr=reshape(jcontract([eye(Complex128,chiu,chid),eye(Complex128,D)],[[-1,-4],[-2,-3]]),chiu,DD,chid)
    end

    leftlm=LinearMap([Gl,Au[1],TT[1],Ad[1],Au[2],TT[2],Ad[2]],[[1,2,3],[1,6,4],[2,4,7,5],[8,3,5],[6,-1,9],[7,9,-2,10],[-3,8,10]],1)
    leig_res=eigs(leftlm,nev=1,v0=Gl[:],ncv=20,tol=1e-8)
    λl,Gl=leig_res
    λl=λl[1]
    Gl=reshape(Gl,chiu,DD,chid)

    rightlm=LinearMap([Gr,Au[2],TT[2],Ad[2],Au[1],TT[1],Ad[1]],[[1,2,3],[6,1,4],[7,4,2,5],[3,8,5],[-1,6,9],[-2,9,7,10],[8,-3,10]],1)
    reig_res=eigs(rightlm,nev=1,v0=Gr[:],ncv=20,tol=1e-8)
    λr,Gr=reig_res
    λr=λr[1]
    Gr=reshape(Gr,chiu,DD,chid)

    @printf("eig info: \n λl=%f+i%f \n λr=%f+i%f \n lniter=%d, lnmult=%d \n rniter=%d, rnmult=%d \n",real(λl),imag(λl),real(λr),imag(λr),leig_res[4],leig_res[5],reig_res[4],reig_res[5])

    wf_norm=λl*jcontract([Gl,Gr],[[1,2,3],[1,2,3]])
    EL=jcontract([Gl,Au[1],THl,Ad[1]],[[1,2,3],[1,-1,4],[2,4,-2,5],[-3,3,5]])
    ER=jcontract([Gr,Au[2],THr,Ad[2]],[[1,2,3],[-1,1,4],[-2,4,2,5],[3,-3,5]])
    energy=jcontract([EL,ER],[[1,2,3],[1,2,3]])/wf_norm

    @printf("iter=%d\n wf_norm=%f+i%f\n energy=%.16f+i%e\n\n",iter,real(wf_norm),imag(wf_norm),real(energy),imag(energy))
     flush(STDOUT)
end


