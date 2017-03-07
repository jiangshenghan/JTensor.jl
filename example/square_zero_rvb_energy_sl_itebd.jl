include("../src/JTensor.jl")
using JTensor

#zero srvb
T=zeros(2,3,3,3,3)
T[1,2,3,3,3]=1
T[2,1,3,3,3]=-1
T[1,3,1,3,3]=-1
T[2,3,2,3,3]=-1
T[1,3,3,1,3]=-1
T[2,3,3,2,3]=-1
T[1,3,3,3,2]=1
T[2,3,3,3,1]=-1

D=size(T,2)
DD=D^2
TT=reshape(jcontract([T,conj(T)],[[1,-1,-3,-5,-7],[1,-2,-4,-6,-8]]),DD,DD,DD,DD)


HS=zeros(Complex128,2,2,3)
HS[:,:,1]=0.5*[0 1; 1 0]
HS[:,:,2]=0.5*[0 -im; im 0]
HS[:,:,3]=0.5*[1 0; 0 -1]

THl=reshape(jcontract([T,HS,conj(T)],[[1,-1,-3,-5,-8],[1,2,-6],[2,-2,-4,-7,-9]]),DD,DD,DD*3,DD)
THr=reshape(jcontract([T,HS,conj(T)],[[1,-1,-4,-6,-8],[1,2,-2],[2,-3,-5,-7,-9]]),DD*3,DD,DD,DD)

Au=reshape(jcontract([T,conj(T)],[[1,-1,2,-3,-5],[1,-2,2,-4,-6]]),DD,DD,DD)
Ad=reshape(jcontract([T,conj(T)],[[1,-3,-5,-1,2],[1,-4,-6,-2,2]]),DD,DD,DD)
Cu=Cd=eye(DD)
TTu=permutedims(TT,[1,3,2,4])
TTd=permutedims(TT,[3,1,4,2])
Flu=Fru=Fld=Frd=jcontract([eye(Complex128,DD),eye(Complex128,DD)],[[-1,-3],[-2,-4]])
chiu=chid=10

Gl=Gr=reshape(jcontract([eye(Complex128,chiu,chid),eye(Complex128,D)],[[-1,-4],[-2,-3]]),chiu,DD,chid)

for iter=1:5
    Bu,Cu,Flu,Fru=sl_mult_mpo_mps([Au],[TTu],chiu,Flu,Fru)
    Bd,Cd,Fld,Frd=sl_mult_mpo_mps([Ad],[TTd],chid,Fld,Frd)
    Au=jcontract([diagm(Cu[1]),Bu[1]],[[-1,1],[1,-2,-3]])
    Ad=jcontract([diagm(Cd[1]),Bd[1]],[[-1,1],[1,-2,-3]])

    if (chiu!=size(Au,1) || chid!=size(Ad,1) || iter==1)
        chiu,chid=size(Au,1),size(Ad,1)
        Flu=Fru=jcontract([eye(Complex128,chiu),eye(Complex128,DD)],[[-1,-3],[-2,-4]])
        Fld=Frd=jcontract([eye(Complex128,chid),eye(Complex128,DD)],[[-1,-3],[-2,-4]])
        Gl=Gr=reshape(jcontract([eye(Complex128,chiu,chid),eye(Complex128,D)],[[-1,-4],[-2,-3]]),chiu,DD,chid)
    end

    leftlm=LinearMap([Gl,Au,TT,Ad],[[1,2,3],[1,-1,4],[2,4,-2,5],[-3,3,5]],1)
    leig_res=eigs(leftlm,nev=1,v0=Gl[:],ncv=20,tol=1e-8)
    λl,Gl=leig_res
    λl=λl[1]
    Gl=reshape(Gl,chiu,DD,chid)

    rightlm=LinearMap([Gr,Au,TT,Ad],[[1,2,3],[-1,1,4],[-2,4,2,5],[3,-3,5]],1)
    reig_res=eigs(rightlm,nev=1,v0=Gr[:],ncv=20,tol=1e-8)
    λr,Gr=reig_res
    λr=λr[1]
    Gr=reshape(Gr,chiu,DD,chid)

    @printf("eig info: \n λl=%f+i%f \n λr=%f+i%f \n lniter=%d, lnmult=%d \n rniter=%d, rnmult=%d \n",real(λl),imag(λl),real(λr),imag(λr),leig_res[4],leig_res[5],reig_res[4],reig_res[5])

    wf_norm=λl*λr*jcontract([Gl,Gr],[[1,2,3],[1,2,3]])
    EL=jcontract([Gl,Au,THl,Ad],[[1,2,3],[1,-1,4],[2,4,-2,5],[-3,3,5]])
    ER=jcontract([Gr,Au,THr,Ad],[[1,2,3],[-1,1,4],[-2,4,2,5],[3,-3,5]])
    energy=jcontract([EL,ER],[[1,2,3],[1,2,3]])/wf_norm

    @printf("iter=%d\n wf_norm=%f+i%f\n energy=%.16f+i%e\n\n",iter,real(wf_norm),imag(wf_norm),real(energy),imag(energy))
     flush(STDOUT)
end

