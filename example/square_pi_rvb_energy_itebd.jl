include("../src/JTensor.jl")
using JTensor

#zero srvb
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

d,D=size(T[1],1,2)

HS=zeros(Complex128,d,d,3)
HS[:,:,1]=0.5*[0 1; 1 0]
HS[:,:,2]=0.5*[0 -im; im 0]
HS[:,:,3]=0.5*[1 0; 0 -1]

Au=[reshape(jcontract([T[i],conj(T[i])],[[1,-1,2,-3,-5],[1,-2,2,-4,-6]]),D^2,D^2,D,D) for i=1:2]
Ad=[reshape(jcontract([T[i],conj(T[i])],[[1,-3,-5,-1,2],[1,-4,-6,-2,2]]),D^2,D^2,D,D) for i=1:2]
Cu=Cd=[eye(D^2) for i=1:2]
Tu=[permutedims(T[i],[1,2,4,3,5]) for i=1:2]
Td=[permutedims(T[i],[1,4,2,5,3]) for i=1:2]
Flu=Fru=Fld=Frd=jcontract([eye(Complex128,D^2),eye(Complex128,D),eye(Complex128,D)],[[-1,-4],[-2,-5],[-3,-6]])
chiu=chid=10

Gl=Gr=jcontract([eye(Complex128,chiu,chid),eye(Complex128,D)],[[-1,-4],[-2,-3]])

for iter=1:5
    Bu,Cu,Flu,Fru=dl_mult_mpo_mps(Au,Tu,chiu,Flu,Fru)
    Bd,Cd,Fld,Frd=dl_mult_mpo_mps(Ad,Td,chid,Fld,Frd)
    Au=[jcontract([diagm(Cu[3-i]),Bu[i]],[[-1,1],[1,-2,-3,-4]]) for i=1:2]
    Ad=[jcontract([diagm(Cd[3-i]),Bd[i]],[[-1,1],[1,-2,-3,-4]]) for i=1:2]

    if (chiu!=size(Au[1],1) || chid!=size(Ad[1],1) || iter==1)
        chiu,chid=size(Au[1],1),size(Ad[1],1)
        Flu=Fru=jcontract([eye(Complex128,chiu),eye(Complex128,D),eye(Complex128,D)],[[-1,-4],[-2,-5],[-3,-6]])
        Fld=Frd=jcontract([eye(Complex128,chid),eye(Complex128,D),eye(Complex128,D)],[[-1,-4],[-2,-5],[-3,-6]])
        Gl=Gr=jcontract([eye(Complex128,chiu,chid),eye(Complex128,D)],[[-1,-4],[-2,-3]])
    end

    leftlm=LinearMap([Gl,Au[1],T[1],conj(T[1]),Ad[1],Au[2],T[2],conj(T[2]),Ad[2]],[[1,2,3,4],[1,10,5,6],[7,2,5,11,8],[7,3,6,12,9],[13,4,8,9],[10,-1,14,15],[16,11,14,-2,17],[16,12,15,-3,18],[-4,13,17,18]],1)
    leig_res=eigs(leftlm,nev=1,v0=Gl[:],ncv=20,tol=1e-8)
    λl,Gl=leig_res
    λl=λl[1]
    Gl=reshape(Gl,chiu,D,D,chid)

    rightlm=LinearMap([Gr,Au[2],T[2],conj(T[2]),Ad[2],Au[1],T[1],conj(T[1]),Ad[1]],[[1,2,3,4],[10,1,5,6],[7,11,5,2,8],[7,12,6,3,9],[4,13,8,9],[-1,10,14,15],[16,-2,14,11,17],[16,-3,15,12,18],[13,-4,17,18]],1)
    reig_res=eigs(rightlm,nev=1,v0=Gr[:],ncv=20,tol=1e-8)
    λr,Gr=reig_res
    λr=λr[1]
    Gr=reshape(Gr,chiu,D,D,chid)

    @printf("eig info: \n λl=%f+i%f \n λr=%f+i%f \n lniter=%d, lnmult=%d \n rniter=%d, rnmult=%d \n",real(λl),imag(λl),real(λr),imag(λr),leig_res[4],leig_res[5],reig_res[4],reig_res[5])

    wf_norm=λl*jcontract([Gl,Gr],[[1,2,3,4],[1,2,3,4]])
    EL=jcontract([Gl,Au[1],T[1],HS,conj(T[1]),Ad[1]],[[1,2,3,4],[1,-1,5,6],[7,2,5,-2,9],[7,8,-3],[8,3,6,-4,10],[-5,4,9,10]])
    ER=jcontract([Gr,Au[2],T[2],HS,conj(T[2]),Ad[2]],[[1,2,3,4],[-1,1,5,6],[7,-2,5,2,9],[7,8,-3],[8,-4,6,3,10],[4,-5,9,10]])
    energy=jcontract([EL,ER],[[1,2,3,4,5],[1,2,3,4,5]])/wf_norm

    @printf("iter=%d\n wf_norm=%f+i%f\n energy=%.16f+i%e\n\n",iter,real(wf_norm),imag(wf_norm),real(energy),imag(energy))
     flush(STDOUT)
end

