
include("../src/JTensor.jl")
using JTensor

elemtype=Complex128

I=[1. 0; 0 1.]
X=[0 1.; 1. 0]
Y=[0 -im*1.; im*1. 0]
Z=[1. 0; 0 -1.]

Sp=[0 1; 0 0]
Sm=[0 0; 1 0]
Sz=0.5*Z

#Ising model
#=
dp=2
h=1
J=1
@show h,J
DH=3
Wh=zeros(Complex128,DH,DH,dp,dp)
Wh[1,1,:,:]=I
Wh[2,1,:,:]=Z
Wh[3,1,:,:]=-h*X
Wh[3,2,:,:]=-J*Z
Wh[3,3,:,:]=I
# =#


#spin-1/2 Heisenberg model
#=
dp=2
J=1
Δ=1
@show J,Δ
DH=5
Wh=zeros(DH,DH,dp,dp)
Wh[1,1,:,:]=I
Wh[2,1,:,:]=Sm
Wh[3,1,:,:]=Sp
Wh[4,1,:,:]=Sz
Wh[5,1,:,:]=0
Wh[5,2,:,:]=J/2*Sp
Wh[5,3,:,:]=J/2*Sm
Wh[5,4,:,:]=Δ*Sz
Wh[5,5,:,:]=I
# =#

#triality model
##=
dp=4
J11=1
J12=1
J13=1
J2=-1
J31=J32=J33=0.1
@show J11,J12,J13
@show J2
@show J31,J32,J33
DH=6
Wh=zeros(Complex128,DH,DH,dp,dp)
Wh[1,1,:,:]=eye(dp)
Wh[2,1,:,:]=J11*kron(Z,I)+J31*kron(Y,X)
Wh[3,1,:,:]=kron(I,Z)
Wh[4,1,:,:]=kron(Z,Y)
Wh[5,1,:,:]=kron(Z,Z)
Wh[6,1,:,:]=J12*kron(X,X)
Wh[6,2,:,:]=kron(Z,I)
Wh[6,3,:,:]=J13*kron(X,Z)-J32*kron(I,Y)
Wh[6,4,:,:]=J2*kron(Z,Z)
Wh[6,5,:,:]=-J33*kron(Y,Z)
Wh[6,6,:,:]=eye(dp)
# =#

#perform vumps
chis=[5,6,7,8,9,10,12,14,17,20,23,27,31]
#chis=[8]

Ns=2
@show chis
for (i,chi) in enumerate(chis)
    @show i,chi

    Ainit=rand(Complex128,chi,chi,dp)
    Al=[Ainit for i=1:Ns]
    Ar=[Ainit for i=1:Ns]
    C=[]

    Al,Ar,C,errp=mult_site_ham_vumps_par([Wh for i=1:Ns],chi,Al,Ar,C,maxiter=200,err0=1e-8)
    #obtain correlation length
    Allm=LinearMap([rand(elemtype,chi,chi),Al[1],conj(Al[1]),Al[2],conj(Al[2])],[[1,2],[1,4,3],[2,5,3],[4,-1,6],[5,-2,6]],1,elemtype=elemtype)
    λl,_=eigs(Allm,nev=2)
    xi=Ns*1/log(abs(λl[1]/λl[2]))
    @show λl,xi
end

