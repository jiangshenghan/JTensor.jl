
include("../src/JTensor.jl")
using JTensor

L=10
dp=4
g1=0.7
g2=1
g3=1
h=0.6
δh=0.2
g4=0.3
g5=0.3
g6=0.3
@show L,dp,g1,g2,g3,h,δh,g4,g5,g6

I=[1 0; 0 1]
X=[0 1; 1 0]
Y=[0 -im; im 0]
Z=[1 0; 0 -1]

#=
#construct MPO for H=\sum_j (g1 Z1.Z1' + g2 X1.X2 + g3 X1.Z2.Z2') - h Z1.Z2.Z1'.Y2' + (g4 Z1.Y1'.X2' - g5 Y2.Z2' - g6 Y1.Z2.Z1'.Z2')
elemtype=Complex128
DH=8
Tj=zeros(elemtype,DH,DH,dp,dp)
Tj[1,1,:,:]=kron(I,I)
Tj[2,1,:,:]=g1*kron(Z,I)
Tj[3,1,:,:]=g3*kron(I,Z)
Tj[4,1,:,:]=-h*kron(Z,Y)
Tj[5,1,:,:]=g4*kron(Y,X)
Tj[6,1,:,:]=-g5*kron(I,Z)
Tj[7,1,:,:]=-g6*kron(Z,Z)
Tj[8,1,:,:]=g2*kron(X,X)
Tj[8,2,:,:]=kron(Z,I)
Tj[8,3,:,:]=kron(X,Z)
Tj[8,4,:,:]=kron(Z,Z)
Tj[8,5,:,:]=kron(Z,I)
Tj[8,6,:,:]=kron(I,Y)
Tj[8,7,:,:]=kron(Y,Z)
Tj[8,8,:,:]=kron(I,I)
=#

dp=4
J11=-1
J12=-1
J13=-1
J2=-4
J31=J32=J33=0.01
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

#obtain boundary MPO
T1=zeros(elemtype,1,DH,dp,dp)
T1[1,:,:,:]=Tj[end,:,:,:]

TL=zeros(elemtype,DH,1,dp,dp)
TL[:,1,:,:]=Tj[:,1,:,:]

H_triality=[]
push!(H_triality,T1)
for j=2:L-1
    Tj[4,1,:,:]=(-h-(-1)^j*δh)*kron(Z,Y) #we try staggered h terms to kill one gapless mode
    push!(H_triality,Tj)
end
TL[4,1,:,:]=(-h-(-1)^L*δh)*kron(Z,Y)
push!(H_triality,TL)

#perform dmrg
dmrg_mpo!(2,[20,30,50],H_triality)
