
include("../src/JTensor.jl")
using JTensor

L=20
dp=4
J1=1
h1=1
J2=2
h2=2
h=0
δh=0

I=[1 0; 0 1]
X=[0 1; 1 0]
Y=[0 -im; im 0]
Z=[1 0; 0 -1]

elemtype=Complex128
DH=4
Tj=zeros(elemtype,DH,DH,dp,dp)
Tj[1,1,:,:]=kron(I,I)
Tj[2,1,:,:]=-J1*kron(Y,I)
Tj[3,1,:,:]=-J2*kron(I,Y)
Tj[end,1,:,:]=-h1*kron(X,I)-h2*kron(I,X)
Tj[end,2,:,:]=kron(Z,I)
Tj[end,3,:,:]=kron(I,Z)
Tj[end,end,:,:]=kron(I,I)

T1=zeros(elemtype,1,DH,dp,dp)
T1[1,:,:,:]=Tj[end,:,:,:]
TL=zeros(elemtype,DH,1,dp,dp)
TL[:,1,:,:]=Tj[:,1,:,:]

H_test=[]
push!(H_test,T1)
for j=2:L-1
    push!(H_test,Tj)
end
push!(H_test,TL)

dmrg_mpo!(2,[20,30],H_test)
