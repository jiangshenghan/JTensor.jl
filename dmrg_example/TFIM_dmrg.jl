
include("../src/JTensor.jl")
using JTensor

L=64
dp=2
J=1
h=1
@show L,dp,J,h

I=[1 0; 0 1]
X=[0 1; 1 0]
Y=[0 -im; im 0]
Z=[1 0; 0 -1]

#construct MPO for H=\sum_i -J Z_i Z_{i+1}-h Z_i^x
DH=3
Tj=zeros(DH,DH,dp,dp)
Tj[1,1,:,:]=I
Tj[2,1,:,:]=Z
Tj[3,1,:,:]=-h*X
Tj[3,2,:,:]=-J*Z
Tj[3,3,:,:]=I

T1=zeros(1,DH,dp,dp)
T1[1,:,:,:]=Tj[3,:,:,:]

TL=zeros(DH,1,dp,dp)
TL[:,1,:,:]=Tj[:,1,:,:]

H_Ising=[]
push!(H_Ising,T1)
for j=2:L-1
    push!(H_Ising,Tj)
end
push!(H_Ising,TL)

#perform dmrg
dmrg_mpo!(3,[20,20,20],H_Ising)
#dmrg_mpo!(4,[6,10,20,30],H_Ising)

