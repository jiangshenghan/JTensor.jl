
include("../src/JTensor.jl")
using JTensor

L=24
dp=2
J=1
Jz=1
@show L,dp,J,Jz

I=[1 0; 0 1]
Sp=[0 1; 0 0]
Sm=[0 0; 1 0]
Sz=[0.5 0; 0 -0.5]

#construct MPO for H=\sum_i J/2 (S+.S-' + S-.S+') + Jz Sz.Sz'
DH=5
Tj=zeros(DH,DH,dp,dp)
Tj[1,1,:,:]=I
Tj[2,1,:,:]=Sp
Tj[3,1,:,:]=Sm
Tj[4,1,:,:]=Sz
Tj[5,1,:,:]=0
Tj[5,2,:,:]=J/2*Sm
Tj[5,3,:,:]=J/2*Sp
Tj[5,4,:,:]=Jz*Sz
Tj[5,5,:,:]=I

T1=zeros(1,DH,dp,dp)
T1[1,:,:,:]=Tj[end,:,:,:]

TL=zeros(DH,1,dp,dp)
TL[:,1,:,:]=Tj[:,1,:,:]

H_heisenberg=[]
push!(H_heisenberg,T1)
for j=2:L-1
    push!(H_heisenberg,Tj)
end
push!(H_heisenberg,TL)

#perform dmrg
dmrg_mpo!(3,[20,30,40],H_heisenberg)
