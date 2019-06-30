#We consider (three-body) Hamiltonian with triality T, where T^3 act as identity on this local terms
#H=\sum_j Z_j^1 Z_{j+1}^1 + X_j^1 X_j^2 + X_{j-1}^1 Z_j^2 Z_{j+1}^2

include("../src/JTensor.jl")
using JTensor

L=32
dp=4
@show L,dp

I=[1 0; 0 1]
X=[0 1; 1 0]
Y=[0 -im; im 0]
Z=[1 0; 0 -1]

elemtype=Complex128
DH=5
#2<j<L-1
Tj=zeros(elemtype,DH,DH,dp,dp)
Tj[1,1,:,:]=kron(I,I)
Tj[2,1,:,:]=kron(Z,I)
Tj[3,1,:,:]=kron(I,Z)
Tj[4,3,:,:]=kron(I,Z)
Tj[5,1,:,:]=kron(X,X)
Tj[5,2,:,:]=kron(Z,I)
Tj[5,4,:,:]=kron(X,I)
Tj[5,5,:,:]=kron(I,I)

T1=zeros(elemtype,1,DH-1,dp,dp)
T1[1,:,:,:]=Tj[end,[1,2,4,5],:,:]
TL=zeros(elemtype,DH-1,1,dp,dp)
TL[:,1,:,:]=Tj[[1,2,4,5],1,:,:]

H_triality=[]
push!(H_triality,T1)
push!(H_triality,Tj[[1,2,4,5],:,:,:])
for j=3:L-2
    push!(H_triality,Tj)
end
push!(H_triality,Tj[:,[1,2,4,5],:,:])
push!(H_triality,TL)
@show length(H_triality)
for j=1:L 
    @show size(H_triality[j])
end

dmrg_mpo!(3,[10,20,40],H_triality)
