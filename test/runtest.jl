
include("../src/JTensor.jl")
using JTensor
using Base.Test

println("test JTensor!")
flush(STDOUT)

#test jcontract
#1. vector*vector
n=rand(1:20)
a=rand(Complex128,n)
b=rand(Complex128,n)
r1=jcontract([a,b],[[1],[1]])
r2=(a.'*b)[1]
@test abs(r1-r2)/abs(r1)<1e-12

#2. matrix*vector
n1,n2=rand(1:20,2)
a=rand(Complex128,n1,n2)
b=rand(Complex128,n2)
r1=jcontract([a,b],[[-1,1],[1]])
r2=a*b
@test norm(r1-r2)/norm(r1)<1e-12

#3. matrix*matrix*matrix
n1,n2,n3,n4=rand(1:20,4)
a=rand(Complex128,n1,n2)
b=rand(Complex128,n2,n3)
c=rand(Complex128,n3,n4)
r1=jcontract([a,b,c],[[-1,1],[1,2],[2,-2]])
r2=a*b*c
@test norm(r1-r2)/norm(r1)<1e-12

#4 tensor multiplication
n1,n2,n3=rand(1:20,3)
v=rand(Complex128,n1,n2,n1)
A=rand(Complex128,n1,n1,n3)
T=rand(Complex128,n2,n2,n3,n3)
r1=jcontract([v,A,T,conj(A)],[[1,2,3],[1,-1,4],[2,-2,4,5],[3,-3,5]])
r1=r1[:]
r2=reshape(reshape(permutedims(v,[2,3,1]),n2*n1,n1)*reshape(A,n1,n1*n3),n2,n1,n1,n3)
r2=reshape(reshape(permutedims(r2,[2,3,1,4]),n1^2,n2*n3)*reshape(permutedims(T,[1,3,2,4]),n2*n3,n2*n3),n1,n1,n2,n3)
r2=reshape(permutedims(r2,[2,3,1,4]),n1*n2,n1*n3)*reshape(permutedims(conj(A),[1,3,2]),n1*n3,n1)
r2=r2[:]
@test norm(r1-r2)/norm(r1)<1e-12


#test linearmap
n=rand(1:20)
a=rand(Complex128,n)
b=rand(Complex128,n,n)
lm=LinearMap([a,b],[[1],[1,-1]],1,issym=false,elemtype=Complex128)
eigval1=eigs(lm;nev=min(5,n))[1]
eigval2=eigs(b.';nev=min(5,n))[1]
@test norm(eigval1-eigval2)/norm(eigval1)<1e-12

#test vumps algorithm
#1. square Ising model
kIsing(β)=1/(sinh(2*β)^2)
ZIsing(β)=exp(1/2pi*quadgk(x->log(cosh(2*β)*cosh(2*β)+1/kIsing(β)*sqrt(1+kIsing(β)^2-2*kIsing(β)*cos(2*x))),0,pi)[1]+log(2)/2)
T=zeros(2,2,2,2)
T[1,1,1,1]=T[2,2,2,2]=1
BIsing(β)=[exp(β) exp(-β); exp(-β) exp(β)]
TIsing(β)=jcontract([T,BIsing(β),BIsing(β)],[[1,2,-2,-4],[-1,1],[-3,2]])
β=rand();
r1=sl_one_vumps(TIsing(β),20)[end-1]
r2=ZIsing(β)
r3=sl_two_vumps(TIsing(β),TIsing(β),20)[end-1]
@test abs(r1-r2)/abs(r1)<1e-5
@test abs(r1-sqrt(r3))/abs(r1)<1e-5

#test itebd algorithm
