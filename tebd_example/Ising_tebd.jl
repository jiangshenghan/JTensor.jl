
include("../src/JTensor.jl")
using JTensor

L=20
dp=2
δt=0.05
tf=50
@show L,δt,tf

I=[1 0; 0 1]
X=[0 1; 1 0]
Y=[0 -im; im 0]
Z=[1 0; 0 -1]

#construct two-site Hamiltonian matrix, where col act on state
#TFIM
J=2
h=1
H=[]
push!(H,-J*jcontract([Z,Z],[[-1,-3],[-2,-4]])-h*jcontract([X,I],[[-1,-3],[-2,-4]])-0.5h*jcontract([I,X],[[-1,-3],[-2,-4]]))
for j=2:L-2
    push!(H,-J*jcontract([Z,Z],[[-1,-3],[-2,-4]])-0.5h*jcontract([X,I],[[-1,-3],[-2,-4]])-0.5h*jcontract([I,X],[[-1,-3],[-2,-4]]))
end
push!(H,-J*jcontract([Z,Z],[[-1,-3],[-2,-4]])-0.5h*jcontract([X,I],[[-1,-3],[-2,-4]])-h*jcontract([I,X],[[-1,-3],[-2,-4]]))
Hmat=[]
for j=1:L-1
    push!(Hmat,reshape(H[j],dp^2,dp^2))
end

#construct time evolution operator U(j,j+1,δt/2)
U=[]
for j=1:L-1
    Heig=eigfact(Hmat[j])
    push!(U,Heig[:vectors]*diagm(exp.(im*0.5δt*Heig[:values]))*Heig[:vectors]')
    U[j]=reshape(transpose(U[j]),dp,dp,dp,dp)
end

#initial state
#product state with one domain wall between site 3 and site 4
A=[]
for j=1:L
    push!(A,zeros(1,1,dp))
    if j<4 A[j][1]=1 else A[j][2]=1 end
end

#tebd
t=0
while t<tf
    @show t,jcontract([A[1],Z,conj(A[1])],[[1,2,3],[3,4],[1,2,4]])/norm(A[1][:])^2
    A=tebd_sweep(A,U,δt,δt)
    t=t+δt
end

