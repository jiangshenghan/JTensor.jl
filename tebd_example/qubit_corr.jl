
include("../src/JTensor.jl")
using JTensor

L=7
dp=2
δt=0.01
tf=100
@show L,δt,tf

I=[1 0; 0 1]
X=[0 1; 1 0]
Y=[0 -im; im 0]
Z=[1 0; 0 -1]

#construct two-site Hamiltonian matrix, where col act on state
#TFIM
J=1
h=0.8
h2=0.1
@show J,h,h2
H=[]
push!(H,-J*jcontract([Z,Z],[[-1,-3],[-2,-4]])-h*jcontract([X,I],[[-1,-3],[-2,-4]])-0.5h*jcontract([I,X],[[-1,-3],[-2,-4]])-h2*jcontract([X,X],[[-1,-3],[-2,-4]]))
for j=2:L-2
    push!(H,-J*jcontract([Z,Z],[[-1,-3],[-2,-4]])-0.5h*jcontract([X,I],[[-1,-3],[-2,-4]])-0.5h*jcontract([I,X],[[-1,-3],[-2,-4]])-h2*jcontract([X,X],[[-1,-3],[-2,-4]]))
end
push!(H,-J*jcontract([Z,Z],[[-1,-3],[-2,-4]])-0.5h*jcontract([X,I],[[-1,-3],[-2,-4]])-h*jcontract([I,X],[[-1,-3],[-2,-4]])-h2*jcontract([X,X],[[-1,-3],[-2,-4]]))
Hmat=[]
for j=1:L-1
    push!(Hmat,reshape(H[j],dp^2,dp^2))
end

#construct time evolution operator Ue(2k,2k+1,δt) and Uo(2k-1,2k,δt/2)
Ue=[]
for k=1:div(L-1,2)
    j=2k
    Heig=eigfact(Hmat[j])
    push!(Ue,Heig[:vectors]*diagm(exp.(-im*δt*Heig[:values]))*Heig[:vectors]')
    Ue[k]=reshape(transpose(Ue[k]),dp,dp,dp,dp)
end
Uo=[]
for k=1:div(L,2)
    j=2k-1
    Heig=eigfact(Hmat[j])
    push!(Uo,Heig[:vectors]*diagm(exp.(-im*0.5δt*Heig[:values]))*Heig[:vectors]')
    Uo[k]=reshape(transpose(Uo[k]),dp,dp,dp,dp)
end

#initial state (we initialize MPS in canonical form)
#product state with one domain wall between site 3 and site 4
A=[]
for j=1:L
    push!(A,zeros(1,1,dp))
    if j<3 A[j][1]=1 else A[j][2]=1 end
end
B=[]
for j=1:L-1 push!(B,ones(1,1)) end

#tebd
t=0
while t<tf
    bdry_tens=jcontract([A[1],B[1]],[[-1,1,-3],[1,-2]])
    bdry_corr=jcontract([bdry_tens,Z,conj(bdry_tens)],[[1,2,3],[3,4],[1,2,4]])/norm(bdry_tens[:])
    bulk_ind=div(L,2)
    bulk_tens=jcontract([B[bulk_ind-1],A[bulk_ind],B[bulk_ind]],[[-1,1],[1,2,-3],[2,-2]])
    bulk_corr=-jcontract([bulk_tens,Z,conj(bulk_tens)],[[1,2,3],[3,4],[1,2,4]])/norm(bulk_tens[:])
    @show t,bdry_corr,bulk_corr
    A,B=tebd_even_odd_one_step(A,B,Ue,Uo;eps=1e-6)
    t=t+δt
end


