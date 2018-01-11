
include("../src/JTensor.jl")
using JTensor

L=40
d=2
δτ=0.05
τf=50
δt1=0.02
t1=5
δt2=0.02
t2=t1+10
δt3=0.02
t3=t2+100

@show L,δt1,t1,δt2,t2,δt3,t3

I=[1 0; 0 1]
X=[0 1; 1 0]
Y=[0 -im; im 0]
Z=[1 0; 0 -1]

#construct two-site Hamiltonian matrix, where col act on state
#TFIM, where at center bond (L/2,L/2+1) is J replaced by λ
J=1
h=0.2
h2=0
λ=-0.05
λf=1 #TODO: Should we choose the same sign λf for both positive and negative λ?
δλ=λf*δt2/(t2-t1)
@show J,h,h2,λ,λf,δλ
H=[]
push!(H,-J*jcontract([Z,Z],[[-1,-3],[-2,-4]])-h*jcontract([X,I],[[-1,-3],[-2,-4]])-0.5h*jcontract([I,X],[[-1,-3],[-2,-4]])-h2*jcontract([X,X],[[-1,-3],[-2,-4]]))
for j=2:L-2
    if j==div(L,2)
        push!(H,-λ*jcontract([Z,Z],[[-1,-3],[-2,-4]])-0.5h*jcontract([X,I],[[-1,-3],[-2,-4]])-0.5h*jcontract([I,X],[[-1,-3],[-2,-4]])-h2*jcontract([X,X],[[-1,-3],[-2,-4]]))
    else
        push!(H,-J*jcontract([Z,Z],[[-1,-3],[-2,-4]])-0.5h*jcontract([X,I],[[-1,-3],[-2,-4]])-0.5h*jcontract([I,X],[[-1,-3],[-2,-4]])-h2*jcontract([X,X],[[-1,-3],[-2,-4]]))
    end
end
push!(H,-J*jcontract([Z,Z],[[-1,-3],[-2,-4]])-0.5h*jcontract([X,I],[[-1,-3],[-2,-4]])-h*jcontract([I,X],[[-1,-3],[-2,-4]])-h2*jcontract([X,X],[[-1,-3],[-2,-4]]))
Hmat=[]
for j=1:L-1
    push!(Hmat,reshape(H[j],d^2,d^2))
end
#@show Hmat

#construct imag time evolution operator Ue(2k,2k+1,δτ) and Uo(2k-1,2k,δτ/2)
Ue=[]
for k=1:div(L-1,2)
    j=2k
    Heig=eigfact(Hmat[j])
    push!(Ue,Heig[:vectors]*diagm(exp.(-δτ*Heig[:values]))*Heig[:vectors]')
    Ue[k]=reshape(transpose(Ue[k]),d,d,d,d)
end
Uo=[]
for k=1:div(L,2)
    j=2k-1
    Heig=eigfact(Hmat[j])
    push!(Uo,Heig[:vectors]*diagm(exp.(-0.5δτ*Heig[:values]))*Heig[:vectors]')
    Uo[k]=reshape(transpose(Uo[k]),d,d,d,d)
end

#initial state (we initialize MPS as a product state with fixed parity)
A=[]
for j=1:L
    push!(A,ones(1,1,d))
end
B=[]
for j=1:L-1 push!(B,ones(1,1)) end

#imaginary tebd, find g.s.
τ=0
while τ<τf
    A,B=tebd_even_odd_one_step(A,B,Ue,Uo;eps=1e-6,chi=50)
    τ=τ+δτ

    #for integer τ, check wf_norm and energy
    @show τ
    if abs(τ-round(τ))<1e-7
        wf_norm=jcontract([A[1],conj(A[1]),B[1],conj(B[1])],[[1,3,2],[1,4,2],[3,-1],[4,-2]])
        for j=2:L-1
            wf_norm=jcontract([wf_norm,A[j],conj(A[j]),B[j],conj(B[j])],[[1,2],[1,4,3],[2,5,3],[4,-1],[5,-2]])
        end
        wf_norm=jcontract([wf_norm,A[L],conj(A[L])],[[1,2],[1,4,3],[2,4,3]])
        
        bond_energy=zeros(L-1)
        bond_energy[1]=jcontract([A[1],B[1],A[2],B[2],H[1],conj(A[1]),conj(B[1]),conj(A[2]),conj(B[2])],[[1,2,5],[2,3],[3,4,6],[4,9],[5,6,7,8],[1,10,7],[10,11],[11,12,8],[12,9]])
        for j=2:L-2
            bond_energy[j]=jcontract([B[j-1],A[j],B[j],A[j+1],B[j+1],H[j],conj(B[j-1]),conj(A[j]),conj(B[j]),conj(A[j+1]),conj(B[j+1])],[[1,2],[2,3,7],[3,4],[4,5,8],[5,6],[7,8,9,10],[1,11],[11,12,9],[12,13],[13,14,10],[14,6]])
        end
        bond_energy[end]=jcontract([B[L-2],A[L-1],B[L-1],A[L],H[L-1],conj(B[L-2]),conj(A[L-1]),conj(B[L-1]),conj(A[L])],[[1,2],[2,3,6],[3,4],[4,5,7],[6,7,8,9],[1,10],[10,11,8],[11,12],[12,5,9]])

        energy=sum(bond_energy)
        @show τ,wf_norm,bond_energy,energy
    end
end

#evolve using Hamiltonian H(λ) with invariant λ
#construct time evolution operator Ue(2k,2k+1,δt1) and Uo(2k-1,2k,δt1/2)
Ue=[]
for k=1:div(L-1,2)
    j=2k
    Heig=eigfact(Hmat[j])
    push!(Ue,Heig[:vectors]*diagm(exp.(-im*δt1*Heig[:values]))*Heig[:vectors]')
    Ue[k]=reshape(transpose(Ue[k]),d,d,d,d)
end
Uo=[]
for k=1:div(L,2)
    j=2k-1
    Heig=eigfact(Hmat[j])
    push!(Uo,Heig[:vectors]*diagm(exp.(-0.5*im*δt1*Heig[:values]))*Heig[:vectors]')
    Uo[k]=reshape(transpose(Uo[k]),d,d,d,d)
end

t=0
cind=div(L,2)
while t<t1
    A,B=tebd_even_odd_one_step(A,B,Ue,Uo;eps=1e-6,chi=50)
    t=t+δt1

    #for integer t, check wf_norm and measure majorana
    if abs(t-round(t))<1e-5
        wf_norm=jcontract([A[1],conj(A[1]),B[1],conj(B[1])],[[1,3,2],[1,4,2],[3,-1],[4,-2]])
        for j=2:L-1
            wf_norm=jcontract([wf_norm,A[j],conj(A[j]),B[j],conj(B[j])],[[1,2],[1,4,3],[2,5,3],[4,-1],[5,-2]])
        end
        wf_norm=jcontract([wf_norm,A[L],conj(A[L])],[[1,2],[1,4,3],[2,4,3]])

        ZZ=jcontract([B[cind-1],A[cind],B[cind],A[cind+1],B[cind+1],Z,Z,conj(B[cind-1]),conj(A[cind]),conj(B[cind]),conj(A[cind+1]),conj(B[cind+1])],[[1,2],[2,3,7],[3,4],[4,5,8],[5,6],[7,9],[8,10],[1,11],[11,12,9],[12,13],[13,14,10],[14,6]]) 
        #XX=jcontract([B[cind-1],A[cind],B[cind],A[cind+1],B[cind+1],X,X,conj(B[cind-1]),conj(A[cind]),conj(B[cind]),conj(A[cind+1]),conj(B[cind+1])],[[1,2],[2,3,7],[3,4],[4,5,8],[5,6],[7,9],[8,10],[1,11],[11,12,9],[12,13],[13,14,10],[14,6]]) 

        @show t,wf_norm,ZZ
    end
    ZZ=jcontract([B[cind-1],A[cind],B[cind],A[cind+1],B[cind+1],Z,Z,conj(B[cind-1]),conj(A[cind]),conj(B[cind]),conj(A[cind+1]),conj(B[cind+1])],[[1,2],[2,3,7],[3,4],[4,5,8],[5,6],[7,9],[8,10],[1,11],[11,12,9],[12,13],[13,14,10],[14,6]]) 
    @show t,λ,ZZ
end

##=
#evolve using Hamiltonian H(λ) with increasing |λ|
#construct time evolution operator Ue(2k,2k+1,δt2) and Uo(2k-1,2k,δt2/2)
Ue=[]
for k=1:div(L-1,2)
    j=2k
    Heig=eigfact(Hmat[j])
    push!(Ue,Heig[:vectors]*diagm(exp.(-im*δt2*Heig[:values]))*Heig[:vectors]')
    Ue[k]=reshape(transpose(Ue[k]),d,d,d,d)
end
Uo=[]
for k=1:div(L,2)
    j=2k-1
    Heig=eigfact(Hmat[j])
    push!(Uo,Heig[:vectors]*diagm(exp.(-0.5*im*δt2*Heig[:values]))*Heig[:vectors]')
    Uo[k]=reshape(transpose(Uo[k]),d,d,d,d)
end

#time evolution from t1 to t2 with time dependent Hamiltonian
t=t1
cind=div(L,2)
λ=0
while t<t2
    λ=λ+δλ
    H[cind]=-λ*jcontract([Z,Z],[[-1,-3],[-2,-4]])-0.5h*jcontract([X,I],[[-1,-3],[-2,-4]])-0.5h*jcontract([I,X],[[-1,-3],[-2,-4]])-h2*jcontract([X,X],[[-1,-3],[-2,-4]])
    Hmat[cind]=reshape(H[cind],d^2,d^2)

    #modify Ue or Uo
    Heig=eigfact(Hmat[cind])
    if iseven(cind)
        Ue[div(cind,2)]=Heig[:vectors]*diagm(exp.(-im*δt2*Heig[:values]))*Heig[:vectors]'
        Ue[div(cind,2)]=reshape(transpose(Ue[div(cind,2)]),d,d,d,d)
    else
        Uo[div(cind+1,2)]=Heig[:vectors]*diagm(exp.(-im*0.5*δt2*Heig[:values]))*Heig[:vectors]'
        Uo[div(cind+1,2)]=reshape(transpose(Uo[div(cind+1,2)]),d,d,d,d)
    end

    A,B=tebd_even_odd_one_step(A,B,Ue,Uo;eps=1e-6,chi=50)
    t=t+δt2

    #for integer t, check wf_norm and measure majorana
    if abs(t-round(t))<1e-5
        wf_norm=jcontract([A[1],conj(A[1]),B[1],conj(B[1])],[[1,3,2],[1,4,2],[3,-1],[4,-2]])
        for j=2:L-1
            wf_norm=jcontract([wf_norm,A[j],conj(A[j]),B[j],conj(B[j])],[[1,2],[1,4,3],[2,5,3],[4,-1],[5,-2]])
        end
        wf_norm=jcontract([wf_norm,A[L],conj(A[L])],[[1,2],[1,4,3],[2,4,3]])

        cind=div(L,2)
        ZZ=jcontract([B[cind-1],A[cind],B[cind],A[cind+1],B[cind+1],Z,Z,conj(B[cind-1]),conj(A[cind]),conj(B[cind]),conj(A[cind+1]),conj(B[cind+1])],[[1,2],[2,3,7],[3,4],[4,5,8],[5,6],[7,9],[8,10],[1,11],[11,12,9],[12,13],[13,14,10],[14,6]]) 
        #XX=jcontract([B[cind-1],A[cind],B[cind],A[cind+1],B[cind+1],X,X,conj(B[cind-1]),conj(A[cind]),conj(B[cind]),conj(A[cind+1]),conj(B[cind+1])],[[1,2],[2,3,7],[3,4],[4,5,8],[5,6],[7,9],[8,10],[1,11],[11,12,9],[12,13],[13,14,10],[14,6]]) 

        @show t,λ,wf_norm,ZZ
    end
    ZZ=jcontract([B[cind-1],A[cind],B[cind],A[cind+1],B[cind+1],Z,Z,conj(B[cind-1]),conj(A[cind]),conj(B[cind]),conj(A[cind+1]),conj(B[cind+1])],[[1,2],[2,3,7],[3,4],[4,5,8],[5,6],[7,9],[8,10],[1,11],[11,12,9],[12,13],[13,14,10],[14,6]]) 
    @show t,λ,ZZ
end

#time evolution from t2 to t3 with H(λf)
#construct time evolution operator Ue(2k,2k+1,δt3) and Uo(2k-1,2k,δt3/2)
#for j in 1:L-1 @show j,Hmat[j]; end
Ue=[]
for k=1:div(L-1,2)
    j=2k
    Heig=eigfact(Hmat[j])
    push!(Ue,Heig[:vectors]*diagm(exp.(-im*δt3*Heig[:values]))*Heig[:vectors]')
    Ue[k]=reshape(transpose(Ue[k]),d,d,d,d)
end
Uo=[]
for k=1:div(L,2)
    j=2k-1
    Heig=eigfact(Hmat[j])
    push!(Uo,Heig[:vectors]*diagm(exp.(-0.5*im*δt3*Heig[:values]))*Heig[:vectors]')
    Uo[k]=reshape(transpose(Uo[k]),d,d,d,d)
end

t=t2
while t<t3
    A,B=tebd_even_odd_one_step(A,B,Ue,Uo;eps=1e-6,chi=50)
    t=t+δt3

    #for integer t, check wf_norm and measure majorana
    if abs(t-round(t))<1e-5
        wf_norm=jcontract([A[1],conj(A[1]),B[1],conj(B[1])],[[1,3,2],[1,4,2],[3,-1],[4,-2]])
        for j=2:L-1
            wf_norm=jcontract([wf_norm,A[j],conj(A[j]),B[j],conj(B[j])],[[1,2],[1,4,3],[2,5,3],[4,-1],[5,-2]])
        end
        wf_norm=jcontract([wf_norm,A[L],conj(A[L])],[[1,2],[1,4,3],[2,4,3]])

        #ZZ=jcontract([B[cind-1],A[cind],B[cind],A[cind+1],B[cind+1],Z,Z,conj(B[cind-1]),conj(A[cind]),conj(B[cind]),conj(A[cind+1]),conj(B[cind+1])],[[1,2],[2,3,7],[3,4],[4,5,8],[5,6],[7,9],[8,10],[1,11],[11,12,9],[12,13],[13,14,10],[14,6]])
        #XX=jcontract([B[cind-1],A[cind],B[cind],A[cind+1],B[cind+1],X,X,conj(B[cind-1]),conj(A[cind]),conj(B[cind]),conj(A[cind+1]),conj(B[cind+1])],[[1,2],[2,3,7],[3,4],[4,5,8],[5,6],[7,9],[8,10],[1,11],[11,12,9],[12,13],[13,14,10],[14,6]]) 

        @show t,wf_norm,ZZ
    end
    ZZ=jcontract([B[cind-1],A[cind],B[cind],A[cind+1],B[cind+1],Z,Z,conj(B[cind-1]),conj(A[cind]),conj(B[cind]),conj(A[cind+1]),conj(B[cind+1])],[[1,2],[2,3,7],[3,4],[4,5,8],[5,6],[7,9],[8,10],[1,11],[11,12,9],[12,13],[13,14,10],[14,6]])
    @show t,λ,ZZ
end
# =#
