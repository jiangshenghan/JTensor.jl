include("../src/JTensor.jl")
using JTensor

#zero srvb
T=zeros(2,3,3,3,3)
T[1,2,3,3,3]=1
T[2,1,3,3,3]=-1
T[1,3,1,3,3]=-1
T[2,3,2,3,3]=-1
T[1,3,3,1,3]=-1
T[2,3,3,2,3]=-1
T[1,3,3,3,2]=1
T[2,3,3,3,1]=-1
virt_spin=[0.5,0]

D=size(T,2)
DD=D^2

TTu=permutedims(reshape(jcontract([T,conj(T)],[[1,-1,-3,-5,-7],[1,-2,-4,-6,-8]]),DD,DD,DD,DD),[1,3,2,4])
TTd=permutedims(TTu,[2,1,4,3])

#symmetry transformation
#D=3 case
if D==3
    W=[0 1 0; -1 0 0; 0 0 1]
    J=[-1 0 0; 0 -1 0; 0 0 1]
end
WW=reshape(jcontract([W,W],[[-1,-3],[-2,-4]]),DD,DD)
JJ=reshape(jcontract([J,J],[[-1,-3],[-2,-4]]),DD,DD)


# #=
#random init MPS Alu,Aru
chi=8

srand()
Alu=rand(Complex128,chi,chi,DD)
Aru=rand(Complex128,chi,chi,DD)

Acu=rand(Complex128,chi,chi,DD)
Cu=rand(Complex128,chi,chi)

Ald=rand(Complex128,chi,chi,DD)
Ard=rand(Complex128,chi,chi,DD)

Acd=rand(Complex128,chi,chi,DD)
Cd=rand(Complex128,chi,chi)

# =#


Fl=[]
Fr=[]

Gl=[]
Gr=[]

maxiter=300
@show maxiter
λ0=zeros(4)
nev=4
err=1
Jc=diagm(ones(chi))

for iter=1:maxiter
    Alu,Aru,Acu,Cu,Fl,Fr,_,err,λ0=JTensor.sl_mag_trans_vumps_test(TTu,chi,Jc,Alu,Aru,Acu,Cu,Fl,Fr,e0=err/10,maxiter=1,ncv=30,nev=nev,λ0=λ0)

    #lower imps by symmetry
    #Ald=jcontract([Alu,WW],[[-1,-2,1],[1,-3]])
    #Ald2=jcontract([Ald,JJ],[[-1,-2,1],[1,-3]])

    Ald,Ard,Acd,Cd,Gl,Gr,_,err,λ0=JTensor.sl_mag_trans_vumps_test(TTd,chi,Jc,Ald,Ard,Acd,Cd,Gl,Gr,maxiter=1,ncv=30,nev=nev,λ0=λ0)

    @show iter
    square_heisenberg([Alu,Alu],[Ald,Ald],[T,T])
    println()
    flush(STDOUT)

    if err<1e-10 break end

end

