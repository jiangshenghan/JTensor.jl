
include("../src/JTensor.jl")
using JTensor

elemtype=Complex128

I=[1. 0; 0 1.]
X=[0 1.; 1. 0]
Y=[0 -im*1.; im*1. 0]
Z=[1. 0; 0 -1.]

Sp=[0 1; 0 0]
Sm=[0 0; 1 0]
Sz=0.5*Z

#Ising model
#=
dp=2
h=1
J=1
@show h,J
DH=3
Wh=zeros(Complex128,DH,DH,dp,dp)
Wh[1,1,:,:]=I
Wh[2,1,:,:]=Z
Wh[3,1,:,:]=-h*X
Wh[3,2,:,:]=-J*Z
Wh[3,3,:,:]=I
#long range hopping
#Wh[2,2,:,:]=0.1*I
# =#

#double Ising model
#dp=4
#h=1
#J=1
#@show h,J
#DH=4
#Wh=zeros(DH,DH,dp,dp)
#Wh[1,1,:,:]=kron(I,I)
#Wh[2,1,:,:]=kron(Z,I)
#Wh[3,1,:,:]=kron(I,Z)
#Wh[4,1,:,:]=-h*(kron(X,I)+kron(I,X))
#Wh[4,2,:,:]=-J*kron(Z,I)
#Wh[4,3,:,:]=-J*kron(I,Z)
#Wh[4,4,:,:]=kron(I,I)

#clock model
#=
dp=2
Jx=-1
Jz=-1
h=0
@show Jx,Jz,h
DH=4
Wh=zeros(DH,DH,dp,dp)
Wh[1,1,:,:]=I
Wh[2,1,:,:]=X
Wh[3,1,:,:]=Z
Wh[4,1,:,:]=0
Wh[4,2,:,:]=Jx*X
Wh[4,3,:,:]=Jz*Z
Wh[4,4,:,:]=I
# =#

#spin-1/2 Heisenberg model
#=
dp=2
J=-1
Δ=1
@show J,Δ
DH=5
Wh=zeros(DH,DH,dp,dp)
Wh[1,1,:,:]=I
Wh[2,1,:,:]=Sm
Wh[3,1,:,:]=Sp
Wh[4,1,:,:]=Sz
Wh[5,1,:,:]=0
Wh[5,2,:,:]=J/2*Sp
Wh[5,3,:,:]=J/2*Sm
Wh[5,4,:,:]=Δ*Sz
Wh[5,5,:,:]=I
# =#

#Wh=zeros(Complex128,DH,DH,dp,dp)
#Wh[1,1,:,:]=I
#Wh[2,1,:,:]=X
#Wh[3,1,:,:]=Y
#Wh[4,1,:,:]=Z
#Wh[5,1,:,:]=0
#Wh[5,2,:,:]=J/4*X
#Wh[5,3,:,:]=J/4*Y
#Wh[5,4,:,:]=Δ/4*Z
#Wh[5,5,:,:]=I


#triality model
##=
dp=4
J11=-1.5
J12=-1
J13=-1
J2=-2
J31=J32=J33=0.4
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
# =#

#perform vumps
errp=0.1
#chis=[5,6,7,8,9,10,12,14,17,20,23,27,31,35,40,45,50]
#chis=[5,6,7,8,9,10,12,14,17,20,23,27,31,35,39,44,49,53,58,63,69,75,81,88,95,103]
#chis=[8]
chis=[20,60,90,120,140,180]

Al=[]
Ar=[]
C=[]
Fl=[]
Fr=[]
@show chis
for (i,chi) in enumerate(chis)
    @show i,chi
    if i>1 
        Al,Ar,C=one_site_vumps_inc_chi(Wh,chi,Al,Ar,C,Fl,Fr)
    end
    Al,Ar,C,Fl,Fr,errp=one_site_ham_vumps(Wh,chi,Al,Ar,C,maxiter=300,err0=1e-8)
    #obtain correlation length
    Allm=LinearMap([rand(elemtype,chi,chi),Al,conj(Al)],[[1,2],[1,-1,3],[2,-2,3]],1,elemtype=elemtype)
    λl,_=eigs(Allm,nev=2)
    xi=1/log(abs(λl[1]/λl[2]))
    @show λl,xi
end

