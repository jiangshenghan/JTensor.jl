
include("../src/JTensor.jl")
using JTensor

# #=
#pi srvb
T=[zeros(2,3,3,3,3) for i=1:2]
T[1][1,2,3,3,3]=1
T[1][2,1,3,3,3]=-1
T[1][1,3,1,3,3]=-1
T[1][2,3,2,3,3]=-1
T[1][1,3,3,1,3]=-1
T[1][2,3,3,2,3]=-1
T[1][1,3,3,3,2]=1
T[1][2,3,3,3,1]=-1

T[2][1,2,3,3,3]=1
T[2][2,1,3,3,3]=-1
T[2][1,3,1,3,3]=1
T[2][2,3,2,3,3]=1
T[2][1,3,3,1,3]=-1
T[2][2,3,3,2,3]=-1
T[2][1,3,3,3,2]=1
T[2][2,3,3,3,1]=-1

virt_spin=[0.5,0]
# =#

#=
#pi rvb D=6
T=readdlm("/home/jiangsb/code/JTensor.jl/tensor_data/square_pi_flux")
T=[T[:,1],T[:,2]]
T=[reshape(T[i],2,6,6,6,6) for i=1:2]
virt_spin=[0,0.5,1]
# =#

D=size(T[1],2)
DD=D^2
TTu=permutedims(reshape(jcontract([T[1],conj(T[1])],[[1,-1,-3,-5,-7],[1,-2,-4,-6,-8]]),DD,DD,DD,DD),[1,3,2,4])

#symmetry transformation
#D=3 case
if D==3
    W=[0 1 0; -1 0 0; 0 0 1]
end
#D=6(0+1/2+1) case
if D==6
    W=zeros(6,6)
    W[1,1]=W[2,3]=W[4,6]=W[6,4]=1
    W[3,2]=W[5,5]=-1
end
WW=reshape(jcontract([W,W],[[-1,-3],[-2,-4]]),DD,DD)


# #=
#random init MPS Alu,Aru
chi_spin=[0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.5,0.5,0.5,0.5,0.5]
chi=Int(sum(x->2x+1,chi_spin))

#spin symmetric subspace
MA=spin_singlet_space_from_cg([chi_spin,chi_spin,virt_spin,virt_spin],[1,-1,1,-1])
MA=reshape(MA,chi,chi,DD,size(MA)[end])
MC=spin_singlet_space_from_cg([chi_spin,chi_spin],[1,-1])
MF=spin_singlet_space_from_cg([chi_spin,virt_spin,virt_spin,chi_spin],[-1,-1,1,1])
MF=reshape(MF,chi,DD,chi,size(MF)[end])

srand()
Alu=rand(Complex128,chi,chi,DD)
Aru=rand(Complex128,chi,chi,DD)
Alu=sym_tensor_proj(Alu,MA)
Aru=sym_tensor_proj(Aru,MA)

Acu=rand(Complex128,chi,chi,DD)
Acu=sym_tensor_proj(Acu,MA)
Cu=rand(Complex128,chi,chi)
Cu=sym_tensor_proj(Cu,MC)
# =#


#=
#init MPS from file
chi_spin=readcsv("/home/jiangsb/code/JTensor.jl/tensor_data/chi40")[:,1]
chi=Int(sum(x->2x+1,chi_spin))
Alrvec=readcsv("/home/jiangsb/code/JTensor.jl/tensor_data/Alr_chi40")
Alvec=Alrvec[:,1]+im*Alrvec[:,2]
Arvec=Alrvec[:,3]+im*Alrvec[:,4]

#spin symmetric subspace
MA=spin_singlet_space_from_cg([chi_spin,chi_spin,virt_spin,virt_spin],[1,-1,1,-1])
MA=reshape(MA,chi,chi,DD,size(MA)[end])
MC=spin_singlet_space_from_cg([chi_spin,chi_spin],[1,-1])
MF=spin_singlet_space_from_cg([chi_spin,virt_spin,virt_spin,chi_spin],[-1,-1,1,1])
MF=reshape(MF,chi,DD,chi,size(MF)[end])

Alu=jcontract([MA,Alvec],[[-1,-2,-3,1],[1]])
Aru=jcontract([MA,Arvec],[[-1,-2,-3,1],[1]])
#check isometry
@show size(Alu),size(Aru)
@show vecnorm(Alu),vecnorm(Aru)
@show diag(jcontract([Alu,conj(Alu)],[[1,-1,2],[1,-2,2]]))
@show diag(jcontract([Aru,conj(Aru)],[[-1,1,2],[-2,1,2]]))
Acu=[]
Cu=[]
# =#

Fl=[]
Fr=[]


inc_spin_no=1*ones(Int,20)
maxiter=100*ones(Int,21)
@show inc_spin_no, maxiter
f0=0

λ0=[0,0,0,0]
nev=4
for inci=1:length(maxiter)

    err=1
    Jc=mapreduce(x->[1-4*mod(x,1) for i=1:2x+1],append!,chi_spin)
    Jc=diagm(Jc)
    @show inci
    @show chi_spin
    @show diag(Jc)

    for iter=1:maxiter[inci]
        #if inci==1 && iter==1
        if err>0.1 || inci==1
            Alu,Aru,Acu,Cu,Fl,Fr,_,err,λ0=JTensor.sl_mag_trans_vumps_test(TTu,chi,Jc,Alu,Aru,Acu,Cu,Fl,Fr,e0=err/10,maxiter=1,ncv=30,nev=nev,λ0=λ0)
        else
            Alu,Aru,Acu,Cu,Fl,Fr,_,err,λ0=JTensor.sl_mag_trans_vumps_test(TTu,chi,Jc,Alu,Aru,Acu,Cu,Fl,Fr,e0=err/10,maxiter=1,ncv=30,nev=nev,f0=true,λ0=λ0)
        end
        Alu=sym_tensor_proj(Alu,MA)
        Aru=sym_tensor_proj(Aru,MA)
        Acu=sym_tensor_proj(Acu,MA)
        Cu=sym_tensor_proj(Cu,MC)
        @show svd_spin_sym_tensor(Cu,[1],[chi_spin,chi_spin],[1,-1])[[2,4]]
        @show vecnorm(sym_tensor_proj(Fl,MF)-Fl)/vecnorm(Fl)
        #Fl=sym_tensor_proj(Fl,MF)
        #Fr=sym_tensor_proj(Fr,MF)

    @show jcontract([Alu,conj(MA)],[[1,2,3],[1,2,3,-1]])
    @show jcontract([Aru,conj(MA)],[[1,2,3],[1,2,3,-1]])

        #lower imps by symmetry
        Ald=jcontract([Alu,WW],[[-1,-2,1],[1,-3]])

        @show iter
        square_heisenberg([Alu,Alu],[Ald,Ald],T)
        println()
        flush(STDOUT)

        if err<1e-10 break end
        
        #=
        #reinit Fl,Fr
        if iter%20==0 && iter<maxiter[inci]
            Fl=[]
            Fr=[]
        end
        =#
    end

    @show jcontract([Alu,conj(MA)],[[1,2,3],[1,2,3,-1]])
    @show jcontract([Aru,conj(MA)],[[1,2,3],[1,2,3,-1]])

    if inci>length(inc_spin_no) break end
    println("two-site algoritim to add more spins:")
    @time A2c=mag_trans_A2c(TTu,Fl,Fr,Alu,Acu,Jc)
    #=
    A2csvals=svd(reshape(permutedims(A2c,[1,3,2,4]),chi*DD,chi*DD))[2]
    @show A2csvals/max(A2csvals...)
    MA2c=spin_singlet_space_from_cg([chi_spin,chi_spin,virt_spin,virt_spin,virt_spin,virt_spin],[1,-1,1,-1,1,-1])
    MA2c=reshape(MA2c,chi,chi,DD,DD,size(MA2c)[end])
    @show jcontract([A2c,conj(MA2c)],[[1,2,3,4],[1,2,3,4,-1]])
    =#

    chi_old=chi
    @time Alu,Aru,chi,chi_spin=spin_sym_dlmps_inc_chi(Alu,Aru,A2c,inc_spin_no[inci],virt_spin,chi_spin,[1,-1,1,-1])
    updated_Cu=zeros(eltype(Cu),chi,chi)
    updated_Cu[1:size(Cu,1),1:size(Cu,2)]=Cu
    Cu=updated_Cu
    Acu=jcontract([Alu,Cu],[[-1,1,-3],[1,-2]])
    Fl_old=Fl
    Fr_old=Fr
    Fl=zeros(Complex128,chi,DD,chi)
    Fl[1:chi_old,:,1:chi_old]=Fl_old
    Fr=zeros(Complex128,chi,DD,chi)
    Fr[1:chi_old,:,1:chi_old]=Fr_old

    #spin symmetric subspace
    MA=spin_singlet_space_from_cg([chi_spin,chi_spin,virt_spin,virt_spin],[1,-1,1,-1])
    MA=reshape(MA,chi,chi,DD,size(MA)[end])
    MC=spin_singlet_space_from_cg([chi_spin,chi_spin],[1,-1])
    MF=spin_singlet_space_from_cg([chi_spin,virt_spin,virt_spin,chi_spin],[-1,-1,1,1])
    MF=reshape(MF,chi,DD,chi,size(MF)[end])
    println()

end

