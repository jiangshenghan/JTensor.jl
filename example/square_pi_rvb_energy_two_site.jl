include("../src/JTensor.jl")
using JTensor

chi_spin=[0,0,0,0,0.5,0.5,0.5,0.5]
#chi_spin=[0,0,0.5,0.5]
chi=Int(sum(x->2x+1,chi_spin))
dchi=[2,2,2]
maxiter=[50,50,50,50]
println("chi=",chi)
println("chi spins: ",chi_spin)
println("maxiter=",maxiter)
println()
flush(STDOUT)


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


#initialize
D=size(T[1],2)
DD=D^2
TTu=[permutedims(reshape(jcontract([T[i],conj(T[i])],[[1,-1,-3,-5,-7],[1,-2,-4,-6,-8]]),DD,DD,DD,DD),[1,3,2,4]) for i=1:2]

#symmetry transformation
#D=3 case
if D==3
    W=[0 1 0; -1 0 0; 0 0 1]
end
WW=reshape(jcontract([W,W],[[-1,-3],[-2,-4]]),DD,DD)

srand()
Alu=complex(rand(chi,chi,DD))
Alu=[Alu,Alu]
Aru=Alu
err=1e-12

for k=1:length(maxiter)
    @show k,chi
    @show chi_spin
    println()

    Acu=Alu
    Cu=rand(Complex128,chi,chi)
    Cu=[Cu,Cu]

    #spin symmetric subspace
    MA=spin_singlet_space_from_cg([chi_spin,chi_spin,virt_spin,virt_spin],[1,-1,1,-1])
    MA=reshape(MA,chi,chi,DD,size(MA)[end])
    MC=spin_singlet_space_from_cg([chi_spin,chi_spin],[1,-1])

    @show vecnorm(Alu[1]-sym_tensor_proj(Alu[1],MA))
    @show vecnorm(Aru[1]-sym_tensor_proj(Aru[1],MA))

    for iter=1:maxiter[k]
        Alu,Aru,Acu,Cu,_,_,_,err=sl_mult_vumps_par(TTu,chi,Alu,Aru,Acu,Cu,e0=err/10,maxiter=1,ncv=20)
        @show vecnorm(Alu[1]-sym_tensor_proj(Alu[1],MA))
        Alu[1]=sym_tensor_proj(Alu[1],MA)
        Aru[1]=sym_tensor_proj(Aru[1],MA)
        @show jcontract([Alu[1],conj(MA)],[[1,2,3],[1,2,3,-1]])
        @show jcontract([Aru[1],conj(MA)],[[1,2,3],[1,2,3,-1]])
        Acu[1]=sym_tensor_proj(Acu[1],MA)
        Cu[1]=sym_tensor_proj(Cu[1],MC)
        Alu=[Alu[1],Alu[1]]
        Aru=[Aru[1],Aru[1]]
        Acu=[Acu[1],Acu[1]]
        Cu=[Cu[1],Cu[1]]

        println("spin sym singualr vals:")
        svd_spin_sym_tensor(Cu[1],[1],[chi_spin,chi_spin],[1,-1])
        println()

        #lower imps by symmetry
        Ald=[jcontract([Alu[i],WW],[[-1,-2,1],[1,-3]]) for i=1:2]

        @printf("iter=%d\n",iter)
        square_heisenberg(Alu,Ald,T)
        println()
        
        if err<1e-8 break end
    end
    if k==length(maxiter) break end
    Alu,Aru,Acu,Cu,Fl,Fr,_,_=sl_mult_vumps_par(TTu,chi,Alu,Aru,Acu,Cu,e0=err/10,maxiter=1,ncv=20)
    Alu,Aru,chi,chi_spin=square_pi_flux_spin_sym_two_site_update(TTu,Fl,Fr,Alu,Aru,dchi[k],virt_spin,chi_spin)
    println()
end
