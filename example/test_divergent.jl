
include("../src/JTensor.jl")
using JTensor


#pi rvb D=6
T=readdlm("/home/jiangsb/code/JTensor.jl/tensor_data/square_pi_flux")
T=[T[:,1],T[:,2]]
T=[reshape(T[i],2,6,6,6,6) for i=1:2]
virt_spin=[0,0.5,1]

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

#init MPS from file
chi_spin=readcsv("/home/jiangsb/code/JTensor.jl/tensor_data/chi40")[:,1]
chi=Int(sum(x->2x+1,chi_spin))
Alrvec=readcsv("/home/jiangsb/code/JTensor.jl/tensor_data/Alr_chi40_II")
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

Fl=[]
Fr=[]


Jc=mapreduce(x->[1-4*mod(x,1) for i=1:2x+1],append!,chi_spin)
Jc=diagm(Jc)

λ0=[]

Alu2,Aru2,Acu,Cu,Fl,Fr,_,err,λ0=JTensor.sl_mag_trans_vumps_test(TTu,chi,Jc,Alu,Aru,Acu,Cu,Fl,Fr,e0=1e-10,maxiter=1,ncv=30,nev=4)
@show 1-abs(dot(Alu2[:],Alu[:]))/(norm(Alu2[:])*norm(Alu[:]))
for t=0:0.1:1
    @show t
    Al_test=t*Alu+(1-t)Alu2
    Ar_test=t*Aru+(1-t)Aru2
    @show 1-abs(dot(Al_test[:],Alu[:]))/(norm(Al_test[:])*norm(Alu[:]))
    @show 1-abs(dot(Al_test[:],Alu2[:]))/(norm(Al_test[:])*norm(Alu2[:]))
    JTensor.sl_mag_trans_vumps_test(TTu,chi,Jc,Al_test,Ar_test,Acu,Cu,Fl,Fr,e0=1e-10,maxiter=1,ncv=30,nev=4,f0=true,λ0=λ0)
end
