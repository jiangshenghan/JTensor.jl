
include("../src/JTensor.jl")
using JTensor

dp=2
elemtype=Complex128

I=[1 0; 0 1]
X=[0 1; 1 0]
Y=[0 -im; im 0]
Z=[1 0; 0 -1]

#construct MPO for H=\sum_i -J Z_i Z_{i+1}-h X_i
h=1
J=1
ham=-J*jcontract([Z,Z],[[-1,-3],[-2,-4]])-h*jcontract([X,I],[[-1,-3],[-2,-4]])
DH=3
WH=zeros(DH,DH,dp,dp)
WH[1,1,:,:]=I
WH[1,2,:,:]=Z
WH[1,3,:,:]=-h*X
WH[2,3,:,:]=-J*Z
WH[3,3,:,:]=I

#perform itebd
chi=4
A=rand(elemtype,chi,chi,dp)
Fl=[]
Fr=[]


#dts=[-0.1,-0.01,-0.001,-1e-4,-1e-5]
dts=[-0.1,-0.01,-0.001]
err_th=1e-5
for dt in dts 
    dt1=(1+im)/2*dt
    dt2=(1-im)/2*dt
    #W_expH=to_exp_H_mpo(WH,dt)
    W_expH_I=to_exp_H_mpo(WH,dt1)
    W_expH_II=to_exp_H_mpo(WH,dt2)
    C=zeros(chi)
    for i=1:3000
        C_last=C
        B,C,Fl,Fr=sl_mult_mpo_mps([A],[W_expH_I],chi,Fl,Fr,hard_truncate=true)
        A=jcontract([diagm(C[1]),B[1]],[[-1,1],[1,-2,-3]])
        B,C,Fl,Fr=sl_mult_mpo_mps([A],[W_expH_II],chi,Fl,Fr,hard_truncate=true)
        A=jcontract([diagm(C[1]),B[1]],[[-1,1],[1,-2,-3]])
        B=B[1]
        C=C[1]
        # measure energy
        if i%10==0
            CAr=jcontract([diagm(C),B,diagm(C)],[[-1,1],[1,2,-3],[2,-2]])
            energy=jcontract([A,ham,conj(A),CAr,conj(CAr)],[[1,2,4],[4,5,6,7],[1,8,6],[2,3,5],[8,3,7]])/jcontract([A,conj(A),CAr,conj(CAr)],[[1,2,4],[1,6,4],[2,3,5],[6,3,5]])
            #@show jcontract([A,conj(A),CAr,conj(CAr)],[[1,2,4],[1,6,4],[2,3,5],[6,3,5]])

            @show chi,dt,i,energy

            #error threhold
            err=vecnorm(abs.(C_last/vecnorm(C_last)-C/vecnorm(C)))
            @show dt,i,err,abs(dt*err_th)
            if err<abs(dt*err_th) break end

            #get entanglement entropy
            C=C/norm(C)
            entropy=sum(x->-x^2*2*log(x),C);
            @show chi,dt,i,entropy
        end

    end
end


