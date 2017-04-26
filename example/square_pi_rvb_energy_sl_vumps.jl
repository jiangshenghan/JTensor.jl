include("../src/JTensor.jl")
using JTensor

chi=[6,8]
maxiter=[30,50]
println("chi=")
println(chi)
println("maxiter=")
println(maxiter)
println()
flush(STDOUT)

#pi srvb
T=[zeros(Complex128,2,3,3,3,3) for i=1:2]
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

#pi rvb D=6
#T=readdlm("/home/jiangsb/code/JTensor.jl/tensor_data/square_pi_flux")
#T=[T[:,1],T[:,2]]
#T=[reshape(T[i],2,6,6,6,6) for i=1:2]


D=size(T[1],2)
DD=D^2

TTu=[permutedims(reshape(jcontract([T[i],conj(T[i])],[[1,-1,-3,-5,-7],[1,-2,-4,-6,-8]]),DD,DD,DD,DD),[1,3,2,4]) for i=1:2]

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
    #W[6,1]=W[3,3]=W[2,5]=W[1,6]=1
    #W[5,2]=W[4,4]=-1
end
WW=reshape(jcontract([W,W],[[-1,-3],[-2,-4]]),DD,DD)

#initilize
Cu=[]
Alu=[]
Aru=[]
Acu=[]
Flu=[]
Fru=[]
err=1e-12

for counti=1:length(chi)
    @printf("counti=%d,\n chi=%d,\n maxiter=%d\n",counti,chi[counti],maxiter[counti])

    if counti>1

        A_update=zeros(Complex128,chi[counti],chi[counti],DD)
        A_update[1:chi[counti-1],1:chi[counti-1],:]=Alu[1]
        Alu=[A_update,A_update]
        A_update[1:chi[counti-1],1:chi[counti-1],:]=Aru[1]
        Aru=[A_update,A_update]
        A_update[1:chi[counti-1],1:chi[counti-1],:]=Acu[1]
        Acu=[A_update,A_update]

        C_update=zeros(Complex128,chi[counti],chi[counti])
        C_update[1:chi[counti-1],1:chi[counti-1],:]=Cu[1]
        Cu=[C_update,C_update]

        #=
        F_update=[zeros(Complex128,chi[counti],DD,chi[counti]) for i=1:2]
        for i=1:2 F_update[i][1:chi[counti-1],:,1:chi[counti-1]]=Flu[i] end
        Flu=F_update
        for i=1:2 F_update[i][1:chi[counti-1],:,1:chi[counti-1]]=Fru[i] end
        Fru=F_update
        ## =#
    end


    #VUMPS to obtain energy
    for iter=1:maxiter[counti]
        #Alu,Aru,Acu,Cu,Flu,Fru,_,err=sl_mult_vumps_par(TTu,chi[counti],Alu,Aru,Acu,Cu,Flu,Fru,e0=err/10,maxiter=1,ncv=20)
        Alu,Aru,Acu,Cu,_,_,_,err=sl_mult_vumps_par(TTu,chi[counti],Alu,Aru,Acu,Cu,e0=err/10,maxiter=1,ncv=20)
        Alu=[Alu[1],Alu[1]]
        Aru=[Aru[1],Aru[1]]
        Acu=[Acu[1],Acu[1]]
        Cu=[Cu[1],Cu[1]]

        #lower imps by symmetry
        Ald=[jcontract([Alu[i],WW],[[-1,-2,1],[1,-3]]) for i=1:2]

        @printf("iter=%d\n",iter)
        square_heisenberg(Alu,Ald,T)
    end

end



