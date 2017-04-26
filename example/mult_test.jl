include("../src/JTensor.jl")
using JTensor

function mult_test(Al,TTu)
    left_tensor_list=[]
    left_legs_list=[]
    chi=size(Al[1],1)
    DD=size(TTu[1],1)
    push!(left_tensor_list,rand(Complex128,chi,DD,chi))
    push!(left_legs_list,[1,2,3])
    for il=1:2
        append!(left_tensor_list,[Al[il],TTu[il],conj(Al[il])])
        legs_list_il=[[1,6,4],[2,7,4,5],[3,8,5]]
        legs_list_il+=(il-1)*5
        if (il==2)
            legs_list_il[1][2]=-1
            legs_list_il[2][2]=-2
            legs_list_il[3][2]=-3
        end
        append!(left_legs_list,legs_list_il)
    end
    leftlm=LinearMap(left_tensor_list,left_legs_list,1,elemtype=Complex128)
    vec=zeros(Complex128,chi^2*DD)
    init_vec=ones(Complex128,chi^2*DD)
    A_mul_B!(vec,leftlm,init_vec)
    @show vecnorm(Al[1])
    @show vecnorm(vec)
end


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

D=size(T[1],2)
DD=D^2

TTu=[permutedims(reshape(jcontract([T[i],conj(T[i])],[[1,-1,-3,-5,-7],[1,-2,-4,-6,-8]]),DD,DD,DD,DD),[1,3,2,4]) for i=1:2]


A=rand(Complex128,6,6,DD)
A_update=zeros(Complex128,8,8,DD)
A_update[1:6,1:6,:]=A
A=[A,A]
A_update=[A_update,A_update]
mult_test(A,TTu)
mult_test(A_update,TTu)

FA=rand(Complex128,6,DD,6)
FA_update=zeros(Complex128,8,DD,8)
FA_update[1:6,:,1:6]=FA
FA=[FA,FA]
FA_update=[FA_update,FA_update]
println("chi=6")
Al,Ar,Ac,C,Fl,Fr=sl_mult_vumps_par(TTu,6,A,A,[],[],FA,FA,e0=1e-12,maxiter=2)
@show vecnorm(Fl[1]),vecnorm(Fl[2])
@show vecnorm(Fr[1]),vecnorm(Fr[2])
@show vecnorm(Ac[1]),vecnorm(Ac[2])
println("chi=8")
Al,Ar,Ac,C,Fl,Fr=sl_mult_vumps_par(TTu,8,A_update,A_update,[],[],FA_update,FA_update,maxiter=2,e0=1e-12)
@show vecnorm(Fl[1]),vecnorm(Fl[1][1:6,:,1:6])
@show vecnorm(Fl[2]),vecnorm(Fl[2][1:6,:,1:6])
@show vecnorm(Fr[1]),vecnorm(Fr[1][1:6,:,1:6])
@show vecnorm(Fr[2]),vecnorm(Fr[2][1:6,:,1:6])
@show vecnorm(Ac[1]),vecnorm(Ac[1][1:6,1:6,:])
@show vecnorm(Ac[2]),vecnorm(Ac[2][1:6,1:6,:])
@show vecnorm(Al[1]),vecnorm(Al[1][1:6,1:6,:])
