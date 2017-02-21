
include("/home/jiangsb/code/JTensor.jl/src/JTensor.jl")
using JTensor

#test square zero flux rvb
#/home/jiangsb/code/JTensor.jl/tensor_data/square_zero_flux
function test_square_zero_rvb_transfer_mat(file_name,χ,d=2,D=6;maxiter=300)
    Tz=readdlm(file_name)
    Tz=reshape(Tz,d,D,D,D,D)
    square_dlmpofp(permutedims(Tz,[1,2,4,3,5]),χ,maxiter=maxiter)
end


#test square pi flux rvb
#/home/jiangsb/code/JTensor.jl/tensor_data/square_pi_flux
function test_square_pi_rvb_transfer_mat(file_name,χ,d=2,D=6;maxiter=300)
    TT=readdlm(file_name)
    Ta=TT[:,1]
    Tb=TT[:,2]
    Ta=reshape(Ta,d,D,D,D,D)
    Tb=reshape(Tb,d,D,D,D,D)
    square_duc_dlmpofp(permutedims(Ta,[1,2,4,3,5]),permutedims(Tb,[1,2,4,3,5]),χ,maxiter=maxiter)
end

#julia transfer_matrix file_name χ
file_name=ARGS[1]
χ=parse(Int,ARGS[2])
println(ARGS[1])
@printf("χ=%d\n\n",χ)
test_square_pi_rvb_transfer_mat(file_name,χ)
