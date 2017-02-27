
include("/home/jiangsb/code/JTensor.jl/example/square_peps_energy_II.jl")

#test square zero flux rvb
#/home/jiangsb/code/JTensor.jl/tensor_data/square_zero_flux
function test_square_zero_rvb_transfer_mat(file_name,χ,d=2,D=6;maxiter=300)
    Tz=readdlm(file_name)
    Tz=reshape(Tz,d,D,D,D,D)
    dl_one_vumps(permutedims(Tz,[1,2,4,3,5]),χ,maxiter=maxiter)
end

function test_square_zero_rvb_energy(file_name,x,d=2,D=6;maxiter=300)
    Tz=readdlm(file_name)
    Tz=reshape(Tz,d,D,D,D,D)
    square_peps_HeisenbergII(Tz,χ,maxiter=maxiter,err=1e-8)
end


#test square pi flux rvb
#/home/jiangsb/code/JTensor.jl/tensor_data/square_pi_flux
function test_square_pi_rvb_transfer_mat(file_name,χ,d=2,D=6;maxiter=300)
    TT=readdlm(file_name)
    Ta=TT[:,1]
    Tb=TT[:,2]
    Ta=reshape(Ta,d,D,D,D,D)
    Tb=reshape(Tb,d,D,D,D,D)
    dl_two_vumps(permutedims(Ta,[1,2,4,3,5]),permutedims(Tb,[1,2,4,3,5]),χ,maxiter=maxiter,e0=1)
end

function test_square_pi_rvb_energy(file_name,χ,d=2,D=6;maxiter=300)
    TT=readdlm(file_name)
    Ta=TT[:,1]
    Tb=TT[:,2]
    Ta=reshape(Ta,d,D,D,D,D)
    Tb=reshape(Tb,d,D,D,D,D)
    square_peps_duc_HeisenbergII(Ta,Tb,χ,maxiter=maxiter,err=1e-8)
end


#running command
#julia square_peps_test.jl file_name χ
file_name=ARGS[1]
χ=parse(Int,ARGS[2])
println(ARGS[1])
@printf("χ=%d\n\n",χ)
flush(STDOUT)
#test_square_zero_rvb_energy(file_name,χ)
test_square_pi_rvb_transfer_mat(file_name,χ)
#test_square_pi_rvb_energy(file_name,χ)
