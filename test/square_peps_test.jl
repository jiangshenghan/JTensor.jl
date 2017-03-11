
include("/home/jiangsb/code/JTensor.jl/example/square_peps_energy_II.jl")
#include("/home/jiangsb/code/JTensor.jl/example/square_peps_energy_III.jl")

#test square zero flux rvb
#/home/jiangsb/code/JTensor.jl/tensor_data/square_zero_flux
function test_square_zero_rvb_transfer_mat(file_name,chi,d=2,D=6;maxiter=300)
    Tz=readdlm(file_name)
    Tz=reshape(Tz,d,D,D,D,D)
    dl_one_vumps(permutedims(Tz,[1,2,4,3,5]),chi,maxiter=maxiter)
end

function test_square_zero_rvb_energy(file_name,x,d=2,D=6;maxiter=300)
    Tz=readdlm(file_name)
    Tz=reshape(Tz,d,D,D,D,D)
    square_peps_HeisenbergII(Tz,chi,maxiter=maxiter,err=1e-8)
end


#test square pi flux rvb
#/home/jiangsb/code/JTensor.jl/tensor_data/square_pi_flux
function test_square_pi_rvb_transfer_mat(file_name,chi,d=2,D=6;maxiter=300)
    TT=readdlm(file_name)
    Ta=TT[:,1]
    Tb=TT[:,2]
    Ta=reshape(Ta,d,D,D,D,D)
    Tb=reshape(Tb,d,D,D,D,D)
    #dl_two_vumps(permutedims(Ta,[1,2,4,3,5]),permutedims(Tb,[1,2,4,3,5]),chi,maxiter=maxiter,e0=1)
    dl_mult_vumps_seq([permutedims(Ta,[1,2,4,3,5]),permutedims(Tb,[1,2,4,3,5])],chi,maxiter=maxiter,e0=1)
end

function test_square_pi_rvb_energy(file_name,chi,d=2,D=6;maxiter=300)
    TT=readdlm(file_name)
    Ta=TT[:,1]
    Tb=TT[:,2]
    Ta=reshape(Ta,d,D,D,D,D)
    Tb=reshape(Tb,d,D,D,D,D)
    square_peps_duc_HeisenbergII(Ta,Tb,chi,maxiter=maxiter,err=1e-8)
    #square_peps_duc_HeisenbergIII([Ta,Tb],chi,maxiter=maxiter,err=1e-12)
end


#running command
#julia square_peps_test.jl file_name chi
file_name=ARGS[1]
chi=parse(Int,ARGS[2])
println(ARGS[1])
@printf("chi=%d\n\n",chi)
flush(STDOUT)
#test_square_zero_rvb_energy(file_name,chi)
#test_square_pi_rvb_transfer_mat(file_name,chi)
test_square_pi_rvb_energy(file_name,chi)
