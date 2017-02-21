
include("../src/JTensor.jl")
using JTensor

#tensor for square pi flux rvb
function test_square_pi_rvb_transfer_mat(χ;d=2,D=6,maxiter=100)
    TT=readdlm("./tensor_data/square_pi_flux")
    Ta=TT[:,1]
    Tb=TT[:,2]
    Ta=reshape(Ta,d,D,D,D,D)
    Tb=reshape(Tb,d,D,D,D,D)
    square_duc_dlmpofp(permutedims(Ta,[1,2,4,3,5]),permutedims(Tb,[1,2,4,3,5]),χ,maxiter=maxiter)
end

test_square_pi_rvb_transfer_mat(10,maxiter=100)
