
module JTensor

export jcontract, LinearMap, polardecomp 
export sl_one_vumps, sl_two_vumps, dl_one_vumps, dl_two_vumps, dl_mult_vumps_seq

include("jcontract.jl")
include("linearmap.jl")
include("matdecomp.jl")

include("vumps/sl_one_vumps.jl")
include("vumps/sl_two_vumps.jl")
include("vumps/dl_one_vumps.jl")
include("vumps/dl_two_vumps.jl")
include("vumps/dl_mult_vumps_seq.jl")

end
