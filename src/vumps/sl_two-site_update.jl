#This file stores functions for two-site update

"""
Obtain updated A2c for magnetic translational symmetric systems

   --Jc--Al--Jc--Ac--Jc--
  /      |       |       \          ---A2c---
Fl-------T-------T-------Fr  =         / \
  \      |       |       /
   --                  --

legs order for A2c: left,right,dl,dr

return A2c
"""
function mag_trans_A2c(T,Fl,Fr,Al,Ac,Jc)
    return jcontract([Fl,Jc,Al,T,Jc,Ac,T,Jc,Fr],[[1,2,-1],[1,3],[3,5,4],[2,6,4,-3],[5,7],[7,9,8],[6,10,8,-4],[9,11],[11,10,-2]])
end
