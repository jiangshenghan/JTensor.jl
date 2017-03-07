include("../src/JTensor.jl")
using JTensor

"""
Obtain NN Heisenberg energy for square peps with two unit cell in x direction T=[TA,TB], with legs (phys,left,up,right,down)
Notice, the physical wavefunction is still translational invariant
Alu are fixed point MPS from up half plane
Ald are fixed point MPS from lower half plane

---Alu[1]---Alu[2]---Cu[2]---Aru[1]---Aru[2]---
   ||       ||               ||       ||
===TT[1]====TT[2]============TT[1]====TT[2]===
   ||       ||               ||       ||
---Ard[1]---Ard[2]---Cd[1]---Ald[1]---Ald[2]---
we print energy for every iteration
"""
 function square_peps_duc_HeisenbergIII(T,chi;maxiter=20,err=1e-8)
     d,D=size(T[1],1,2)

     Alu=Ald=[]
     Aru=Ard=[]
     Acu=Cu=Flu=Fru=[]
     Acd=Cd=Fld=Frd=[]

     LA=rand(Complex128,chi,D,D,chi)
     RB=rand(Complex128,chi,D,D,chi)

     HS=zeros(Complex128,d,d,3)
     HS[:,:,1]=0.5*[0 1; 1 0]
     HS[:,:,2]=0.5*[0 -im; im 0]
     HS[:,:,3]=0.5*[1 0; 0 -1]

     energy=0
     uerr=derr=1

     for iter=1:maxiter
         Alu,Aru,Acu,Cu,Flu,Fru,ufe,uerr=dl_mult_vumps_seq([permutedims(T[1],[1,2,4,3,5]),permutedims(T[2],[1,2,4,3,5])],chi,Alu,Aru,Acu,Cu,Flu,Fru,e0=uerr,maxiter=1)
         Ald,Ard,Acd,Cd,Fld,Frd,dfe,derr=dl_mult_vumps_seq([permutedims(T[1],[1,4,2,5,3]),permutedims(T[2],[1,4,2,5,3])],chi,Ald,Ard,Acd,Cd,Fld,Frd,e0=derr,maxiter=1)

         leftlm=LinearMap([LA,Alu[1],T[1],conj(T[1]),Ard[1],Alu[2],T[2],conj(T[2]),Ard[2]],[[1,2,3,4],[1,10,5,6],[7,2,5,11,8],[7,3,6,12,9],[13,4,8,9],[10,-1,14,15],[16,11,14,-2,17],[16,12,15,-3,18],[-4,13,17,18]],1)
         λlA,LA=eigs(leftlm,nev=1,v0=LA[:],tol=max(err/100,uerr/100,derr/100))
         λlA=λlA[1]
         LA=reshape(LA,chi,D,D,chi)

         rightlm=LinearMap([RB,Aru[2],T[2],conj(T[2]),Ald[2],Aru[1],T[1],conj(T[1]),Ald[1]],[[1,2,3,4],[10,1,5,6],[7,11,5,2,8],[7,12,6,3,9],[4,13,8,9],[-1,10,14,15],[16,-2,14,11,17],[16,-3,15,12,18],[13,-4,17,18]],1)
         λrB,RB=eigs(rightlm,nev=1,v0=RB[:],tol=max(err/100,uerr/100,derr/100))
         λrB=λrB[1]
         RB=reshape(RB,chi,D,D,chi)

         EL=jcontract([LA,Alu[1],T[1],HS,conj(T[1]),Ard[1]],[[1,2,3,4],[1,-1,5,6],[7,2,5,-2,9],[7,8,-3],[8,3,6,-4,10],[-5,4,9,10]])
         ER=jcontract([RB,Acu[2],T[2],HS,conj(T[2]),Acd[2]],[[1,2,3,4],[-1,1,5,6],[7,-2,5,2,9],[7,8,-3],[8,-4,6,3,10],[4,-5,9,10]])
         wf_norm=(λlA*jcontract([LA,Cu[2],Cd[1],RB],[[1,2,3,4],[1,5],[6,4],[5,2,3,6]]))
         energy=jcontract([EL,ER],[[1,2,3,4,5],[1,2,3,4,5]])/wf_norm

         @printf(" iter = %d, \n λl = %f + i %f \n λr = %f + i %f \n wf_norm: %f + i %f \n energy = %.16f + i %e \n \n",iter,real(λlA),imag(λlA),real(λrB),imag(λrB),real(wf_norm),imag(wf_norm),real(energy),imag(energy))
        flush(STDOUT)
     end

     return energy
 end

