include("../src/JTensor.jl")
using JTensor

 """
 Obtain NN Heisenberg energy for square peps state, translational invariant tensor is assumed
 For tensor T, legs order is (phys,left,up,right,down)
 we print energy for every iteration
 """
 function square_peps_HeisenbergII(T,χ;maxiter=20,err=1e-8)
     d,D=size(T,1,2)

     Al=permutedims(reshape(qr(rand(Complex128,χ*D^2,χ))[1],χ,D,D,χ),[1,4,2,3])
     Ar=permutedims(Al,[2,1,3,4])
     Bl=Al
     Br=Ar
     Ac=CA=FlA=FrA=Bc=CB=FlB=FrB=[]

     L=rand(Complex128,χ,D,D,χ)
     R=rand(Complex128,χ,D,D,χ)

     HS=zeros(Complex128,d,d,3)
     HS[:,:,1]=0.5*[0 1; 1 0]
     HS[:,:,2]=0.5*[0 -im; im 0]
     HS[:,:,3]=0.5*[1 0; 0 -1]

     energy=0
     Aerr_mean=Berr_mean=1

     for iter=1:maxiter
         Al,Ar,Ac,CA,FlA,FrA,feA,Aerr_mean=dl_one_vumps(permutedims(T,[1,2,4,3,5]),χ,Al,Ar,Ac,CA,FlA,FrA,maxiter=1,e0=Aerr_mean)
         Bl,Br,Bc,CB,FlB,FrB,feB,Berr_mean=dl_one_vumps(permutedims(T,[1,4,2,5,3]),χ,Bl,Br,Bc,CB,FlB,FrB,maxiter=1,e0=Berr_mean)

         leftlm=LinearMap([L,Al,T,conj(T),Br],[[1,2,3,4],[1,-1,5,6],[7,2,5,-2,8],[7,3,6,-3,9],[-4,4,8,9]],1)
         rightlm=LinearMap([R,Ar,T,conj(T),Bl],[[1,2,3,4],[-1,1,5,6],[7,-2,5,2,8],[7,-3,6,3,9],[4,-4,8,9]],1)
         λl,L=eigs(leftlm,nev=1,v0=L[:],tol=max(err/100,Aerr_mean/100,Berr_mean/100))
         λr,R=eigs(rightlm,nev=1,v0=R[:],tol=max(err/100,Aerr_mean/100,Berr_mean/100))
         λl=λl[1]
         λr=λr[1]
         L=reshape(L,χ,D,D,χ)
         R=reshape(R,χ,D,D,χ)

         EL=jcontract([L,Al,T,HS,conj(T),Br],[[1,2,3,4],[1,-1,5,6],[7,2,5,-2,9],[7,8,-3],[8,3,6,-4,10],[-5,4,9,10]])
         ER=jcontract([R,Ac,T,HS,conj(T),Bc],[[1,2,3,4],[-1,1,5,6],[7,-2,5,2,9],[7,8,-3],[8,-4,6,3,10],[4,-5,9,10]])
         wf_norm=(λl*λr*jcontract([L,CA,CB,R],[[1,2,3,4],[1,5],[6,4],[5,2,3,6]]))
         energy=jcontract([EL,ER],[[1,2,3,4,5],[1,2,3,4,5]])/wf_norm

         @printf("iter = %d \n λl = %f + i %f \n λr = %f + i %f \n wf_norm: %f + i %f \n energy = %.16f + i %e \n \n",iter,real(λl),imag(λl),real(λr),imag(λr),real(wf_norm),imag(wf_norm),real(energy),imag(energy))
         flush(STDOUT)
     end

     return energy
 end


"""
Obtain NN Heisenberg energy for square peps with two unit cell in x direction TA and TB, with legs (phys,left,up,right,down)
Notice, the physical wavefunction is still translational invariant
Alu,Blu,... are fixed point MPS from up half plane
Ald,Bld,... are fixed point MPS from lower half plane

---Alu---Blu---C2u---Aru---Bru---
   ||    ||          ||    ||
===TTA===TTB=========TTA===TTB===
   ||    ||          ||    ||
---Ard---Brd---C1d---Ald---Bld---
we print energy for every iteration
"""
 function square_peps_duc_HeisenbergII(TA,TB,χ;maxiter=20,err=1e-8)
     d,D=size(TA,1,2)

     Alu=Blu=Ald=Bld=permutedims(reshape(qr(rand(Complex128,χ*D^2,χ))[1],χ,D,D,χ),[1,4,2,3])
     Aru=Bru=Ard=Brd=permutedims(Alu,[2,1,3,4])
     Acu=Bcu=C1u=C2u=FAlu=FAru=FBlu=FBru=[]
     Acd=Bcd=C1d=C2d=FAld=FArd=FBld=FBrd=[]

     LA=rand(Complex128,χ,D,D,χ)
     RB=rand(Complex128,χ,D,D,χ)

     HS=zeros(Complex128,d,d,3)
     HS[:,:,1]=0.5*[0 1; 1 0]
     HS[:,:,2]=0.5*[0 -im; im 0]
     HS[:,:,3]=0.5*[1 0; 0 -1]

     energy=0
     uerr_mean=derr_mean=1

     for iter=1:maxiter
         Alu,Aru,Acu,Blu,Bru,Bcu,C1u,C2u,FAlu,FAru,FBlu,FBru,ufe,uerr_mean=dl_two_vumps(permutedims(TA,[1,2,4,3,5]),permutedims(TB,[1,2,4,3,5]),χ,Alu,Aru,Blu,Bru,Acu,Bcu,C1u,C2u,FAlu,FAru,FBlu,FBru,e0=uerr_mean,maxiter=1)
         Ald,Ard,Acd,Bld,Brd,Bcd,C1d,C2d,FAld,FArd,FBld,FBrd,dfe,derr_mean=dl_two_vumps(permutedims(TA,[1,4,2,5,3]),permutedims(TB,[1,4,2,5,3]),χ,Ald,Ard,Bld,Brd,Acd,Bcd,C1d,C2d,FAld,FArd,FBld,FBrd,e0=derr_mean,maxiter=1)

         #generate the fixed point tensor from lower half-plane by symmetry, where we assume the symmetry transforms trivially
         #println("symmetry!")
         #println()
         ##D=3 case
         #if D==3
         #    W=[0 1 0; -1 0 0; 0 0 1]
         #    JW=[0 -1 0; 1 0 0; 0 0 1]
         #end
         ##D=6(0⊕1/2⊕1) case
         #if D==6
         #    W=zeros(6,6)
         #    W[6,1]=W[3,3]=W[2,5]=W[1,6]=1
         #    W[5,2]=W[4,4]=-1
         #    J=diagm([1,-1,1,1,-1,1])
         #    JW=J*W
         #end
         #for zero flux rvb
         #Ald,Ard,Acd,Bld,Brd,Bcd,C1d,C2d=jcontract([Alu,W,W],[[-1,-2,1,2],[1,-3],[2,-4]]),jcontract([Aru,W,W],[[-1,-2,1,2],[1,-3],[2,-4]]),jcontract([Acu,W,W],[[-1,-2,1,2],[1,-3],[2,-4]]),jcontract([Blu,JW,JW],[[-1,-2,1,2],[1,-3],[2,-4]]),jcontract([Bru,JW,JW],[[-1,-2,1,2],[1,-3],[2,-4]]),jcontract([Bcu,JW,JW],[[-1,-2,1,2],[1,-3],[2,-4]]),C1u,C2u
         #for pi flux rvb
         #Ald,Ard,Acd,Bld,Brd,Bcd,C1d,C2d=jcontract([Alu,W,W],[[-1,-2,1,2],[1,-3],[2,-4]]),jcontract([Aru,W,W],[[-1,-2,1,2],[1,-3],[2,-4]]),jcontract([Acu,W,W],[[-1,-2,1,2],[1,-3],[2,-4]]),jcontract([Blu,W,W],[[-1,-2,1,2],[1,-3],[2,-4]]),jcontract([Bru,W,W],[[-1,-2,1,2],[1,-3],[2,-4]]),jcontract([Bcu,W,W],[[-1,-2,1,2],[1,-3],[2,-4]]),C1u,C2u

         leftlm=LinearMap([LA,Alu,TA,conj(TA),Ard,Blu,TB,conj(TB),Brd],[[1,2,3,4],[1,10,5,6],[7,2,5,11,8],[7,3,6,12,9],[13,4,8,9],[10,-1,14,15],[16,11,14,-2,17],[16,12,15,-3,18],[-4,13,17,18]],1)
         λlA,LA=eigs(leftlm,nev=1,v0=LA[:],tol=max(err/100,uerr_mean/100,derr_mean/100))
         λlA=λlA[1]
         LA=reshape(LA,χ,D,D,χ)

         rightlm=LinearMap([RB,Bru,TB,conj(TB),Bld,Aru,TA,conj(TA),Ald],[[1,2,3,4],[10,1,5,6],[7,11,5,2,8],[7,12,6,3,9],[4,13,8,9],[-1,10,14,15],[16,-2,14,11,17],[16,-3,15,12,18],[13,-4,17,18]],1)
         λrB,RB=eigs(rightlm,nev=1,v0=RB[:],tol=max(err/100,uerr_mean/100,derr_mean/100))
         λrB=λrB[1]
         RB=reshape(RB,χ,D,D,χ)

         EL=jcontract([LA,Alu,TA,HS,conj(TA),Ard],[[1,2,3,4],[1,-1,5,6],[7,2,5,-2,9],[7,8,-3],[8,3,6,-4,10],[-5,4,9,10]])
         ER=jcontract([RB,Bcu,TB,HS,conj(TB),Bcd],[[1,2,3,4],[-1,1,5,6],[7,-2,5,2,9],[7,8,-3],[8,-4,6,3,10],[4,-5,9,10]])
         wf_norm=(λlA*jcontract([LA,C2u,C1d,RB],[[1,2,3,4],[1,5],[6,4],[5,2,3,6]]))
         energy=jcontract([EL,ER],[[1,2,3,4,5],[1,2,3,4,5]])/wf_norm

         #EL2=jcontract([LA,Acu,TA,HS,conj(TA),Acd],[[1,2,3,4],[1,-1,5,6],[7,2,5,-2,9],[7,8,-3],[8,3,6,-4,10],[-5,4,9,10]])
         #ER2=jcontract([RB,Bru,TB,HS,conj(TB),Bld],[[1,2,3,4],[-1,1,5,6],[7,-2,5,2,9],[7,8,-3],[8,-4,6,3,10],[4,-5,9,10]])
         #energy2=jcontract([EL2,ER2],[[1,2,3,4,5],[1,2,3,4,5]])/wf_norm

         @printf(" iter = %d, \n λl = %f + i %f \n λr = %f + i %f \n wf_norm: %f + i %f \n energy = %.16f + i %e \n \n",iter,real(λlA),imag(λlA),real(λrB),imag(λrB),real(wf_norm),imag(wf_norm),real(energy),imag(energy))
        flush(STDOUT)
     end

     return energy
 end
