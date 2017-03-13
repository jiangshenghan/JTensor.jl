
"""
Measure nearest neighbour Heisenberg energy for spin-1/2 square model

---Au[1]---Au[2]---
   |       |   
---THl-----THr---
   |       |
---Ad[1]---Ad[2]---

legs order for Au: left,right,down
legs order for Ad: right,left,up
legs order for T: phys,left,up,right,down

"""
function square_heisenberg(Au,Ad,T)
    HS=zeros(Complex128,2,2,3)
    HS[:,:,1]=0.5*[0 1; 1 0]
    HS[:,:,2]=0.5*[0 -im; im 0]
    HS[:,:,3]=0.5*[1 0; 0 -1]

    chi=size(Au[1],1)
    D=size(T[1],2)
    DD=D^2

    TT=[reshape(jcontract([T[i],conj(T[i])],[[1,-1,-3,-5,-7],[1,-2,-4,-6,-8]]),DD,DD,DD,DD) for i=1:2]
    THl=reshape(jcontract([T[1],HS,conj(T[1])],[[1,-1,-3,-5,-8],[1,2,-6],[2,-2,-4,-7,-9]]),DD,DD,DD*3,DD)
    THr=reshape(jcontract([T[2],HS,conj(T[2])],[[1,-1,-4,-6,-8],[1,2,-2],[2,-3,-5,-7,-9]]),DD*3,DD,DD,DD)

    Gl=Gr=reshape(jcontract([eye(Complex128,chi,chi),eye(Complex128,D)],[[-1,-4],[-2,-3]]),chi,DD,chi)

    leftlm=LinearMap([Gl,Au[1],TT[1],Ad[1],Au[2],TT[2],Ad[2]],[[1,2,3],[1,6,4],[2,4,7,5],[8,3,5],[6,-1,9],[7,9,-2,10],[-3,8,10]],1)
    leig_res=eigs(leftlm,nev=1,v0=Gl[:],ncv=20,tol=1e-8)
    λl,Gl=leig_res
    λl=λl[1]
    Gl=reshape(Gl,chi,DD,chi)

    rightlm=LinearMap([Gr,Au[2],TT[2],Ad[2],Au[1],TT[1],Ad[1]],[[1,2,3],[6,1,4],[7,4,2,5],[3,8,5],[-1,6,9],[-2,9,7,10],[8,-3,10]],1)
    reig_res=eigs(rightlm,nev=1,v0=Gr[:],ncv=20,tol=1e-8)
    λr,Gr=reig_res
    λr=λr[1]
    Gr=reshape(Gr,chi,DD,chi)

    @printf("eig info: \n λl=%f+i%f \n λr=%f+i%f \n lniter=%d, lnmult=%d \n rniter=%d, rnmult=%d \n",real(λl),imag(λl),real(λr),imag(λr),leig_res[4],leig_res[5],reig_res[4],reig_res[5])

    wf_norm=λl*jcontract([Gl,Gr],[[1,2,3],[1,2,3]])
    EL=jcontract([Gl,Au[1],THl,Ad[1]],[[1,2,3],[1,-1,4],[2,4,-2,5],[-3,3,5]])
    ER=jcontract([Gr,Au[2],THr,Ad[2]],[[1,2,3],[-1,1,4],[-2,4,2,5],[3,-3,5]])
    energy=jcontract([EL,ER],[[1,2,3],[1,2,3]])/wf_norm

    @printf("wf_norm=%f+i%f\n energy=%.16f+i%e\n\n",real(wf_norm),imag(wf_norm),real(energy),imag(energy))
     flush(STDOUT)
end
