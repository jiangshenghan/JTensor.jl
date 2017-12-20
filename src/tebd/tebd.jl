"""
Apply time evolution operator for two site Hamiltonian on MPS

Time evolution U(j,j+1;δt/2)~exp(-i H(j,j+1)δt/2)=V.exp(...).V^†

Sweep from j=1 to L-1, and then sweep back from j=L-1 to 1

---A[j]---A[j+1]---
   |      |           => Contract, svd, truncate
  U(j,j+1;δt/2)
   |      |

A1 and AL are also stored as a three leg tensor for convenience
legs order of A: lrp
U[j]=U(j,j+1,δt/2)
legs order of U: lu,ru,ld,rd

chi is the maximal bond dimension while eps is the minimum value singular value we kept. We normalize the largest singular value to be 1

return (A)
"""
#TODO: Test more on this function!
function tebd_sweep(A,U,δt,tf;chi=200,eps=1e-8)
    L=size(A,1)
    dp=size(U[1],1)
    t=0

    while t<tf
        @show t
        #sweep from left to right
        for j=1:L-1
            Dl=size(A[j],1)
            Dr=size(A[j+1],2)
            Tj=jcontract([A[j],A[j+1],U[j]],[[-1,1,2],[1,-3,3],[2,3,-2,-4]])

            #svd and drop off small singular value
            Tj_svd=svdfact(reshape(Tj,Dl*dp,Dr*dp))
            svals=Tj_svd[:S]/Tj_svd[:S][1]
            DD=min(findfirst(x->x<eps,svals)-1,chi)
            if DD<0 DD=min(chi,size(svals,1)) end
            @show j,svals[1:DD]

            #update MPS tensor, with center move from j to j+1
            A[j]=permutedims(reshape(Tj_svd[:U][:,1:DD],Dl,dp,DD),[1,3,2])
            A[j+1]=reshape(diagm(svals[1:DD])*Tj_svd[:Vt][1:DD,:],DD,Dr,dp)
        end

        #sweep from right to left
        for j=L-1:-1:1
            Dl=size(A[j],1)
            Dr=size(A[j+1],2)
            Tj=jcontract([A[j],A[j+1],U[j]],[[-1,1,2],[1,-3,3],[2,3,-2,-4]])

            #svd and drop off small singular value
            Tj_svd=svdfact(reshape(Tj,Dl*dp,Dr*dp))
            svals=Tj_svd[:S]/Tj_svd[:S][1]
            DD=min(findfirst(x->x<eps,svals)-1,chi)
            if DD<0 DD=min(chi,size(svals,1)) end
            @show j,svals[1:DD]

            #update MPS tensor, with center move from j+1 to j
            A[j]=permutedims(reshape(Tj_svd[:U][:,1:DD]*diagm(svals[1:DD]),Dl,dp,DD),[1,3,2])
            A[j+1]=reshape(Tj_svd[:Vt][1:DD,:],DD,Dr,dp)
        end
        t=t+δt
    end

    return A
end



"""
Apply time evolution operator with δt for two site Hamiltonian on MPS

Time evolution divided to even and odd group Uo[j]~U(2j-1,j) and Ue[j]~U(2j,2j+1)

Apply U(δt)~Uo(δt/2)Ue(δt)Uo(δt/2)+O(δt^3)

The input and output MPS is always set to be canonical form, with diagonal B[j] stores singular value

-B[j-1]-A[j]-B[j]-A[j+1]-B[j+1]-
        |         |               => Contract to T tensor, svd, truncate
        U(j,j+1;δt)
        |         |

A1 and AL are also stored as a three leg tensor for convenience
legs order of A: lrp
legs order of B: lr
Uo evolve δt/2 while Ue evolve δt
legs order of U: lu,ru,ld,rd
legs order of T: l,ld,r,rd

chi is the maximal bond dimension while eps is the minimum value singular value we kept. We normalize the largest singular value to be 1

return (A,B)
"""
function tebd_even_odd_one_step(A,B,Ue,Uo;chi=200,eps=1e-8)
    L=size(A,1)
    dp=size(Ue[1],1)

    #apply Uo(δt/2)
    for k=1:div(L,2)
        j=2k-1
        Dl=size(A[j],1)
        Dr=size(A[j+1],1)

        #construct Tj
        if j==1
            Tj=jcontract([A[j],B[j],A[j+1],B[j+1],Uo[k]],[[-1,2,3],[2,4],[4,5,6],[5,-3],[3,6,-2,-4]])
        elseif j==L-1
            Tj=jcontract([B[j-1],A[j],B[j],A[j+1],Uo[k]],[[-1,1],[1,2,3],[2,4],[4,-3,6],[3,6,-2,-4]])
        else
            Tj=jcontract([B[j-1],A[j],B[j],A[j+1],B[j+1],Uo[k]],[[-1,1],[1,2,3],[2,4],[4,5,6],[5,-3],[3,6,-2,-4]])
        end

        #svd and drop off small singular value
        Tj_svd=svdfact(reshape(Tj,Dl*dp,Dr*dp))
        svals=Tj_svd[:S]/Tj_svd[:S][1]
        DD=min(findfirst(x->x<eps,svals)-1,chi)
        if DD<0 DD=min(chi,size(svals,1)) end
        @show j,DD,svals[1:DD]

        #update A[j], B[j], A[j+1]
        B[j]=diagm(svals[1:DD])

        if j==1 A[j]=Tj_svd[:U][:,1:DD]
        else A[j]=inv(B[j-1])*Tj_svd[:U][:,1:DD] end
        A[j]=permutedims(reshape(A[j],Dl,dp,DD),[1,3,2])

        if j==L-1 A[j+1]=Tj_svd[:Vt][1:DD,:]
        else A[j+1]=Tj_svd[:Vt][1:DD,:]*inv(B[j+1]) end
        A[j+1]=reshape(A[j+1],DD,Dr,dp)
    end

    #apply Ue(δt)
    for k=1:div(L-1,2)
        j=2k
        Dl=size(A[j],1)
        Dr=size(A[j+1],1)

        #construct Tj
        if j==L-1
            Tj=jcontract([B[j-1],A[j],B[j],A[j+1],Ue[k]],[[-1,1],[1,2,3],[2,4],[4,-3,6],[3,6,-2,-4]])
        else
            Tj=jcontract([B[j-1],A[j],B[j],A[j+1],B[j+1],Ue[k]],[[-1,1],[1,2,3],[2,4],[4,5,6],[5,-3],[3,6,-2,-4]])
        end

        #svd and drop off small singular value
        Tj_svd=svdfact(reshape(Tj,Dl*dp,Dr*dp))
        svals=Tj_svd[:S]/Tj_svd[:S][1]
        DD=min(findfirst(x->x<eps,svals)-1,chi)
        if DD<0 DD=min(chi,size(svals,1)) end
        @show j,DD,svals[1:DD]

        #update A[j], B[j], A[j+1]
        B[j]=diagm(svals[1:DD])

        A[j]=inv(B[j-1])*Tj_svd[:U][:,1:DD]
        A[j]=permutedims(reshape(A[j],Dl,dp,DD),[1,3,2])

        if j==L-1 A[j+1]=Tj_svd[:Vt][1:DD,:]
        else A[j+1]=Tj_svd[:Vt][1:DD,:]*inv(B[j+1]) end
        A[j+1]=reshape(A[j+1],DD,Dr,dp)
    end

    #apply Uo(δt/2)
    for k=1:div(L,2)
        j=2k-1
        Dl=size(A[j],1)
        Dr=size(A[j+1],1)

        #construct Tj
        if j==1
            Tj=jcontract([A[j],B[j],A[j+1],B[j+1],Uo[k]],[[-1,2,3],[2,4],[4,5,6],[5,-3],[3,6,-2,-4]])
        elseif j==L-1
            Tj=jcontract([B[j-1],A[j],B[j],A[j+1],Uo[k]],[[-1,1],[1,2,3],[2,4],[4,-3,6],[3,6,-2,-4]])
        else
            Tj=jcontract([B[j-1],A[j],B[j],A[j+1],B[j+1],Uo[k]],[[-1,1],[1,2,3],[2,4],[4,5,6],[5,-3],[3,6,-2,-4]])
        end

        #svd and drop off small singular value
        Tj_svd=svdfact(reshape(Tj,Dl*dp,Dr*dp))
        svals=Tj_svd[:S]/Tj_svd[:S][1]
        DD=min(findfirst(x->x<eps,svals)-1,chi)
        if DD<0 DD=min(chi,size(svals,1)) end
        @show j,DD,svals[1:DD]

        #update A[j], B[j], A[j+1]
        B[j]=diagm(svals[1:DD])

        if j==1 A[j]=Tj_svd[:U][:,1:DD]
        else A[j]=inv(B[j-1])*Tj_svd[:U][:,1:DD] end
        A[j]=permutedims(reshape(A[j],Dl,dp,DD),[1,3,2])

        if j==L-1 A[j+1]=Tj_svd[:Vt][1:DD,:]
        else A[j+1]=Tj_svd[:Vt][1:DD,:]*inv(B[j+1]) end
        A[j+1]=reshape(A[j+1],DD,Dr,dp)
    end
end

