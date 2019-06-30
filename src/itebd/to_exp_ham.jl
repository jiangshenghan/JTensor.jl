"""
Given WH (MPO for Hamiltonian), to obtain W_expH (MPO for exp(tH))
Translational symmetry is assumed

We apply algorithm presented in 1407.1832, e.g. Eq.(B1, B2)
WH is assumed to be right-up triangle
WH=[1 C D; 0 A B; 0 0 1]
W_expH=[WD WC; WB WA]
A,WA ~ N1xN2
B,WB ~ N1x1
C,WC ~ 1xN2
D,WD ~ 1x1
error for WexpH ~ t^2

WH are four leg tensors:
   |
---WH---
   |
legs order: lrud

returns W_expH
"""
#TODO: Check if WA is correct
function to_exp_H_mpo(WH,t;elemtype=Complex128)

    dp=size(WH,3)
    N1=size(WH,1)-2
    N2=size(WH,2)-2
    t=complex(t)

    A=WH[2:1+N1,2:1+N2,:,:]
    B=WH[2:1+N1,end,:,:]
    C=WH[1,2:1+N2,:,:]
    D=WH[1,end,:,:]
    @show dp,N1,N2
    #@show size(A),A
    #@show size(B),B
    #@show size(C),C
    #@show size(D),D

    bp=[0 0; 1 0]
    b=[0 1; 0 0]

    WA=zeros(elemtype,size(A))
    if vecnorm(A)!=0
        for j1=1:N1, j2=1:N2
            M=kron(A[j1,j2,:,:],bp,bp)+sqrt(t)*kron(B[j1,:,:],bp,eye(2))+sqrt*kron(C[j2,:,:],eye(2),bp)+t*kron(D,eye(2),eye(2))
            WA[j1,j2,:,:]=(kron(eye(dp),b,b)*expm(M))[1:4:end,1:4:end]
         end
     end

    WB=zeros(elemtype,size(B))
    if vecnorm(B)!=0
        for j1=1:N1
            M=sqrt(t)*kron(B[j1,:,:],bp)+t*kron(D,eye(2))
            WB[j1,:,:]=(kron(eye(dp),b)*expm(M))[1:2:end,1:2:end]
        end
    end

    WC=zeros(elemtype,size(C))
    if vecnorm(C)!=0
        for j2=1:N2
            M=sqrt(t)*kron(C[j2,:,:],bp)+t*kron(D,eye(2))
            WC[j2,:,:]=(kron(eye(dp),b)*expm(M))[1:2:end,1:2:end]
        end
    end

    WD=expm(t*D)

    W_expH=zeros(elemtype,N1+1,N2+1,dp,dp)
    W_expH[1,1,:,:]=WD
    W_expH[1,2:end,:,:]=WC
    W_expH[2:end,1,:,:]=WB
    W_expH[2:end,2:end,:,:]=WA

    return W_expH
end

"""
Another algorithm, simpler but less accurate
"""
function to_exp_H_mpo_II(WH,t;elemtype=Complex128)
    dp=size(WH,3)
    N1=size(WH,1)-2
    N2=size(WH,2)-2
    t=complex(t)

    A=WH[2:1+N1,2:1+N2,:,:]
    B=WH[2:1+N1,end,:,:]
    C=WH[1,2:1+N2,:,:]
    D=WH[1,end,:,:]
    @show dp,N1,N2

    bp=[0 0; 1 0]
    b=[0 1; 0 0]

    W_expH=zeros(elemtype,N1+1,N2+1,dp,dp)
    W_expH[1,1,:,:]=eye(dp)+t*D
    W_expH[1,2:end,:,:]=sqrt(t)*C
    W_expH[2:end,1,:,:]=sqrt(t)*B
    W_expH[2:end,2:end,:,:]=A

    return W_expH
end
