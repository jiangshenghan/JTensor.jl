# this file stores functions projecting general tensors to symmetric subspace

"""
direct sum of matrix
return A⊕ B
"""
function direct_sum(A,B)
    return [A zeros(size(A,1),size(B,2)); zeros(size(B,1),size(A,2)) B]
end

"""
direct sum of matrix
return A[1]⊕ A[2]...
"""
function direct_sum(As)
    M=As[1]
    for i=2:size(As,1) M=direct_sum(M,As[i]) end
    return M
end


"""
returns a basis for the intersection of the vector spaces spanned by the vectors in S1,S2
"""
function int_basis(S1,S2)
    S=[S1 S2]
    NS=nullspace(S)
    return qr(S1*NS[1:size(S1,2),:])[1]
end


"""
given certain spin rep (can be reducible in general), obtain spin operators
Convention: 
1. arrow=+1 means rep, return S, and arrow=-1 means conjugate rep, return -S^t
2. inside irep Si, m from +Si to -Si

return Sz,S^+ 
"""
function spin_ops(spin_rep,arrow)
    spin_rep=spin_rep*1.0
    Sz=[diagm(spin_rep[i]:-1:-spin_rep[i])*arrow for i=1:size(spin_rep,1)]

    Sp=copy(Sz)
    for i=1:size(spin_rep,1)
        spin_deg=Int(2*spin_rep[i]+1)
        Sp[i]=zeros(spin_deg,spin_deg)
        for ind=2:spin_deg
            m=spin_rep[i]-ind+1
            Sp[i][ind-1,ind]=sqrt(spin_rep[i]*(spin_rep[i]+1)-m*(m+1))
        end
        if arrow==-1 Sp[i]=-Sp[i].' end
    end

    Sz=direct_sum(Sz)
    Sp=direct_sum(Sp)

    return Sz,Sp
end


"""
spin_reps denotes representations for each legs of the tensor 
obtain spin singlet subspace M. The projector P can be constructed as P=MM'

   \    /
  --M==M'--
   /    \

return M
"""
function spin_sym_space(spin_reps,arrows)
    N=size(spin_reps,1)
    leg_dims=[Int(sum(x->2x+1,spin_reps[i])) for i=1:N]
    tot_dims=prod(leg_dims)
    legs_list=[[-i,-N-i] for i=1:N]

    Sz=Sp=zeros(leg_dims...,leg_dims...)
    for i=1:size(spin_reps,1)
        op_res=spin_ops(spin_reps[i],arrows[i])

        ztensor_list=[]
        ptensor_list=[]
        for j=1:size(spin_reps,1)
            if j==i
                push!(ztensor_list,op_res[1])
                push!(ptensor_list,op_res[2])
            else
                push!(ztensor_list,eye(leg_dims[j]))
                push!(ptensor_list,eye(leg_dims[j]))
            end
        end

        Sz+=jcontract(ztensor_list,legs_list)
        Sp+=jcontract(ptensor_list,legs_list)
    end

    M=int_basis(nullspace(reshape(Sz,tot_dims,tot_dims)),nullspace(reshape(Sp,tot_dims,tot_dims)))
    return reshape(M,leg_dims...,div(length(M),tot_dims))
end
