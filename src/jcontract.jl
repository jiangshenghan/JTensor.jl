
# function tensor= jcontract(tensor_list,leg_list)
"""
Contracts a single tensor network.
tensor_list is an array of tensors, which stores tensors to be contracted in chronological order
we associate a graph with the contractions and label the edges to be contracted with positive integers and the outgoing ones with negative numbers; 

legs_list is an array with vectors labelling legs of tensor
the output tensor is sorted according to negative legs (-1,-2,-3)
"""
function jcontract(tensor_list,legs_list)

    #TODO:contract several legs of a single tensor

    #initialization
    result_tensor=tensor_list[1]
    result_legs=legs_list[1]
    result_dims=collect(size(result_tensor))

    for k=2:size(tensor_list,1)
        #special case: the previous contraction gives a scalar, namely, result tensor is a number
        if result_legs==[]
            result_tensor=result_tensor*tensor_list[k]
            result_legs=legs_list[k]
            result_dims=collect(size(tensor_list[k]))
            continue
        end

        curr_tensor=tensor_list[k]
        curr_legs=legs_list[k]
        curr_dims=collect(size(curr_tensor))

        comm_legs=intersect(result_legs,curr_legs)
        result_comm_inds=map(y->findfirst(x->x==y,result_legs),comm_legs)
        result_diff_inds=setdiff(1:ndims(result_tensor),result_comm_inds)
        curr_comm_inds=map(y->findfirst(x->x==y,curr_legs),comm_legs)
        curr_diff_inds=setdiff(1:ndims(curr_tensor),curr_comm_inds)

        #@show k
        #println("comm_legs: ",comm_legs)
        #println("result_comm_inds: ",result_comm_inds)
        #println("result_diff_inds: ",result_diff_inds)
        #println("curr_comm_inds: ",curr_comm_inds)
        #println("curr_diff_inds: ",curr_diff_inds)

        result_tensor=permutedims(result_tensor,[result_diff_inds;result_comm_inds])
        result_tensor=reshape(result_tensor,prod(result_dims[result_diff_inds]),prod(result_dims[result_comm_inds]))
        curr_tensor=permutedims(curr_tensor,[curr_comm_inds;curr_diff_inds])
        curr_tensor=reshape(curr_tensor,prod(curr_dims[curr_comm_inds]),prod(curr_dims[curr_diff_inds]))

        result_tensor*=curr_tensor
        result_legs=[result_legs[result_diff_inds];curr_legs[curr_diff_inds]]
        result_dims=[result_dims[result_diff_inds];curr_dims[curr_diff_inds]]

        #the case where result_tensor is a scalar
        if result_legs==[] result_tensor=result_tensor[1]; continue end

        result_tensor=reshape(result_tensor,result_dims...)


        #println("result_legs: ",result_legs)
        #println("result_dims: ", result_dims)
        #println("size(result_tensor): ",size(result_tensor))
        #println("result_tensor: ",result_tensor)
    end

    #the case for scalar without negative legs, return a number
    if (result_legs==[]) return result_tensor[1] end

    #reorder the indices of result tensor according to negative legs
    final_order=sortperm(-result_legs)
    result_tensor=permutedims(result_tensor,final_order);
    #println("final_order: ",final_order)

    return result_tensor
end
