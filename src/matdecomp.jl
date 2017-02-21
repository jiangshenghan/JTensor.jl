# this file stores multiple methods for matrix decomposition

"""
polar decomposition X=U*A with A positive semidefinite
"""
function polardecomp(X)
    U,S,V=svd(X)
    return U*V',V*diagm(S)*V'
end
