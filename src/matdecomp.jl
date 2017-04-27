# this file stores multiple methods for matrix decomposition

"""
polar decomposition X=U*A with A positive semidefinite
return U,A
"""
function polardecomp(X)
    U,S,V=svd(X)
    return U*V',V*diagm(S)*V'
end

"""
positive qr decomposition
return Q,R
"""
function posqr(X)
    Q,R=qr(X)
    s=diagm(sign(diag(R)));
    Q=Q*s
    R=s*R
    return Q,R
end


