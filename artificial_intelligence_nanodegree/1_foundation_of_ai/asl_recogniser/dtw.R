x = rnorm(10)
y = rnorm(10)


dtr <- function(x, y, bound){
    n = length(x)
    m = length(y)
    diffMat = outer(x, y, '-')^2
    
    dtwMat = matrix(NA, nr = n, nc = m)
    dtwMat[1, ] = Inf
    dtwMat[, 1] = Inf
    dtwMat[1, 1] = 0 
    for(xi in seq(2, n)){
        for(yi in seq(2, m)){
            cost = diffMat[xi, yi]
            dtwMat[xi, yi] = cost + min(dtwMat[xi - 1, yi],
                                        dtwMat[xi, yi - 1],
                                        dtwMat[xi - 1, xi - 1])
        }
    }
    print(dtwMat)
    dtwMat[n, m]
}

dtr(x, y)
