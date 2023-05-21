N = 10 
nk = 3 
DY1 = (kronecker(rep(1,N),diag(nk)))

 rw     = c(t(tau))

   Dkj1    = as.matrix.csr(kronecker(rep(1,N),diag(nk)))
   fit    = slm.wfit(Dkj1,DY1,rw)
   A[1,]  = coef(fit)[1:nk]


   array(3*(1 + 0.8*runif(3*nk)),c(3,nk))

   rand(3, nk) .* 3 .* (1 .+ 0.8 .* rand(3, nk))