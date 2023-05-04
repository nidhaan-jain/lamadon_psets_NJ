DY1  = as.matrix(kronecker(Y1 ,rep(1,nk)))

 rw     = c(t(tau))

   Dkj1    = as.matrix.csr(kronecker(rep(1,N),diag(nk)))
   fit    = slm.wfit(Dkj1,DY1,rw)
   A[1,]  = coef(fit)[1:nk]