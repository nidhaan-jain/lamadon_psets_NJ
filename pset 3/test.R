tau = array(0,c(N,nk))
   lpm = array(0,c(N,nk))
   lik = 0
   for (i in 1:N) {
      ltau = log(pk)
      lnorm1 = lognormpdf(Y1[i], A[1,], S[1,])
      lnorm2 = lognormpdf(Y2[i], A[2,], S[2,])
      lnorm3 = lognormpdf(Y3[i], A[3,], S[3,])
      lall = ltau + lnorm2 + lnorm1 +lrnorm3
      lpm[i,] = lall
      lik = lik + logsumexp(lall)
      tau[i,] = exp(lall - logsumexp(lall))
   }
