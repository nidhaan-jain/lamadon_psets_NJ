 #=
 
    TOPICS IN LABOR: EARNINGS AND EMPLOYMENT 

    PROBLEM SET 3

    NIDHAAN JAIN 

=#

########### load packages  
using LinearAlgebra
using Distributions
using GLM 

########### define intermediate functions 

# obtain the (log of) the normal pdf, given a value of wages (y), mean, and standard deviation
# this function will eventuallt require a transpose of the means and standard deviations 
function pdf_value(x, μ=0, σ=1)  
    z = -0.5 * (  (Y .- μ) ./ σ )^2   .- 0.5 * log(2.0*pi) .- log.(σ)
    return z
end

# obtain the sum of the likelihood for a given k   
function sum_exp(v)
    vm = maximum(v)
    z = (exp.(v .- vm))
    z = z[1] + z[2] + z[3]
    z = log(z) + vm
end 

# obtain the posterior for p_k 
# this function will eventually take the following arguments- 
# Y1 (vector of 1st period realisations), Y2, Y3, μ (matrix of k-specific means over time), σ (same as before, but sd), pk, n, k
τ = zeros(n, k) # here, n is the number of individuals and k is the number of latent groups 
lpm = zeros(n, k)
lik = 0 
for i in 1:n 
    ltau = log.(pk)
    lnorm1 = pdf_value(Y1[i], μ[1, :], σ[1, :])
    lnorm2 = pdf_value(Y2[i], μ[2, :], σ[2, :])
    lnorm3 = pdf_value(Y3[i], μ[3, :], σ[3, :])
    lall = ltau .+ lnorm1 .+ lnorm2 .+ lnorm3
    lpm[i, :] = lall
    lik = lik + sum_exp(lall)
    τ[i, :] = exp.(lall .- sum_exp(lall))
end 


########### update the means and variances 

DY1  = kron(Y1, ones(k))
DY2  = kron(Y2, ones(k))
DY3  = kron(Y3, ones(k))
Dkj1 = kron(speye(N),sparse(1:k,1:k))
Dkj2 = kron(speye(N),sparse(1:k,1:k))
Dkj3 = kron(speye(N),sparse(1:k,1:k))

rw = τ'
fit = wfit(lm, Dkj1, DY1, rw)
A[1,:] = coef(fit)[1:nk]'



   