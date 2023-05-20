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
    z = (exp.(v .- vm)) # z is a vector that replaces the max. likelihood with zero 
    z = z[1] + z[2] + z[3] # replaces z with a scalar containing the sum of the likelihoods 
    z = log(z) + vm # the desired sum of the likelihoods
end 

# obtain the posterior for p_k 
# this function will eventually take the following arguments- 
# Y1 (vector of 1st period realisations), Y2, Y3, μ (matrix of k-specific means over time), σ (same as before, but sd), pk, n, k
# τ returns the posteriors (prob. that a given individual, represented by a row, belongs to a given group, represented by a column)
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
Dkj1 = kron(ones(n), Diagonal(ones(k)))
Dkj2 = kron(ones(n), Diagonal(ones(k)))
Dkj3 = kron(ones(n), Diagonal(ones(k)))

rw = reshape(τ,:) # reshape the matrix of posteriors into a vector
linear_fit = lm(Dkj1, DY1, wts = rw) # weighted least squares fit of Dkj1 and DY1 using rw weights

# recover coefficients from the linear fit 
β = coef(linear_fit)
A[1,:] = β[1:k] # intercepts

β_v = lm(Dkj1, residuals(linear_fit) .^ 2 / rw, rw)
S[1,:] = sqrt.(coef(β_v)) # standard deviations

