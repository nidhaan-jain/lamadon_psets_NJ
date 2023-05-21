 #=================================================
    TOPICS IN LABOR: EARNINGS AND EMPLOYMENT 

    PROBLEM SET 3

    NIDHAAN JAIN 

    SECTIONS: 
    1. DEFINE EM FUNCTION AND INTERMEDIATE FUNCTIONS 
    2. EVALUATE H AND Q FUNCTIONS 
    3. SIMULATE DATA 
    4. APPLICATION TO PSID DATA 
    5. APPLICATION WITH AUTOCORRELATION 

===================================================#

########### load packages  
using LinearAlgebra
using Distributions
using GLM 
using Random 
using Plots
using DataFrames
using StatFiles 


########### set script parameters 
Random.seed!(2324) # set seed for reproducibility
n = 1000 # number of individuals 
k = 3 # number of latent groups 
sd_scale = 1 # scale of the standard deviation used in simulating data 

########### 1. define EM function and intermediate functions 

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


# write function to compute the parameters 
function compute_parameters(Y1, Y2, Y3, μ, σ, pk, n, k, tol)

error = 9999

while error > tol 


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

    ########### 2. insert checks for the H and Q functions 

    # Q function
    Q1 = sum(tau_last * lpm_last) 
    Q2 = sum(tau_last * lpm)

    # H function 
    H1 = -sum(tau_last * log.(tau_last))
    H2 = -sum(tau_last * log.(tau))

    error = abs(lik - lik_last) # error in the likelihood function 

    return(β)
    return(β_v)
    return(τ)


end # end the loop 

end # end the function 


########### 3. simulate data 
a_vector = 0.8 .* rand(3 * k) # vector of random numbers from uniform distribution 
a_vector = 3 .* (1 .+ a_vector) 
A_master = reshape(a_vector, 3, k)

S_master = ones(3, k)

pk_master = rand(Dirichlet(ones(k)), 1)
pk_master_vector = vec(pk_master)

y1 = zeros(n)
y2 = zeros(n)
y3 = zeros(n)
K = zeros(n)

k_types = collect(1:k)

K = sample(k_types, Weights(pk_master_vector), 1000, replace = true) # wv= pk_master

for i in 1:n
    k_element = K[i]
    y1[i] = A_master[1, k_element] + S_master[1, k_element] * rand(Normal(0, 1))*sd_scale
    y2[i] = A_master[2, k_element] + S_master[2, k_element] * rand(Normal(0, 1))*sd_scale
    y3[i] = A_master[3, k_element] + S_master[3, k_element] * rand(Normal(0, 1))*sd_scale
end 

df = DataFrame(y1 = y1, y2 = y2, y3 = y3, k = K)


# apply the EM algorithm to this data 

y1 = vec(df.y1)
y2 = vec(df.y2)
y3 = vec(df.y3)

parameters = compute_parameters(y1, y2, y3, A_master, S_master, pk_master_vector, n, k, 0.01)


########### 4. apply to PSID 

df_psid = DataFrame(load("/Users/nidhaanjit/Dropbox/PhD coursework/02 Topics in Labor_Earnings and Employment/psets/lamadon_psets_NJ/pset 3/AER_2012_1549_data/output/data4estimation.dta"))

df_psid = dropmissing!(df_psid, [:log_y, :year, :marit, :state_st])

# obtain residuals 
model = lm(@formula(log_y ~ year + marit + state_st), df_psid)
df_psid[:, :log_yr] = residuals(model)

# create lagged values 
df_psid = sort!(df_psid, [:id, :year])
df_psid[:, :log_yr_11] = lag(df_psid.log_yr, 1)
df_psid[:, :log_yr_12] = lag(df_psid.log_yr, 2)

# assign vector values so we have the same notation as before 
y1_psid = vec(df_psid.log_yr_12)  
y2_psid = vec(df_psid.log_yr_11)
y3_psid = vec(df_psid.log_yr)

# apply the EM algorithm to this data with different numbers of mixtures 
psid_computation_3 = compute_parameters(y1_psid, y2_psid, y3_psid, A_master, S_master, pk_master_vector, n, 3, 0.01)
psid_computation_4 = compute_parameters(y1_psid, y2_psid, y3_psid, A_master, S_master, pk_master_vector, n, 4, 0.01)
psid_computation_5 = compute_parameters(y1_psid, y2_psid, y3_psid, A_master, S_master, pk_master_vector, n, 5, 0.01)

########### 4. estimate the model with autocorrelation 

ρ = 0.6 

# obtain four lags 
df_psid[:, :log_yr_11] = lag(df_psid.log_yr, 1)
df_psid[:, :log_yr_12] = lag(df_psid.log_yr, 2)
df_psid[:, :log_yr_13] = lag(df_psid.log_yr, 3)

# account for autocorrelation 
y1_psid = vec(df_psid.log_yr_13)  
y2_psid = vec(df_psid.log_yr_12)
y3_psid = vec(df_psid.log_yr_13)
y4_psid = vec(df_psid.log_yr)

y2_psid = y2_psid .- ρ .* y1_psid
y3_psid = y3_psid .- ρ .* y2_psid
y4_psid = y4_psid .- ρ .* y3_psid

# estimate the parameters, but now account for the fact that we're using 4 observations 

error = 9999

while error > tol 


    # obtain the posterior for p_k 
    # this function will eventually take the following arguments- 
    # Y1 (vector of 1st period realisations), Y2, Y3, μ (matrix of k-specific means over time), σ (same as before, but sd), pk, n, k
    # τ returns the posteriors (prob. that a given individual, represented by a row, belongs to a given group, represented by a column)
    τ = zeros(n, k) # here, n is the number of individuals and k is the number of latent groups 
    lpm = zeros(n, k)
    lik = 0 
    for i in 1:n 
        ltau = log.(pk)
        lnorm1 = pdf_value(y1_psid[i], μ[1, :], σ[1, :])
        lnorm2 = pdf_value(y2_psid[i], μ[2, :], σ[2, :])
        lnorm3 = pdf_value(y3_psid[i], μ[3, :], σ[3, :])
        lall = ltau .+ lnorm1 .+ lnorm2 .+ lnorm3 .+ lnorm4 
        lpm[i, :] = lall
        lik = lik + sum_exp(lall)
        τ[i, :] = exp.(lall .- sum_exp(lall))
    end 


    ########### update the means and variances 

    DY1  = kron(y1_psid, ones(k))
    DY2  = kron(y2_psid, ones(k))
    DY3  = kron(y3_psid, ones(k))
    DY4  = kron(y4_psid, ones(k))
    Dkj1 = kron(ones(n), Diagonal(ones(k)))
    Dkj2 = kron(ones(n), Diagonal(ones(k)))
    Dkj3 = kron(ones(n), Diagonal(ones(k)))
    Dkj4 = kron(ones(n), Diagonal(ones(k)))

    rw = reshape(τ,:) # reshape the matrix of posteriors into a vector
    linear_fit = lm(Dkj1, DY1, wts = rw) # weighted least squares fit of Dkj1 and DY1 using rw weights

    # recover coefficients from the linear fit 
    β = coef(linear_fit)
    A[1,:] = β[1:k] # intercepts

    β_v = lm(Dkj1, residuals(linear_fit) .^ 2 / rw, rw)
    S[1,:] = sqrt.(coef(β_v)) # standard deviations

    ########### 2. insert checks for the H and Q functions 

    # Q function
    Q1 = sum(tau_last * lpm_last) 
    Q2 = sum(tau_last * lpm)

    # H function 
    H1 = -sum(tau_last * log.(tau_last))
    H2 = -sum(tau_last * log.(tau))

    error = abs(lik - lik_last) # error in the likelihood function 

    return(β)
    return(β_v)
    return(τ)


end # end the loop 


















