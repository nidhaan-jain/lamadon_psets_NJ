/* This file calculates the after tax elasticities.
   
   Notes
   =====
   The nature of the exercise: Given the estimated Frisch Elasticities, hold constant 
   pi and s per household, and check the response to wage shocks if taxes were not 
   progressive i.e. (tau==0). 
	   
*/

clear all
cd "$output"
local nreps 220		/* Set the same as in parameter file T4C1 used to generate this directory */
local bskeep 201	/* Set the same as in parameter file T4C1 used to generate this directory + 1 */
local counter=`nreps'+1

*** Define the MATA code that calculates invQ*R as a void function that can be called in the loop.
mata:
void calc_invQR()
{
	q= st_data((1,.),("l2_q"))
	tau= st_data((1,.),("tau"))

	vpsi_c_lam	= st_local("psi_c_lam")
	vpsi_c_w1	= st_local("psi_c_w1")
	vpsi_c_w2	= st_local("psi_c_w2")
	vpsi_h1_lam	= st_local("psi_h1_lam")
	vpsi_h1_w1	= st_local("psi_h1_w1")
	vpsi_h1_w2	= st_local("psi_h1_w2")
	vpsi_h2_lam	= st_local("psi_h2_lam")
	vpsi_h2_w1	= st_local("psi_h2_w1")
	vpsi_h2_w2	= st_local("psi_h2_w2")
	
	psi_c_lam	= st_data((1,.),(vpsi_c_lam))
	psi_c_w1	= st_data((1,.),(vpsi_c_w1))
	psi_c_w2	= st_data((1,.),(vpsi_c_w2))
	psi_h1_lam	= st_data((1,.),(vpsi_h1_lam))
	psi_h1_w1	= st_data((1,.),(vpsi_h1_w1))
	psi_h1_w2	= st_data((1,.),(vpsi_h1_w2))
	psi_h2_lam	= st_data((1,.),(vpsi_h2_lam))
	psi_h2_w1	= st_data((1,.),(vpsi_h2_w1))
	psi_h2_w2	= st_data((1,.),(vpsi_h2_w2))

	eta_c_p		= strtoreal(st_local("eta_c_p"))
	eta_c_w1	= strtoreal(st_local("eta_c_w1"))
	eta_h1_w1	= strtoreal(st_local("eta_h1_w1"))
	eta_c_w2	= strtoreal(st_local("eta_c_w2"))
	eta_h1_w2	= strtoreal(st_local("eta_h1_w2"))
	eta_h2_w2	= strtoreal(st_local("eta_h2_w2"))
	eta_h1_p	= strtoreal(st_local("eta_h1_p"))
	eta_h2_p	= strtoreal(st_local("eta_h2_p"))
	eta_h2_w1 	= strtoreal(st_local("eta_h2_w1"))

	Q=J(3,3,0)
	R=J(3,3,0)

	for (i=1 ; i<=rows(q) ; i++) {
		Q[1,1]= 1
		Q[1,2]= tau[i]*q[i]*(eta_c_w1 + eta_c_w2)
		Q[1,3]= tau[i]*(1-q[i])*(eta_c_w1 + eta_c_w2)
		Q[2,2]= 1+tau[i]*q[i]*(eta_h1_w1 + eta_h1_w2)
		Q[2,3]= tau[i]*(1-q[i])*(eta_h1_w1 + eta_h1_w2)
		Q[3,2]= tau[i]*q[i]*(eta_h2_w1 + eta_h2_w2)
		Q[3,3]= 1+tau[i]*(1-q[i])*(eta_h2_w1 + eta_h2_w2)

		R[1,1]= -eta_c_p + eta_c_w1 + eta_c_w2
		R[1,2]= eta_c_w1 - tau[i]*q[i]*(eta_c_w1 + eta_c_w2)
		R[1,3]= eta_c_w2 - tau[i]*(1-q[i])*(eta_c_w1 + eta_c_w2)
		R[2,1]= eta_h1_p + eta_h1_w1 + eta_h1_w2
		R[2,2]= eta_h1_w1 - tau[i]*q[i]*(eta_h1_w1 + eta_h1_w2)
		R[2,3]= eta_h1_w2 - tau[i]*(1-q[i])*(eta_h1_w1 + eta_h1_w2)
		R[3,1]= eta_h2_p + eta_h2_w1 + eta_h2_w2
		R[3,2]= eta_h2_w1 - tau[i]*q[i]*(eta_h2_w1 + eta_h2_w2)
		R[3,3]= eta_h2_w2 - tau[i]*(1-q[i])*(eta_h2_w1 + eta_h2_w2)
		
		PSI= pinv(Q)*R
		
		psi_c_lam[i]=PSI[1,1]
		psi_c_w1[i]=PSI[1,2]
		psi_c_w2[i]=PSI[1,3]
		psi_h1_lam[i]=PSI[2,1]
		psi_h1_w1[i]=PSI[2,2]
		psi_h1_w2[i]=PSI[2,3]
		psi_h2_lam[i]=PSI[3,1]
		psi_h2_w1[i]=PSI[3,2]
		psi_h2_w2[i]=PSI[3,3]

	}

	st_store(.,vpsi_c_lam,psi_c_lam)
	st_store(.,vpsi_c_w1,psi_c_w1)
	st_store(.,vpsi_c_w2,psi_c_w2)
	st_store(.,vpsi_h1_lam,psi_h1_lam)
	st_store(.,vpsi_h1_w1,psi_h1_w1)
	st_store(.,vpsi_h1_w2,psi_h1_w2)
	st_store(.,vpsi_h2_lam,psi_h2_lam)
	st_store(.,vpsi_h2_w1,psi_h2_w1)
	st_store(.,vpsi_h2_w2,psi_h2_w2)
}
end

*** Calculate the elasticities for each bootstrap replication
forvalues i=1/`counter' {
	local rep=`i'-1
	
	if `rep'==0 {
		u "$directory/data_het12pi1fv10cons5_rep0.dta", clear
	}
	if `rep'>0 {
		u "$directory/data_`rep'", clear
	}

	replace tau=0
	cap drop _*

	tempvar psi_c_lam psi_c_w1 psi_c_w2 ///
			psi_h1_lam psi_h1_w1 psi_h1_w2 ///
			psi_h2_lam psi_h2_w1 psi_h2_w2 /// 
			psi_h_lam psi_h_w1 psi_h_w2 ///
			k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 ///
			pi denom `eps_string'
		
	local eta_c_p	= eta_c_p
	local eta_c_w1	= eta_c_w1
	local eta_h1_w1	= eta_h1_w1	 
	local eta_c_w2	= eta_c_w2
	local eta_h1_w2	= eta_h1_w2
	local eta_h2_w2	= eta_h2_w2
	local a0		= a0
	local eta_h1_p	= eta_h1_p
	local eta_h2_p	= eta_h2_p
	local eta_h2_w1 = eta_h2_w1

	local s_omega 	= s_omega
	
	gen `pi'	= `a0'*pi_hat  

	gen `psi_c_lam'=.
	gen `psi_c_w1'=.
	gen `psi_c_w2'=.
	gen `psi_h1_lam'=.
	gen `psi_h1_w1'=.
	gen `psi_h1_w2'=.
	gen `psi_h2_lam'=.
	gen `psi_h2_w1'=.
	gen `psi_h2_w2'=.

	noisily	mata: calc_invQR()

	gen `psi_h_lam' = s_hat*`psi_h1_lam' + (1-s_hat)*`psi_h2_lam'
	gen `psi_h_w1' 	= s_hat*`psi_h1_w1' + (1-s_hat)*`psi_h2_w1'
	gen `psi_h_w2' 	= s_hat*`psi_h1_w2' + (1-s_hat)*`psi_h2_w2'
	gen `denom'  	= `psi_c_lam' - (1-tau)*(`pi')*`psi_h_lam'
	
	gen `k1' = `psi_c_w1'
	gen `k2' = `psi_c_w2'
	gen `k3' = `psi_c_w1' + `psi_c_lam'*((1-tau)*(`pi')*(s_hat+`psi_h_w1')-`psi_c_w1')/`denom'
	gen `k4' = `psi_c_w2' + `psi_c_lam'*((1-tau)*(`pi')*((1-s_hat)+`psi_h_w2')-`psi_c_w2')/`denom'
	
	gen `k5' = 1 + `psi_h1_w1'
	gen `k6' = `psi_h1_w2'
	gen `k7' = 1 + `psi_h1_w1' + `psi_h1_lam'*((1-tau)*(`pi')*(s_hat+`psi_h_w1') - `psi_c_w1')/`denom'
	gen `k8' = `psi_h1_w2' + `psi_h1_lam'*((1-tau)*(`pi')*((1-s_hat)+`psi_h_w2') - `psi_c_w2')/`denom'

	gen `k9' = `psi_h2_w1'
	gen `k10' = 1 + `psi_h2_w2'
	gen `k11' = `psi_h2_w1' + `psi_h2_lam'*((1-tau)*(`pi')*(s_hat+`psi_h_w1') - `psi_c_w1')/`denom'
	gen `k12' = 1 + `psi_h2_w2' + `psi_h2_lam'*((1-tau)*(`pi')*((1-s_hat)+`psi_h_w2') - `psi_c_w2')/`denom'
	
	foreach var in psi_c_lam psi_c_w1 psi_c_w2 ///
			psi_h1_lam psi_h1_w1 psi_h1_w2 ///
			psi_h2_lam psi_h2_w1 psi_h2_w2 /// 
			psi_h_lam psi_h_w1 psi_h_w2 ///
			k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 {
			
			gen `var'_AT = ``var''
			
			}
	if `rep'==0 {
		save "$directory/kAT_0_data.dta", replace
	}
    
	collapse k*AT
	save "$directory/kAT_`rep'", replace
			
}

*** Generate a table of elasticities with s.e.
u "$directory/kAT_0", clear
gen rep=0
forvalues rep=1/`nreps' {
	append using "$directory/kAT_`rep'"
	replace rep=`rep' if rep==.
}
order rep
duplicates drop k1_AT - k12_AT, force	/* This gets rid of the bootstrap replications that failed, since these do not update the data set therefore the same dataset is saved again */ 
keep in 1/`bskeep'

foreach var of varlist  k1_AT - k12_AT {
	egen iqr_`var'_temp = iqr(`var') if rep!=0
	egen iqr_`var' = max(iqr_`var'_temp)
	gen se_`var'_iqr= iqr_`var'/1.349
	drop iqr_`var'*
}
drop rep
xpose, clear varname
keep _v v1
order _v

gen index=1
replace index = sum(index)

gen seiqrtag=_v=="se_k1_AT_iqr"
su index if seiqrtag==1
gen se_iqr=v1[_n+r(mean)-1]

ren v1 point_estimate
su index if seiqrtag==1
local  n=r(mean)
drop in `n'/l

drop index *tag

save "$directory/bootstrap_AT_elasticities", replace
outsheet using "$directory/bootstrap_AT_elasticities.csv", replace comma
outsheet using "$graphs/Table5_ext_AT_elasticities.csv", replace comma

/* Erase all but the true data replication which is in the Marshallian figure */ 
forvalues i=1/`counter' {
	local rep=`i'-1
	erase "$directory/kAT_`rep'.dta"
}

cd "$do"
