/* GMM Estimation of the Frisch elasticities given the estimates of the variance */ 


if ${medass}==1 {
	replace D3 = med_assets
	replace pi_tax = 1-(D3/(Q1+D3)) if pi_tax!=.
}

*=================================================
* Prepare the ratios for the Frisch Demand matrix
*=================================================
*** Linear tax case
gen c=exp(log_c)
gen y=exp(log_y)
gen yw=exp(log_yw)
gen toty=exp(log_toty)

foreach var in c y yw {
	su `var', de
	global mean_`var' = r(mean)	
}

global c_over_y=${mean_c}/${mean_y}
global c_over_yw=${mean_c}/${mean_yw}
global y_over_yw=${mean_y}/${mean_yw}

*** Non-Linear tax case
gen atrate = xsi*(1-tau)*toty^(-tau) 
foreach var in atrate  {
	su `var', de
	global mean_`var' = r(mean)	
}
global c_over_y_tax =${c_over_y}*(${mean_atrate})^(-1)
global c_over_yw_tax=${c_over_yw}*(${mean_atrate})^(-1)

global c_over_y =${c_over_y_tax}	/* note that when taxes are shut, xsi=1 and tau==0 so back to c_over_y without taxes */
global c_over_yw=${c_over_yw_tax}

*==============================
* Estimate Frisch Elasticities
*==============================
/* 	note: for linear taxes (het_elasticities==10) the taxes are set to zero in the 
	file 'tax_rate_estimation_snap', and then het_elasticities assigned 12, so in this
	file the estimation is identical */ 

*** Specification: non-linear taxes with beta ==0
if $het_elasticities == 12 & $use_pi == 1 {
	reg duc_lag
	global s_omega 		= -_b[_cons]
	global se_s_omega	= _se[_cons]

	gen pi_hat=pi_tax

	cap program drop gmm_age_fv4_pi5_ns_nltax_b0
	#delimit;
	gmm gmm_age_fv4_pi5_ns_nltax_b0, nequations(31)
	parameters(eta_c_p eta_c_w1 eta_h1_w1 eta_c_w2 eta_h1_w2 eta_h2_w2)
	$gmm_config 
	from(eta_c_p ${eta_c_p_0} eta_c_w1 ${eta_c_w1_0} eta_h1_w1 ${eta_h1_w1_0} 
		eta_c_w2 ${eta_c_w2_0} eta_h1_w2 ${eta_h1_w2_0} eta_h2_w2 ${eta_h2_w2_0})
	conv_maxiter($maxiter)
	$nocommonesample quickd  vce(cluster person);
	#delimit cr
	cap log close
		
	predict ts_resid*
	save "$directory/ts_resid", replace

	matrix b=e(b)
	matrix sb=vecdiag(e(V))
	mata
		sb = st_matrix("sb")
		sb=sb:^0.5
		st_matrix("sb",sb)
	end

	global eta_c_p	= b[1,1]
	global eta_c_w1	= b[1,2]
	global eta_h1_w1= b[1,3]
	global eta_c_w2	= b[1,4]
	global eta_h1_w2= b[1,5]
	global eta_h2_w2= b[1,6]
	global a0		= 1
	
	global eta_h1_p	= -${eta_c_w1}*${c_over_y}
	global eta_h2_p	= -${eta_c_w2}*${c_over_yw}
	global eta_h2_w1= ${eta_h1_w2}*${y_over_yw}
	
	global se_eta_c_p	= sb[1,1]
	global se_eta_c_w1	= sb[1,2]
	global se_eta_h1_w1 = sb[1,3]
	global se_eta_c_w2	= sb[1,4]
	global se_eta_h1_w2 = sb[1,5]
	global se_eta_h2_w2 = sb[1,6]
	global se_a0		= sb[1,7]

	global se_eta_h1_p	= ${se_eta_c_w1}*${c_over_y}
	global se_eta_h2_p	= ${se_eta_c_w2}*${c_over_yw}
	global se_eta_h2_w1 = ${se_eta_h1_w2}*${y_over_yw}
	
	foreach name in 	se_eta_c_p eta_c_p se_eta_h1_w1 eta_h1_w1 se_eta_h2_w2 eta_h2_w2 /// 
						se_eta_h1_p eta_h1_p se_eta_c_w1 eta_c_w1 /// 
						se_eta_h2_p eta_h2_p se_eta_c_w2 eta_c_w2 ///
						se_eta_h1_w2 eta_h1_w2 se_eta_h2_w1 eta_h2_w1 ///
						se_s_omega s_omega /// 
						a0 se_a0 {
			gen `name' = $`name'
	}
	ren pi_hat pi
}

*** Specification: non-linear taxes with multiplicative beta
if  $het_elasticities == 12 & $use_pi == 7 {
	reg duc_lag
	global s_omega 		= -_b[_cons]
	global se_s_omega	= _se[_cons]

	gen pi_hat=pi_tax

	cap program drop gmm_age_fv4_pi5_ns_nltax_add
	#delimit;
	gmm gmm_age_fv4_pi5_ns_nltax_add, nequations(31)
	parameters(eta_c_p eta_c_w1 eta_h1_w1 eta_c_w2 eta_h1_w2 eta_h2_w2	a0)
	$gmm_config 
	from(eta_c_p ${eta_c_p_0} eta_c_w1 ${eta_c_w1_0} eta_h1_w1 ${eta_h1_w1_0} 
		eta_c_w2 ${eta_c_w2_0} eta_h1_w2 ${eta_h1_w2_0} eta_h2_w2 ${eta_h2_w2_0} a0 ${a0_0})
	conv_maxiter($maxiter)
	$nocommonesample quickd  vce(cluster person);
	#delimit cr
	cap log close
	
	predict ts_resid*
	save "$directory/ts_resid", replace

	matrix b=e(b)
	matrix sb=vecdiag(e(V))
	mata
		sb = st_matrix("sb")
		sb=sb:^0.5
		st_matrix("sb",sb)
	end

	global eta_c_p	= b[1,1]
	global eta_c_w1	= b[1,2]
	global eta_h1_w1= b[1,3]
	global eta_c_w2	= b[1,4]
	global eta_h1_w2= b[1,5]
	global eta_h2_w2= b[1,6]
	global a0		= b[1,7]
	
	global eta_h1_p	= -${eta_c_w1}*${c_over_y}
	global eta_h2_p	= -${eta_c_w2}*${c_over_yw}
	global eta_h2_w1= ${eta_h1_w2}*${y_over_yw}
	
	global se_eta_c_p	= sb[1,1]
	global se_eta_c_w1	= sb[1,2]
	global se_eta_h1_w1 = sb[1,3]
	global se_eta_c_w2	= sb[1,4]
	global se_eta_h1_w2 = sb[1,5]
	global se_eta_h2_w2 = sb[1,6]
	global se_a0		= sb[1,7]

	global se_eta_h1_p	= ${se_eta_c_w1}*${c_over_y}
	global se_eta_h2_p	= ${se_eta_c_w2}*${c_over_yw}
	global se_eta_h2_w1 = ${se_eta_h1_w2}*${y_over_yw}
	
	foreach name in 	se_eta_c_p eta_c_p se_eta_h1_w1 eta_h1_w1 se_eta_h2_w2 eta_h2_w2 /// 
						se_eta_h1_p eta_h1_p se_eta_c_w1 eta_c_w1 /// 
						se_eta_h2_p eta_h2_p se_eta_c_w2 eta_c_w2 ///
						se_eta_h1_w2 eta_h1_w2 se_eta_h2_w1 eta_h2_w1 ///
						se_s_omega s_omega /// 
						a0 se_a0 {
			gen `name' = $`name'
	}
	ren pi_hat pi

}

*** non-linear taxes with separability and beta==0
if  $het_elasticities == 1 & $use_pi == 1 {
	reg duc_lag
	global s_omega 		= -_b[_cons]
	global se_s_omega	= _se[_cons]

	gen pi_hat=pi_tax

	cap program drop gmm_age_fv4_pi5_sep_nltax_b0
	#delimit;
	gmm gmm_age_fv4_pi5_sep_nltax_b0, nequations(31)
	parameters(eta_c_p eta_h1_w1 eta_h2_w2)
	$gmm_config 
	from(eta_c_p ${eta_c_p_0} eta_h1_w1 ${eta_h1_w1_0} 
		 eta_h2_w2 ${eta_h2_w2_0})
	conv_maxiter($maxiter)
	$nocommonesample quickd  vce(cluster person);
	#delimit cr
	cap log close
	
	predict ts_resid*
	save "$directory/ts_resid", replace

	matrix b=e(b)
	matrix sb=vecdiag(e(V))
	mata
		sb = st_matrix("sb")
		sb=sb:^0.5
		st_matrix("sb",sb)
	end

	global eta_c_p	= b[1,1]
	global eta_c_w1	= 0
	global eta_h1_w1= b[1,2]
	global eta_c_w2	= 0
	global eta_h1_w2= 0
	global eta_h2_w2= b[1,3]
	global a0		= 1
	
	global eta_h1_p	= -${eta_c_w1}*${c_over_y}
	global eta_h2_p	= -${eta_c_w2}*${c_over_yw}
	global eta_h2_w1= ${eta_h1_w2}*${y_over_yw}
	
	global se_eta_c_p	= sb[1,1]
	global se_eta_c_w1	= 0
	global se_eta_h1_w1 = sb[1,2]
	global se_eta_c_w2	= 0
	global se_eta_h1_w2 = 0
	global se_eta_h2_w2 = sb[1,3]
	global se_a0		= sb[1,4]

	global se_eta_h1_p	= ${se_eta_c_w1}*${c_over_y}
	global se_eta_h2_p	= ${se_eta_c_w2}*${c_over_yw}
	global se_eta_h2_w1 = ${se_eta_h1_w2}*${y_over_yw}
	
	foreach name in 	se_eta_c_p eta_c_p se_eta_h1_w1 eta_h1_w1 se_eta_h2_w2 eta_h2_w2 /// 
						se_eta_h1_p eta_h1_p se_eta_c_w1 eta_c_w1 /// 
						se_eta_h2_p eta_h2_p se_eta_c_w2 eta_c_w2 ///
						se_eta_h1_w2 eta_h1_w2 se_eta_h2_w1 eta_h2_w1 ///
						se_s_omega s_omega /// 
						a0 se_a0 {
			gen `name' = $`name'
	}
	ren pi_hat pi

}


*========================
* Reporting the results in a single table (also used for the bootstrap)
*========================
gen pi_hat=pi
gen alpha=$a0
ren se_a0 se_alpha
gen se_a0=.

tempvar psi_c_lam psi_c_w1 psi_c_w2 ///
		psi_h1_lam psi_h1_w1 psi_h1_w2 ///
		psi_h2_lam psi_h2_w1 psi_h2_w2 /// 
		psi_h_lam psi_h_w1 psi_h_w2 ///
		k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 ///
		pi denom `eps_string'
	
local eta_c_p	= $eta_c_p
local eta_c_w1	= $eta_c_w1
local eta_h1_w1	= $eta_h1_w1	 
local eta_c_w2	= $eta_c_w2
local eta_h1_w2	= $eta_h1_w2
local eta_h2_w2	= $eta_h2_w2
local a0		= $a0

local eta_h1_p	= -`eta_c_w1'*${c_over_y}
local eta_h2_p	= -`eta_c_w2'*${c_over_yw}
local eta_h2_w1 = `eta_h1_w2'*${y_over_yw}

local s_omega 	= $s_omega

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

mata
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

end

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
		
		gen `var' = ``var''
		
		}

forvalues i=1/12  {
	egen mean_k`i' = mean(k`i')
}

foreach var of varlist psi* {
	egen mean_`var' = mean(`var')
}

cd "$directory"
save data_$poststring, replace
u data_$poststring, clear

su pi
replace a0=r(mean)

su s_hat if pi!=.
gen s_report = r(mean)
gen se_s_report = 0

levelsof age_cat, local(age_cat)
foreach var in  s_u1 s_v1 r_u1u2 r_v1v2 {
	foreach a in `age_cat' {
		su `var' if age_cat==`a'
		gen `var'_`a'=r(mean)
		su se_`var' if age_cat==`a'
		gen se_`var'_`a'=r(mean)
	}
drop `var' se_`var'
}

levelsof agew_cat, local(agew_cat)
foreach var in  s_u2 s_v2 {
	foreach a in `agew_cat' {
		su `var' if agew_cat==`a'
		gen `var'_`a'=r(mean)
		su se_`var' if agew_cat==`a'
		gen se_`var'_`a'=r(mean)
	}
drop `var' se_`var'
}

foreach var in s_me_y s_me_yw_cons {
	foreach a in `age_cat' {
		su `var' if age_cat==`a'
		gen `var'_`a'=r(mean)
	}
drop `var'
}

drop age_cat*
replace observations = $observations

keep in 1
keep 	mean_k* *eta* *s_u1* *s_u2* *s_v1* *s_v2* *s_omega alpha se_alpha /// 
		*r_u1* *r_u2* *r_v1* *r_v2* *r_u1u2* *r_v1v2* ///
		*a0 *s_report observations
drop  z_eta l2_z_eta m_s* m_r* l2_s* l2_r*

ren a0 pi
ren se_a0 se_pi

local varstring  ""
foreach var of varlist s_u1_* s_v1_* s_u2_* s_v2_* r_u1u2* r_v1v2* {
	local varstring  "`varstring' `var' se_`var'"
	di "`varstring'"
}
di "`varstring'"
order 	mean_k1 mean_k2 mean_k3 mean_k4 mean_k5 mean_k6 mean_k7 mean_k8 mean_k9 mean_k10 mean_k11 mean_k12 ///
		eta_c_p se_eta_c_p  pi se_pi alpha se_alpha  /// 
		eta_h1_w1 se_eta_h1_w1 eta_h2_w2 se_eta_h2_w2 /// 
		eta_c_w1 se_eta_c_w1 eta_h1_p se_eta_h1_p ///
		eta_c_w2 se_eta_c_w2 eta_h2_p se_eta_h2_p ///
		eta_h1_w2 se_eta_h1_w2 eta_h2_w1 se_eta_h2_w1 s_report se_s_report ///
		`varstring' ///
		r_u1 se_r_u1 r_u2 se_r_u2 r_v1 se_r_v1 r_v2 se_r_v2 /// 
		s_omega se_s_omega observations ///
		
ren alpha beta
replace beta=1-beta
replace pi=1-pi

xpose, clear varname
ren v1 value
gen se = value[_n+1] if substr(_varname[_n+1],1,3)=="se_"
gen t_stat = value/se 
label var se "NOTE: GMM standard error not corrected for first stage estimates"
drop if substr(_varname,1,3)=="se_"
order _var 

save estimates_$poststring, replace
outsheet using estimates_${poststring}.csv, replace comma

cd "$do"

erase "$output/$directory/ss_resid.dta"
erase "$output/$directory/ts_resid.dta"
