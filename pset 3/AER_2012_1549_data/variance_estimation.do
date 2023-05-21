/* GMM Estimation of the variance parameters by age (does not need consumption moments or earnings moments) */ 

cd "$output"
set more off

*********************
** User Parameters **
*********************
scalar sh_me_yw=sh_me_y
scalar sh_me_hw=sh_me_h
scalar sh_me_ww=sh_me_w

**************************************************
** Stage zero - gen measurement error estimates **
**************************************************

/* ME for male */

foreach name in y h w {
	gen s_me_`name' = .
}
gen cov_me_y_h = .

levelsof age_cat, local(age_cat)
foreach a in `age_cat' {
	foreach name in y h w {
		su log_`name' if age_cat==`a'
		scalar sd=r(sd)
		replace s_me_`name' = (sd^2)*sh_me_`name' if age_cat==`a'
	} 
	replace cov_me_y_h = 0.5*(s_me_y + s_me_h - s_me_w) if age_cat==`a'
}

/* ME for female (need to condition on working in two periods, since these are the only moments used in estimation) */ 
foreach name in y yw h hw w ww {
	gen s_me_`name'_cons = .
}
gen cov_me_yw_hw_cons = .

levelsof agew_cat, local(agew_cat)
foreach a in `agew_cat' {
	foreach name in yw hw ww {
		su log_`name' if agew_cat==`a' & duyw!=. & duww!=.
		scalar sd=r(sd)
		replace s_me_`name'_cons = (sd^2)*sh_me_`name' if agew_cat==`a'
	}

	replace cov_me_yw_hw_cons = 0.5*(s_me_yw_cons + s_me_hw_cons - s_me_ww_cons) if agew_cat==`a'
}
 
***********************************************
*** Stage 1: estimating variance parameters ***
***********************************************
* Prepare age categories
gen agecat1=age<=37
gen agecat2=age==38
gen agecat3=age==39
gen agecat4=age>=40 & age<=42
gen agecat5=age==43
gen agecat6=age==44
gen agecat7=age>=45 & age<=47
gen agecat8=age==48
gen agecat9=age==49
gen agecat10=age>=50 & age<=52
gen agecat11=age==53
gen agecat12=age==54
gen agecat13=age>=55 

gen tagecat1=age<=39
gen tagecat2=age>=40 & age<=44
gen tagecat3=age>=45 & age<=49
gen tagecat4=age>=50 & age<=54
gen tagecat5=age>=55 

gen agewcat1=agew<=37
gen agewcat2=agew==38
gen agewcat3=agew==39
gen agewcat4=agew>=40 & agew<=42
gen agewcat5=agew==43
gen agewcat6=agew==44
gen agewcat7=agew>=45 & agew<=47
gen agewcat8=agew==48
gen agewcat9=agew==49
gen agewcat10=agew>=50 & agew<=52
gen agewcat11=agew==53
gen agewcat12=agew==54
gen agewcat13=agew>=55 

gen tagewcat1=agew<=39
gen tagewcat2=agew>=40 & agew<=44
gen tagewcat3=agew>=45 & agew<=49
gen tagewcat4=agew>=50 & agew<=54
gen tagewcat5=agew>=55 

if $fix_var == 10 {

	**********************************************************************************
	** Shutting off participation correction. Allowing for variance to change by age *                                                             					*
	**********************************************************************************

	foreach name in s_u1 s_v1 se_s_u1 se_s_v1 observations ///
					s_u2 s_v2 r_u1u2 r_v1v2 r_u1 r_u2 r_v1 r_v2 /// 
					se_s_u2 se_s_v2 se_r_u1u2 se_r_v1v2 se_r_u1 se_r_u2 se_r_v1 se_r_v2 {
	gen `name'=.
}

	#delimit;
	gmm (duw2   -({p1}+{p1}+{t1}+{t1})	*agecat1  
				-({p1}+{p2}+{t2}+{t1})	*agecat2  
				-({p2}+{p2}+{t2}+{t1})	*agecat3 
				-({p2}+{p2}+{t2}+{t2})	*agecat4 
				-({p3}+{p2}+{t3}+{t2})	*agecat5 	   
				-({p3}+{p3}+{t3}+{t2})	*agecat6    
				-({p3}+{p3}+{t3}+{t3})	*agecat7 
				-({p4}+{p3}+{t4}+{t3})	*agecat8 
				-({p4}+{p4}+{t4}+{t3})	*agecat9 
				-({p4}+{p4}+{t4}+{t4})	*agecat10 
				-({p5}+{p4}+{t5}+{t4})	*agecat11 
				-({p5}+{p5}+{t5}+{t4})	*agecat12 	  
				-({p5}+{p5}+{t5}+{t5})	*agecat13 - 2*s_me_w) 		  
		(duw_lag +{t1}*tagecat1 +{t2}*tagecat2 +{t3}*tagecat3+{t4}*tagecat4+{t5}*tagecat5 + s_me_w)  
		(duww2  -({wp1}+{wp1}+{wt1}+{wt1})	*agewcat1  
				-({wp1}+{wp2}+{wt2}+{wt1})	*agewcat2  
				-({wp2}+{wp2}+{wt2}+{wt1})	*agewcat3 
				-({wp2}+{wp2}+{wt2}+{wt2})	*agewcat4 
				-({wp3}+{wp2}+{wt3}+{wt2})	*agewcat5 	   
				-({wp3}+{wp3}+{wt3}+{wt2})	*agewcat6    
				-({wp3}+{wp3}+{wt3}+{wt3})	*agewcat7 
				-({wp4}+{wp3}+{wt4}+{wt3})	*agewcat8 
				-({wp4}+{wp4}+{wt4}+{wt3})	*agewcat9 
				-({wp4}+{wp4}+{wt4}+{wt4})	*agewcat10 
				-({wp5}+{wp4}+{wt5}+{wt4})	*agewcat11 
				-({wp5}+{wp5}+{wt5}+{wt4})	*agewcat12 	  
				-({wp5}+{wp5}+{wt5}+{wt5})	*agewcat13 - 2*s_me_ww) 		  
		(duww_lag +{wt1}*tagewcat1 +{wt2}*tagewcat2 +{wt3}*tagewcat3+{wt4}*tagewcat4+{wt5}*tagewcat5 + s_me_ww) 
		(duw_duww -({rp1}+{rp1}+{rt1}+{rt1})	*agecat1  
				-({rp1}+{rp2}+{rt2}+{rt1})	*agecat2  
				-({rp2}+{rp2}+{rt2}+{rt1})	*agecat3 
				-({rp2}+{rp2}+{rt2}+{rt2})	*agecat4 
				-({rp3}+{rp2}+{rt3}+{rt2})	*agecat5 	   
				-({rp3}+{rp3}+{rt3}+{rt2})	*agecat6    
				-({rp3}+{rp3}+{rt3}+{rt3})	*agecat7 
				-({rp4}+{rp3}+{rt4}+{rt3})	*agecat8 
				-({rp4}+{rp4}+{rt4}+{rt3})	*agecat9 
				-({rp4}+{rp4}+{rt4}+{rt4})	*agecat10 
				-({rp5}+{rp4}+{rt5}+{rt4})	*agecat11 
				-({rp5}+{rp5}+{rt5}+{rt4})	*agecat12 	  
				-({rp5}+{rp5}+{rt5}+{rt5})	*agecat13) 		  
		(duw_duww_lag +{rt1}*tagecat1 +{rt2}*tagecat2 +{rt3}*tagecat3+{rt4}*tagecat4+{rt5}*tagecat5)
		(duww_duw_lag +{rt1}*tagecat1 +{rt2}*tagecat2 +{rt3}*tagecat3+{rt4}*tagecat4+{rt5}*tagecat5),  
		$gmm_config
		from(p1 $s_v1_0 p2 $s_v1_0 p3 $s_v1_0 p4 $s_v1_0 p5 $s_v1_0 
		t1 $s_u1_0 t2 $s_u1_0 t3 $s_u1_0 t4 $s_u1_0 t5 $s_u1_0
		wp1 $s_v2_0 wp2 $s_v2_0 wp3 $s_v2_0 wp4 $s_v2_0 wp5 $s_v2_0 
		wt1 $s_u2_0 wt2 $s_u2_0 wt3 $s_u2_0 wt4 $s_u2_0 wt5 $s_u2_0
		rp1 $r_v1v2_0 rp2 $r_v1v2_0 rp3 $r_v1v2_0 rp4 $r_v1v2_0 rp5 $r_v1v2_0 
		rt1 $r_u1u2_0 rt2 $r_u1u2_0 rt3 $r_u1u2_0 rt4 $r_u1u2_0 rt5 $r_u1u2_0)
		instruments(1 5: agecat1-agecat13,nocons) instruments(2 6 7: tagecat1-tagecat5,nocons) 
		instruments(3: agewcat1-agewcat13,nocons) instruments(4: tagewcat1-tagewcat5,nocons) 
		conv_maxiter($maxiter)
		$nocommonesample quickd vce(cluster person);
		#delimit cr

		matrix obs=e(N_byequation)
		global observations=obs[1,1]
		predict ss_resid*
		
	/* Assign the variance to age categories */ 		
	foreach a in `age_cat' {
		replace s_u1 = _b[/t`a'] if age_cat == `a'
		replace s_v1 = _b[/p`a'] if age_cat == `a'
		replace se_s_u1 = _se[/t`a'] if age_cat == `a'
		replace se_s_v1 = _se[/p`a'] if age_cat == `a'
		replace s_u2 = _b[/wt`a'] if agew_cat == `a'
		replace s_v2 = _b[/wp`a'] if agew_cat == `a'
		replace se_s_u2 = _se[/wt`a'] if agew_cat == `a'
		replace se_s_v2 = _se[/wp`a'] if agew_cat == `a'
		replace r_u1u2 = _b[/rt`a'] if age_cat == `a'
		replace r_v1v2 = _b[/rp`a'] if age_cat == `a'
		replace se_r_u1u2 = _se[/rt`a'] if age_cat == `a'
		replace se_r_v1v2 = _se[/rp`a'] if age_cat == `a'
	}

	/* Generate the mean variance over two age groups for second step moments without lags */ 		
	gen 	m_s_u1 = _b[/t1] 				if age<=37
	replace m_s_u1 = 0.5*(_b[/t2]+_b[/t1])  if age==38
	replace m_s_u1 = 0.5*(_b[/t2]+_b[/t1])  if age==39
	replace m_s_u1 = _b[/t2]				if age>=40 & age<=42
	replace m_s_u1 = 0.5*(_b[/t3]+_b[/t2]) 	if age==43
	replace m_s_u1 = 0.5*(_b[/t3]+_b[/t2]) 	if age==44
	replace m_s_u1 = _b[/t3]				if age>=45 & age<=47
	replace m_s_u1 = 0.5*(_b[/t4]+_b[/t3]) 	if age==48
	replace m_s_u1 = 0.5*(_b[/t4]+_b[/t3]) 	if age==49
	replace m_s_u1 = _b[/t4]				if age>=50 & age<=52
	replace m_s_u1 = 0.5*(_b[/t5]+_b[/t4]) 	if age==53
	replace m_s_u1 = 0.5*(_b[/t5]+_b[/t4]) 	if age==54
	replace m_s_u1 = _b[/t5]				if age>=55 

	gen 	m_s_v1 = _b[/p1] 				if age<=37
	replace m_s_v1 = 0.5*(_b[/p2]+_b[/p1])  if age==38
	replace m_s_v1 = _b[/p2]				if age==39
	replace m_s_v1 = _b[/p2]				if age>=40 & age<=42
	replace m_s_v1 = 0.5*(_b[/p3]+_b[/p2]) 	if age==43
	replace m_s_v1 = _b[/p3]			 	if age==44
	replace m_s_v1 = _b[/p3]				if age>=45 & age<=47
	replace m_s_v1 = 0.5*(_b[/p4]+_b[/p3]) 	if age==48
	replace m_s_v1 = _b[/p4] 				if age==49
	replace m_s_v1 = _b[/p4]				if age>=50 & age<=52
	replace m_s_v1 = 0.5*(_b[/p5]+_b[/p4]) 	if age==53
	replace m_s_v1 = _b[/p5] 				if age==54
	replace m_s_v1 = _b[/p5]				if age>=55 

	gen 	m_r_u1u2 = _b[/rt1] 				if age<=37
	replace m_r_u1u2 = 0.5*(_b[/rt2]+_b[/rt1])  if age==38
	replace m_r_u1u2 = 0.5*(_b[/rt2]+_b[/rt1]) 	if age==39
	replace m_r_u1u2 = _b[/rt2]					if age>=40 & age<=42
	replace m_r_u1u2 = 0.5*(_b[/rt3]+_b[/rt2]) 	if age==43
	replace m_r_u1u2 = 0.5*(_b[/rt3]+_b[/rt2]) 	if age==44
	replace m_r_u1u2 = _b[/rt3]					if age>=45 & age<=47
	replace m_r_u1u2 = 0.5*(_b[/rt4]+_b[/rt3]) 	if age==48
	replace m_r_u1u2 = 0.5*(_b[/rt4]+_b[/rt3]) 	if age==49
	replace m_r_u1u2 = _b[/rt4]					if age>=50 & age<=52
	replace m_r_u1u2 = 0.5*(_b[/rt5]+_b[/rt4]) 	if age==53
	replace m_r_u1u2 = 0.5*(_b[/rt5]+_b[/rt4]) 	if age==54
	replace m_r_u1u2 = _b[/rt5]					if age>=55 

	gen 	m_r_v1v2 = _b[/rp1] 				if age<=37
	replace m_r_v1v2 = 0.5*(_b[/rp2]+_b[/rp1]) 	if age==38
	replace m_r_v1v2 = _b[/rp2]					if age==39
	replace m_r_v1v2 = _b[/rp2]					if age>=40 & age<=42
	replace m_r_v1v2 = 0.5*(_b[/rp3]+_b[/rp2]) 	if age==43
	replace m_r_v1v2 = _b[/rp3]			 		if age==44
	replace m_r_v1v2 = _b[/rp3]					if age>=45 & age<=47
	replace m_r_v1v2 = 0.5*(_b[/rp4]+_b[/rp3]) 	if age==48
	replace m_r_v1v2 = _b[/rp4] 				if age==49
	replace m_r_v1v2 = _b[/rp4]					if age>=50 & age<=52
	replace m_r_v1v2 = 0.5*(_b[/rp5]+_b[/rp4]) 	if age==53
	replace m_r_v1v2 = _b[/rp5] 				if age==54
	replace m_r_v1v2 = _b[/rp5]					if age>=55 
	
	gen 	m_s_u2 = _b[/wt1] 					if agew<=37
	replace m_s_u2 = 0.5*(_b[/wt2]+_b[/wt1])  	if agew==38
	replace m_s_u2 = 0.5*(_b[/wt2]+_b[/wt1])  	if agew==39
	replace m_s_u2 = _b[/wt2]					if agew>=40 & agew<=42
	replace m_s_u2 = 0.5*(_b[/wt3]+_b[/wt2]) 	if agew==43
	replace m_s_u2 = 0.5*(_b[/wt3]+_b[/wt2]) 	if agew==44
	replace m_s_u2 = _b[/wt3]					if agew>=45 & agew<=47
	replace m_s_u2 = 0.5*(_b[/wt4]+_b[/wt3]) 	if agew==48
	replace m_s_u2 = 0.5*(_b[/wt4]+_b[/wt3]) 	if agew==49
	replace m_s_u2 = _b[/wt4]					if agew>=50 & agew<=52
	replace m_s_u2 = 0.5*(_b[/wt5]+_b[/wt4]) 	if agew==53
	replace m_s_u2 = 0.5*(_b[/wt5]+_b[/wt4]) 	if agew==54
	replace m_s_u2 = _b[/wt5]					if agew>=55 

	gen 	m_s_v2 = _b[/wp1] 					if agew<=37
	replace m_s_v2 = 0.5*(_b[/wp2]+_b[/wp1])  	if agew==38
	replace m_s_v2 = _b[/wp2]					if agew==39
	replace m_s_v2 = _b[/wp2]					if agew>=40 & agew<=42
	replace m_s_v2 = 0.5*(_b[/wp3]+_b[/wp2]) 	if agew==43
	replace m_s_v2 = _b[/wp3]			 		if agew==44
	replace m_s_v2 = _b[/wp3]					if agew>=45 & agew<=47
	replace m_s_v2 = 0.5*(_b[/wp4]+_b[/wp3]) 	if agew==48
	replace m_s_v2 = _b[/wp4] 					if agew==49
	replace m_s_v2 = _b[/wp4]					if agew>=50 & agew<=52
	replace m_s_v2 = 0.5*(_b[/wp5]+_b[/wp4]) 	if agew==53
	replace m_s_v2 = _b[/wp5] 					if agew==54
	replace m_s_v2 = _b[/wp5]					if agew>=55 
	
	/* Generate the lagged variance estimates for the second step estimation (this is only for transitory) */ 		
	gen 	l2_s_u1 = _b[/t1] if age<=39
	replace l2_s_u1 = _b[/t2] if age>=40 & age<=44
	replace l2_s_u1 = _b[/t3] if age>=45 & age<=49
	replace l2_s_u1 = _b[/t4] if age>=50 & age<=54
	replace l2_s_u1 = _b[/t5] if age>=55 

	gen 	l2_r_u1u2 = _b[/rt1] if age<=39
	replace l2_r_u1u2 = _b[/rt2] if age>=40 & age<=44
	replace l2_r_u1u2 = _b[/rt3] if age>=45 & age<=49
	replace l2_r_u1u2 = _b[/rt4] if age>=50 & age<=54
	replace l2_r_u1u2 = _b[/rt5] if age>=55 

	gen 	l2_s_u2 = _b[/wt1] if agew<=39
	replace l2_s_u2 = _b[/wt2] if agew>=40 & agew<=44
	replace l2_s_u2 = _b[/wt3] if agew>=45 & agew<=49
	replace l2_s_u2 = _b[/wt4] if agew>=50 & agew<=54
	replace l2_s_u2 = _b[/wt5] if agew>=55 

	* Ignoring Selection in this specification
	foreach name in r_u1 r_u2 r_v1 r_v2 {
		replace `name' = 0
		gen m_`name' = 0
		}
	foreach name in r_u1 r_u2 {
		gen l2_`name' = 0
		}
	
}

if $fix_var == 9 {

	*******************************************************************
	** Shutting off participation correction. Variance fixed over age *                                                             					*
	*******************************************************************
	
	* Replace measurement error with age constant m.e.
	foreach name in y h w {
		su log_`name'
		scalar sd=r(sd)
		replace s_me_`name' = (sd^2)*sh_me_`name'
	}
	replace cov_me_y_h = 0.5*(s_me_y + s_me_h - s_me_w)

	* ME for female 
	foreach name in yw hw ww {
		su log_`name' if duyw!=. & duww!=.
		scalar sd=r(sd)
		replace s_me_`name'_cons = (sd^2)*sh_me_`name'
	}

	replace cov_me_yw_hw_cons = 0.5*(s_me_yw_cons + s_me_hw_cons - s_me_ww_cons)
	
	
	* Estimate with age constant variances (note that to keep code similar in the Frisch step, still assign parameters to age categories
	foreach name in s_u1 s_v1 se_s_u1 se_s_v1 observations ///
					s_u2 s_v2 r_u1u2 r_v1v2 r_u1 r_u2 r_v1 r_v2 /// 
					se_s_u2 se_s_v2 se_r_u1u2 se_r_v1v2 se_r_u1 se_r_u2 se_r_v1 se_r_v2 {
		gen `name'=.
	}

	#delimit;
	gmm  	(duw2 - 2*({s_u1} + {s_v1} + s_me_w))
			(duww2		- ({s_u2}*2 + {s_v2}*2 + 2*s_me_ww_cons))
			(duw_lag - (- {s_u1} - s_me_w))
			(duww_lag	- (-{s_u2} - s_me_ww_cons))
			(duw_duww	- ((2*{r_u1u2}) + (2*{r_v1v2})))
			(duw_duww_lag - (-{r_u1u2}))
			(duww_duw_lag - (-{r_u1u2}))
			$gmm_config 
			from(s_u1 $s_u1_0 s_v1 $s_v1_0 s_u2 $s_u2_0 s_v2 $s_v2_0 r_u1u2 $r_u1u2_0 r_v1v2 $r_v1v2_0)
			$nocommonesample quickd  vce(cluster person);
	#delimit cr

	matrix obs=e(N_byequation)
	global observations=obs[1,1]
	predict ss_resid*

	/* Assign the variance to variables */ 		
	foreach var in s_u1 s_v1 s_u2 s_v2 r_u1u2 r_v1v2 {
		replace `var' = _b[/`var'] 
		replace se_`var' = _se[/`var']
	}
	
	/* Generate the mean variance over two age groups for second step moments without lags (for compatibility with age varying case) */ 		
	foreach var in s_u1 s_v1 s_u2 s_v2 r_u1u2 r_v1v2 {
		gen 	m_`var' = _b[/`var']
	}
	
	/* Generate the lagged variance (for compatibility with age varying case) */ 		
	foreach var in s_u1 s_u2 r_u1u2 {
		gen 	l2_`var' = _b[/`var']
	}
	
	* Ignoring Selection in this specification
	foreach name in r_u1 r_u2 r_v1 r_v2 {
		replace `name' = 0
		gen m_`name' = 0
		}
	foreach name in r_u1 r_u2 {
		gen l2_`name' = 0
		}
	
}

if $fix_var == 4 {

	*******************************************************
	** With participation correction and variances by age *                                                             					*
	*******************************************************

	foreach name in s_u1 s_v1 se_s_u1 se_s_v1 observations ///
					s_u2 s_v2 r_u1u2 r_v1v2 r_u1 r_u2 r_v1 r_v2 /// 
					se_s_u2 se_s_v2 se_r_u1u2 se_r_v1v2 se_r_u1 se_r_u2 se_r_v1 se_r_v2 {
	gen `name'=.
	}
	
	#delimit;
	gmm (duw2   -({p1}+{p1}+{t1}+{t1})	*agecat1  
				-({p1}+{p2}+{t2}+{t1})	*agecat2  
				-({p2}+{p2}+{t2}+{t1})	*agecat3 
				-({p2}+{p2}+{t2}+{t2})	*agecat4 
				-({p3}+{p2}+{t3}+{t2})	*agecat5 	   
				-({p3}+{p3}+{t3}+{t2})	*agecat6    
				-({p3}+{p3}+{t3}+{t3})	*agecat7 
				-({p4}+{p3}+{t4}+{t3})	*agecat8 
				-({p4}+{p4}+{t4}+{t3})	*agecat9 
				-({p4}+{p4}+{t4}+{t4})	*agecat10 
				-({p5}+{p4}+{t5}+{t4})	*agecat11 
				-({p5}+{p5}+{t5}+{t4})	*agecat12 	  
				-({p5}+{p5}+{t5}+{t5})	*agecat13 - 2*s_me_w) 		  
		(duw_lag + {t1}*tagecat1 +{t2}*tagecat2 +{t3}*tagecat3+{t4}*tagecat4+{t5}*tagecat5 + s_me_w)  
		(duww2  -({wp1}+{wp1}+{wt1}+{wt1} - ({wrv1})^2*z_eta*inv_mills - ({wru1})^2*z_eta*inv_mills - ({wru1})^2*l2_z_eta*l2_inv_mills)	*agewcat1  
				-({wp2}+{wp1}+{wt2}+{wt1} - ({wrv2})^2*z_eta*inv_mills - ({wru2})^2*z_eta*inv_mills - ({wru1})^2*l2_z_eta*l2_inv_mills) *agewcat2  
				-({wp2}+{wp2}+{wt2}+{wt1} - ({wrv2})^2*z_eta*inv_mills - ({wru2})^2*z_eta*inv_mills - ({wru1})^2*l2_z_eta*l2_inv_mills)	*agewcat3 
				-({wp2}+{wp2}+{wt2}+{wt2} - ({wrv2})^2*z_eta*inv_mills - ({wru2})^2*z_eta*inv_mills - ({wru2})^2*l2_z_eta*l2_inv_mills)	*agewcat4 
				-({wp3}+{wp2}+{wt3}+{wt2} - ({wrv3})^2*z_eta*inv_mills - ({wru3})^2*z_eta*inv_mills - ({wru2})^2*l2_z_eta*l2_inv_mills)	*agewcat5 	   
				-({wp3}+{wp3}+{wt3}+{wt2} - ({wrv3})^2*z_eta*inv_mills - ({wru3})^2*z_eta*inv_mills - ({wru2})^2*l2_z_eta*l2_inv_mills)	*agewcat6    
				-({wp3}+{wp3}+{wt3}+{wt3} - ({wrv3})^2*z_eta*inv_mills - ({wru3})^2*z_eta*inv_mills - ({wru3})^2*l2_z_eta*l2_inv_mills)	*agewcat7 
				-({wp4}+{wp3}+{wt4}+{wt3} - ({wrv4})^2*z_eta*inv_mills - ({wru4})^2*z_eta*inv_mills - ({wru3})^2*l2_z_eta*l2_inv_mills)	*agewcat8 
				-({wp4}+{wp4}+{wt4}+{wt3} - ({wrv4})^2*z_eta*inv_mills - ({wru4})^2*z_eta*inv_mills - ({wru3})^2*l2_z_eta*l2_inv_mills)	*agewcat9 
				-({wp4}+{wp4}+{wt4}+{wt4} - ({wrv4})^2*z_eta*inv_mills - ({wru4})^2*z_eta*inv_mills - ({wru4})^2*l2_z_eta*l2_inv_mills)	*agewcat10 
				-({wp5}+{wp4}+{wt5}+{wt4} - ({wrv5})^2*z_eta*inv_mills - ({wru5})^2*z_eta*inv_mills - ({wru4})^2*l2_z_eta*l2_inv_mills)	*agewcat11 
				-({wp5}+{wp5}+{wt5}+{wt4} - ({wrv5})^2*z_eta*inv_mills - ({wru5})^2*z_eta*inv_mills - ({wru4})^2*l2_z_eta*l2_inv_mills)	*agewcat12 	  
				-({wp5}+{wp5}+{wt5}+{wt5} - ({wrv5})^2*z_eta*inv_mills - ({wru5})^2*z_eta*inv_mills - ({wru5})^2*l2_z_eta*l2_inv_mills)	*agewcat13 - 2*s_me_ww) 		  
		(duww_lag + ({wt1}-({wru1})^2*l2_z_eta*l2_inv_mills)*tagewcat1 +({wt2}-({wru2})^2*l2_z_eta*l2_inv_mills)*tagewcat2 
				+({wt3}-({wru3})^2*l2_z_eta*l2_inv_mills)*tagewcat3+({wt4}-({wru4})^2*l2_z_eta*l2_inv_mills)*tagewcat4
				+({wt5}-({wru5})^2*l2_z_eta*l2_inv_mills)*tagewcat5 + s_me_ww) 
		(duw_duww - ({rp1}+{rp1}+{rt1}+{rt1} - ({wrv1}*{wrv1}*z_eta*inv_mills) - ({ru1}*{wru1}*z_eta*inv_mills) - ({ru1}*{wru1}*l2_z_eta*l2_inv_mills))	*agecat1  
				-({rp2}+{rp1}+{rt2}+{rt1} - ({rv2}*{wrv2}*z_eta*inv_mills) - ({ru2}*{wru2}*z_eta*inv_mills) - ({ru1}*{wru1}*l2_z_eta*l2_inv_mills))	*agecat2  
				-({rp2}+{rp2}+{rt2}+{rt1} - ({rv2}*{wrv2}*z_eta*inv_mills) - ({ru2}*{wru2}*z_eta*inv_mills) - ({ru1}*{wru1}*l2_z_eta*l2_inv_mills))	*agecat3 
				-({rp2}+{rp2}+{rt2}+{rt2} - ({rv2}*{wrv2}*z_eta*inv_mills) - ({ru2}*{wru2}*z_eta*inv_mills) - ({ru2}*{wru2}*l2_z_eta*l2_inv_mills))	*agecat4 
				-({rp3}+{rp2}+{rt3}+{rt2} - ({rv3}*{wrv3}*z_eta*inv_mills) - ({ru3}*{wru3}*z_eta*inv_mills) - ({ru2}*{wru2}*l2_z_eta*l2_inv_mills))	*agecat5 	   
				-({rp3}+{rp3}+{rt3}+{rt2} - ({rv3}*{wrv3}*z_eta*inv_mills) - ({ru3}*{wru3}*z_eta*inv_mills) - ({ru2}*{wru2}*l2_z_eta*l2_inv_mills))	*agecat6    
				-({rp3}+{rp3}+{rt3}+{rt3} - ({rv3}*{wrv3}*z_eta*inv_mills) - ({ru3}*{wru3}*z_eta*inv_mills) - ({ru3}*{wru3}*l2_z_eta*l2_inv_mills))	*agecat7 
				-({rp4}+{rp3}+{rt4}+{rt3} - ({rv4}*{wrv4}*z_eta*inv_mills) - ({ru4}*{wru4}*z_eta*inv_mills) - ({ru3}*{wru3}*l2_z_eta*l2_inv_mills))	*agecat8 
				-({rp4}+{rp4}+{rt4}+{rt3} - ({rv4}*{wrv4}*z_eta*inv_mills) - ({ru4}*{wru4}*z_eta*inv_mills) - ({ru3}*{wru3}*l2_z_eta*l2_inv_mills))	*agecat9 
				-({rp4}+{rp4}+{rt4}+{rt4} - ({rv4}*{wrv4}*z_eta*inv_mills) - ({ru4}*{wru4}*z_eta*inv_mills) - ({ru4}*{wru4}*l2_z_eta*l2_inv_mills))	*agecat10 
				-({rp5}+{rp4}+{rt5}+{rt4} - ({rv5}*{wrv5}*z_eta*inv_mills) - ({ru5}*{wru5}*z_eta*inv_mills) - ({ru4}*{wru4}*l2_z_eta*l2_inv_mills))	*agecat11 
				-({rp5}+{rp5}+{rt5}+{rt4} - ({rv5}*{wrv5}*z_eta*inv_mills) - ({ru5}*{wru5}*z_eta*inv_mills) - ({ru4}*{wru4}*l2_z_eta*l2_inv_mills))	*agecat12 	  
				-({rp5}+{rp5}+{rt5}+{rt5} - ({rv5}*{wrv5}*z_eta*inv_mills) - ({ru5}*{wru5}*z_eta*inv_mills) - ({ru5}*{wru5}*l2_z_eta*l2_inv_mills))	*agecat13) 		  
		(duw_duww_lag + ({rt1} - {ru1}*{wru1}*l2_z_eta*l2_inv_mills)*tagecat1 + ({rt2} - {ru2}*{wru2}*l2_z_eta*l2_inv_mills)*tagecat2 
			+ ({rt3} - {ru3}*{wru3}*l2_z_eta*l2_inv_mills)*tagecat3 + ({rt4} - {ru4}*{wru4}*l2_z_eta*l2_inv_mills)*tagecat4
			+ ({rt5} - {ru5}*{wru5}*l2_z_eta*l2_inv_mills)*tagecat5)
		(duww_duw_lag + ({rt1} - {ru1}*{wru1}*l2_z_eta*l2_inv_mills)*tagecat1 + ({rt2} - {ru2}*{wru2}*l2_z_eta*l2_inv_mills)*tagecat2 
			+ ({rt3} - {ru3}*{wru3}*l2_z_eta*l2_inv_mills)*tagecat3 + ({rt4} - {ru4}*{wru4}*l2_z_eta*l2_inv_mills)*tagecat4
			+ ({rt5} - {ru5}*{wru5}*l2_z_eta*l2_inv_mills)*tagecat5)
		(duw 	- ({ru1}*inv_mills - {ru1}*l2_inv_mills + {rv1}*inv_mills)*(agecat1)
				- ({ru2}*inv_mills - {ru1}*l2_inv_mills + {rv2}*inv_mills)*(agecat2 + agecat3)
				- ({ru2}*inv_mills - {ru2}*l2_inv_mills + {rv2}*inv_mills)*(agecat4)
				- ({ru3}*inv_mills - {ru2}*l2_inv_mills + {rv3}*inv_mills)*(agecat5 + agecat6)
				- ({ru3}*inv_mills - {ru3}*l2_inv_mills + {rv3}*inv_mills)*(agecat7)
				- ({ru4}*inv_mills - {ru3}*l2_inv_mills + {rv4}*inv_mills)*(agecat8 + agecat9)
				- ({ru4}*inv_mills - {ru4}*l2_inv_mills + {rv4}*inv_mills)*(agecat10)
				- ({ru5}*inv_mills - {ru4}*l2_inv_mills + {rv5}*inv_mills)*(agecat11 + agecat12)
				- ({ru5}*inv_mills - {ru5}*l2_inv_mills + {rv5}*inv_mills)*(agecat13))  
		(duww 	- ({wru1}*inv_mills - {wru1}*l2_inv_mills + {wrv1}*inv_mills)*(agewcat1)
				- ({wru2}*inv_mills - {wru1}*l2_inv_mills + {wrv2}*inv_mills)*(agewcat2 + agewcat3)
				- ({wru2}*inv_mills - {wru2}*l2_inv_mills + {wrv2}*inv_mills)*(agewcat4)
				- ({wru3}*inv_mills - {wru2}*l2_inv_mills + {wrv3}*inv_mills)*(agewcat5 + agewcat6)
				- ({wru3}*inv_mills - {wru3}*l2_inv_mills + {wrv3}*inv_mills)*(agewcat7)
				- ({wru4}*inv_mills - {wru3}*l2_inv_mills + {wrv4}*inv_mills)*(agewcat8 + agewcat9)
				- ({wru4}*inv_mills - {wru4}*l2_inv_mills + {wrv4}*inv_mills)*(agewcat10)
				- ({wru5}*inv_mills - {wru4}*l2_inv_mills + {wrv5}*inv_mills)*(agewcat11 + agewcat12)
				- ({wru5}*inv_mills - {wru5}*l2_inv_mills + {wrv5}*inv_mills)*(agewcat13)),
 		$gmm_config
		from(p1 $s_v1_0 p2 $s_v1_0 p3 $s_v1_0 p4 $s_v1_0 p5 $s_v1_0 
		t1 $s_u1_0 t2 $s_u1_0 t3 $s_u1_0 t4 $s_u1_0 t5 $s_u1_0
		wp1 $s_v2_0 wp2 $s_v2_0 wp3 $s_v2_0 wp4 $s_v2_0 wp5 $s_v2_0 
		wt1 $s_u2_0 wt2 $s_u2_0 wt3 $s_u2_0 wt4 $s_u2_0 wt5 $s_u2_0
		rp1 $r_v1v2_0 rp2 $r_v1v2_0 rp3 $r_v1v2_0 rp4 $r_v1v2_0 rp5 $r_v1v2_0 
		rt1 $r_u1u2_0 rt2 $r_u1u2_0 rt3 $r_u1u2_0 rt4 $r_u1u2_0 rt5 $r_u1u2_0
		rv1 $r_v1_0 rv2 $r_v1_0 rv3 $r_v1_0 rv4 $r_v1_0 rv5 $r_v1_0
		wrv1 $r_v2_0 wrv2 $r_v2_0 wrv3 $r_v2_0 wrv4 $r_v2_0 wrv5 $r_v2_0 
		ru1 $r_u1_0 ru2 $r_u1_0 ru3 $r_u1_0 ru4 $r_u1_0 ru5 $r_u1_0 
		wru1 $r_u2_0 wru2 $r_u2_0 wru3 $r_u2_0 wru4 $r_u2_0 wru5 $r_u2_0)
		instruments(1 5: agecat1-agecat13,nocons) instruments(2 6 7: tagecat1-tagecat5,nocons) 
		instruments(3: agewcat1-agewcat13,nocons) instruments(4: tagewcat1-tagewcat5,nocons)
		instruments(8: agecat1-agecat13 inv_mills l2_inv_mills,nocons) instruments(9: agewcat1-agewcat13 inv_mills l2_inv_mills,nocons)
		conv_maxiter($maxiter)
		$nocommonesample quickd vce(cluster person);
		#delimit cr

		matrix obs=e(N_byequation)
		global observations=obs[1,1]
		predict ss_resid*

	/* Assign the variance to age categories */ 		
	foreach a in `age_cat' {
		replace s_u1 = _b[/t`a'] if age_cat == `a'
		replace s_v1 = _b[/p`a'] if age_cat == `a'
		replace se_s_u1 = _se[/t`a'] if age_cat == `a'
		replace se_s_v1 = _se[/p`a'] if age_cat == `a'
		replace s_u2 = _b[/wt`a'] if agew_cat == `a'
		replace s_v2 = _b[/wp`a'] if agew_cat == `a'
		replace se_s_u2 = _se[/wt`a'] if agew_cat == `a'
		replace se_s_v2 = _se[/wp`a'] if agew_cat == `a'
		replace r_u1u2 = _b[/rt`a'] if age_cat == `a'
		replace r_v1v2 = _b[/rp`a'] if age_cat == `a'
		replace se_r_u1u2 = _se[/rt`a'] if age_cat == `a'
		replace se_r_v1v2 = _se[/rp`a'] if age_cat == `a'
			
		replace r_u1 = _b[/ru`a'] if age_cat == `a'
		replace r_v1 = _b[/rv`a'] if age_cat == `a'
		replace se_r_u1 = _se[/ru`a'] if age_cat == `a'
		replace se_r_v1 = _se[/rv`a'] if age_cat == `a'
		replace r_u2 = _b[/wru`a'] if agew_cat == `a'
		replace r_v2 = _b[/wrv`a'] if agew_cat == `a'
		replace se_r_u2 = _se[/wru`a'] if agew_cat == `a'
		replace se_r_v2 = _se[/wrv`a'] if agew_cat == `a'
	}

	/* Generate the mean variance over two age groups for second step moments without lags */ 		
	gen 	m_s_u1 = _b[/t1] 				if age<=37
	replace m_s_u1 = 0.5*(_b[/t2]+_b[/t1])  if age==38
	replace m_s_u1 = 0.5*(_b[/t2]+_b[/t1])  if age==39
	replace m_s_u1 = _b[/t2]				if age>=40 & age<=42
	replace m_s_u1 = 0.5*(_b[/t3]+_b[/t2]) 	if age==43
	replace m_s_u1 = 0.5*(_b[/t3]+_b[/t2]) 	if age==44
	replace m_s_u1 = _b[/t3]				if age>=45 & age<=47
	replace m_s_u1 = 0.5*(_b[/t4]+_b[/t3]) 	if age==48
	replace m_s_u1 = 0.5*(_b[/t4]+_b[/t3]) 	if age==49
	replace m_s_u1 = _b[/t4]				if age>=50 & age<=52
	replace m_s_u1 = 0.5*(_b[/t5]+_b[/t4]) 	if age==53
	replace m_s_u1 = 0.5*(_b[/t5]+_b[/t4]) 	if age==54
	replace m_s_u1 = _b[/t5]				if age>=55 

	gen 	m_s_v1 = _b[/p1] 				if age<=37
	replace m_s_v1 = 0.5*(_b[/p2]+_b[/p1])  if age==38
	replace m_s_v1 = _b[/p2]				if age==39
	replace m_s_v1 = _b[/p2]				if age>=40 & age<=42
	replace m_s_v1 = 0.5*(_b[/p3]+_b[/p2]) 	if age==43
	replace m_s_v1 = _b[/p3]			 	if age==44
	replace m_s_v1 = _b[/p3]				if age>=45 & age<=47
	replace m_s_v1 = 0.5*(_b[/p4]+_b[/p3]) 	if age==48
	replace m_s_v1 = _b[/p4] 				if age==49
	replace m_s_v1 = _b[/p4]				if age>=50 & age<=52
	replace m_s_v1 = 0.5*(_b[/p5]+_b[/p4]) 	if age==53
	replace m_s_v1 = _b[/p5] 				if age==54
	replace m_s_v1 = _b[/p5]				if age>=55 

	gen 	m_r_u1u2 = _b[/rt1] 				if age<=37
	replace m_r_u1u2 = 0.5*(_b[/rt2]+_b[/rt1])  if age==38
	replace m_r_u1u2 = 0.5*(_b[/rt2]+_b[/rt1]) 	if age==39
	replace m_r_u1u2 = _b[/rt2]					if age>=40 & age<=42
	replace m_r_u1u2 = 0.5*(_b[/rt3]+_b[/rt2]) 	if age==43
	replace m_r_u1u2 = 0.5*(_b[/rt3]+_b[/rt2]) 	if age==44
	replace m_r_u1u2 = _b[/rt3]					if age>=45 & age<=47
	replace m_r_u1u2 = 0.5*(_b[/rt4]+_b[/rt3]) 	if age==48
	replace m_r_u1u2 = 0.5*(_b[/rt4]+_b[/rt3]) 	if age==49
	replace m_r_u1u2 = _b[/rt4]					if age>=50 & age<=52
	replace m_r_u1u2 = 0.5*(_b[/rt5]+_b[/rt4]) 	if age==53
	replace m_r_u1u2 = 0.5*(_b[/rt5]+_b[/rt4]) 	if age==54
	replace m_r_u1u2 = _b[/rt5]					if age>=55 

	gen 	m_r_v1v2 = _b[/rp1] 				if age<=37
	replace m_r_v1v2 = 0.5*(_b[/rp2]+_b[/rp1]) 	if age==38
	replace m_r_v1v2 = _b[/rp2]					if age==39
	replace m_r_v1v2 = _b[/rp2]					if age>=40 & age<=42
	replace m_r_v1v2 = 0.5*(_b[/rp3]+_b[/rp2]) 	if age==43
	replace m_r_v1v2 = _b[/rp3]			 		if age==44
	replace m_r_v1v2 = _b[/rp3]					if age>=45 & age<=47
	replace m_r_v1v2 = 0.5*(_b[/rp4]+_b[/rp3]) 	if age==48
	replace m_r_v1v2 = _b[/rp4] 				if age==49
	replace m_r_v1v2 = _b[/rp4]					if age>=50 & age<=52
	replace m_r_v1v2 = 0.5*(_b[/rp5]+_b[/rp4]) 	if age==53
	replace m_r_v1v2 = _b[/rp5] 				if age==54
	replace m_r_v1v2 = _b[/rp5]					if age>=55 
	
	gen 	m_s_u2 = _b[/wt1] 					if agew<=37
	replace m_s_u2 = 0.5*(_b[/wt2]+_b[/wt1])  	if agew==38
	replace m_s_u2 = 0.5*(_b[/wt2]+_b[/wt1])  	if agew==39
	replace m_s_u2 = _b[/wt2]					if agew>=40 & agew<=42
	replace m_s_u2 = 0.5*(_b[/wt3]+_b[/wt2]) 	if agew==43
	replace m_s_u2 = 0.5*(_b[/wt3]+_b[/wt2]) 	if agew==44
	replace m_s_u2 = _b[/wt3]					if agew>=45 & agew<=47
	replace m_s_u2 = 0.5*(_b[/wt4]+_b[/wt3]) 	if agew==48
	replace m_s_u2 = 0.5*(_b[/wt4]+_b[/wt3]) 	if agew==49
	replace m_s_u2 = _b[/wt4]					if agew>=50 & agew<=52
	replace m_s_u2 = 0.5*(_b[/wt5]+_b[/wt4]) 	if agew==53
	replace m_s_u2 = 0.5*(_b[/wt5]+_b[/wt4]) 	if agew==54
	replace m_s_u2 = _b[/wt5]					if agew>=55 

	gen 	m_s_v2 = _b[/wp1] 					if agew<=37
	replace m_s_v2 = 0.5*(_b[/wp2]+_b[/wp1])  	if agew==38
	replace m_s_v2 = _b[/wp2]					if agew==39
	replace m_s_v2 = _b[/wp2]					if agew>=40 & agew<=42
	replace m_s_v2 = 0.5*(_b[/wp3]+_b[/wp2]) 	if agew==43
	replace m_s_v2 = _b[/wp3]			 		if agew==44
	replace m_s_v2 = _b[/wp3]					if agew>=45 & agew<=47
	replace m_s_v2 = 0.5*(_b[/wp4]+_b[/wp3]) 	if agew==48
	replace m_s_v2 = _b[/wp4] 					if agew==49
	replace m_s_v2 = _b[/wp4]					if agew>=50 & agew<=52
	replace m_s_v2 = 0.5*(_b[/wp5]+_b[/wp4]) 	if agew==53
	replace m_s_v2 = _b[/wp5] 					if agew==54
	replace m_s_v2 = _b[/wp5]					if agew>=55 
	
	gen 	m_r_u1 = _b[/ru1] 					if age<=37
	replace m_r_u1 = 0.5*(_b[/ru2]+_b[/ru1])  	if age==38
	replace m_r_u1 = 0.5*(_b[/ru2]+_b[/ru1])  	if age==39
	replace m_r_u1 = _b[/ru2]					if age>=40 & age<=42
	replace m_r_u1 = 0.5*(_b[/ru3]+_b[/ru2]) 	if age==43
	replace m_r_u1 = 0.5*(_b[/ru3]+_b[/ru2]) 	if age==44
	replace m_r_u1 = _b[/ru3]					if age>=45 & age<=47
	replace m_r_u1 = 0.5*(_b[/ru4]+_b[/ru3]) 	if age==48
	replace m_r_u1 = 0.5*(_b[/ru4]+_b[/ru3]) 	if age==49
	replace m_r_u1 = _b[/ru4]					if age>=50 & age<=52
	replace m_r_u1 = 0.5*(_b[/ru5]+_b[/ru4]) 	if age==53
	replace m_r_u1 = 0.5*(_b[/ru5]+_b[/ru4]) 	if age==54
	replace m_r_u1 = _b[/ru5]					if age>=55 
	
	gen 	m_r_u2 = _b[/wru1] 					if agew<=37
	replace m_r_u2 = 0.5*(_b[/wru2]+_b[/wru1])  if agew==38
	replace m_r_u2 = 0.5*(_b[/wru2]+_b[/wru1])  if agew==39
	replace m_r_u2 = _b[/wru2]					if agew>=40 & agew<=42
	replace m_r_u2 = 0.5*(_b[/wru3]+_b[/wru2]) 	if agew==43
	replace m_r_u2 = 0.5*(_b[/wru3]+_b[/wru2]) 	if agew==44
	replace m_r_u2 = _b[/wru3]					if agew>=45 & agew<=47
	replace m_r_u2 = 0.5*(_b[/wru4]+_b[/wru3]) 	if agew==48
	replace m_r_u2 = 0.5*(_b[/wru4]+_b[/wru3]) 	if agew==49
	replace m_r_u2 = _b[/wru4]					if agew>=50 & agew<=52
	replace m_r_u2 = 0.5*(_b[/wru5]+_b[/wru4]) 	if agew==53
	replace m_r_u2 = 0.5*(_b[/wru5]+_b[/wru4]) 	if agew==54
	replace m_r_u2 = _b[/wru5]					if agew>=55
	
	gen 	m_r_v1 = _b[/rv1] 					if age<=37
	replace m_r_v1 = 0.5*(_b[/rv2]+_b[/rv1])  	if age==38
	replace m_r_v1 = 0.5*(_b[/rv2]+_b[/rv1])  	if age==39
	replace m_r_v1 = _b[/rv2]					if age>=40 & age<=42
	replace m_r_v1 = 0.5*(_b[/rv3]+_b[/rv2]) 	if age==43
	replace m_r_v1 = 0.5*(_b[/rv3]+_b[/rv2]) 	if age==44
	replace m_r_v1 = _b[/rv3]					if age>=45 & age<=47
	replace m_r_v1 = 0.5*(_b[/rv4]+_b[/rv3]) 	if age==48
	replace m_r_v1 = 0.5*(_b[/rv4]+_b[/rv3]) 	if age==49
	replace m_r_v1 = _b[/rv4]					if age>=50 & age<=52
	replace m_r_v1 = 0.5*(_b[/rv5]+_b[/rv4]) 	if age==53
	replace m_r_v1 = 0.5*(_b[/rv5]+_b[/rv4]) 	if age==54
	replace m_r_v1 = _b[/rv5]					if age>=55 
	
	gen 	m_r_v2 = _b[/wrv1] 					if agew<=37
	replace m_r_v2 = 0.5*(_b[/wrv2]+_b[/wrv1])  if agew==38
	replace m_r_v2 = 0.5*(_b[/wrv2]+_b[/wrv1])  if agew==39
	replace m_r_v2 = _b[/wrv2]					if agew>=40 & agew<=42
	replace m_r_v2 = 0.5*(_b[/wrv3]+_b[/wrv2]) 	if agew==43
	replace m_r_v2 = 0.5*(_b[/wrv3]+_b[/wrv2]) 	if agew==44
	replace m_r_v2 = _b[/wrv3]					if agew>=45 & agew<=47
	replace m_r_v2 = 0.5*(_b[/wrv4]+_b[/wrv3]) 	if agew==48
	replace m_r_v2 = 0.5*(_b[/wrv4]+_b[/wrv3]) 	if agew==49
	replace m_r_v2 = _b[/wrv4]					if agew>=50 & agew<=52
	replace m_r_v2 = 0.5*(_b[/wrv5]+_b[/wrv4]) 	if agew==53
	replace m_r_v2 = 0.5*(_b[/wrv5]+_b[/wrv4]) 	if agew==54
	replace m_r_v2 = _b[/wrv5]					if agew>=55
	
	
	/* Generate the lagged variance */ 		
	gen 	l2_s_u1 = _b[/t1] if age<=39
	replace l2_s_u1 = _b[/t2] if age>=40 & age<=44
	replace l2_s_u1 = _b[/t3] if age>=45 & age<=49
	replace l2_s_u1 = _b[/t4] if age>=50 & age<=54
	replace l2_s_u1 = _b[/t5] if age>=55 

	gen 	l2_r_u1u2 = _b[/rt1] if age<=39
	replace l2_r_u1u2 = _b[/rt2] if age>=40 & age<=44
	replace l2_r_u1u2 = _b[/rt3] if age>=45 & age<=49
	replace l2_r_u1u2 = _b[/rt4] if age>=50 & age<=54
	replace l2_r_u1u2 = _b[/rt5] if age>=55 

	gen 	l2_s_u2 = _b[/wt1] if agew<=39
	replace l2_s_u2 = _b[/wt2] if agew>=40 & agew<=44
	replace l2_s_u2 = _b[/wt3] if agew>=45 & agew<=49
	replace l2_s_u2 = _b[/wt4] if agew>=50 & agew<=54
	replace l2_s_u2 = _b[/wt5] if agew>=55 
	
	gen 	l2_r_u1 = _b[/ru1] if age<=39
	replace l2_r_u1 = _b[/ru2] if age>=40 & age<=44
	replace l2_r_u1 = _b[/ru3] if age>=45 & age<=49
	replace l2_r_u1 = _b[/ru4] if age>=50 & age<=54
	replace l2_r_u1 = _b[/ru5] if age>=55 
	
	gen 	l2_r_u2 = _b[/wru1] if age<=39
	replace l2_r_u2 = _b[/wru2] if age>=40 & age<=44
	replace l2_r_u2 = _b[/wru3] if age>=45 & age<=49
	replace l2_r_u2 = _b[/wru4] if age>=50 & age<=54
	replace l2_r_u2 = _b[/wru5] if age>=55 
		
}

save "$directory/ss_resid", replace
sort person year


