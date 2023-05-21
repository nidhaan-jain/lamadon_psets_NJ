/* This file prepares the life-cycle fit Figure 9 */ 

cd "$output"

* Choose min and max ages
local minage 30
local maxage 57

u "$directory/data_het12pi1fv10cons5_rep0.dta", clear

keep k1-k12 s_u1 s_u2 s_v1 s_v2 r_u1u2 r_v1v2 age log_c log_y log_w log_h l2_q pi s_me_y s_me_w s_me_h s_me_yw_cons s_me_ww_cons s_me_hw_cons 

* Do not use observations which did not contribute to any moment in the Frisch estimation when calculating average kappa by age
foreach var of varlist k1-k12 {
	replace `var'=. if l2_q==. | pi==.
}
drop l2_q pi

*** Calculate the observed variances over age and their standard errors (note: previous versions had bootstrapped s.e., which deliver very similar results)
foreach var of varlist log_* {
	gen sd_`var' = .
	gen se_sd_`var' = .
	gen sdelement_`var' = .
	levelsof age, local(ages)
	foreach a in `ages' {
		reg `var' if age==`a'
		replace sdelement_`var' = (`var' - _b[_cons])^2 if age==`a'
		reg sdelement_`var' if age==`a'
		replace se_sd_`var' = _se[_cons] if age==`a'
		replace sd_`var' = _b[_cons] if age==`a'
	}
}

drop log_* sdelement_*

foreach var of varlist k1-k12 s_u1 s_u2 s_v1 s_v2 r_u1u2 r_v1v2 sd_log_* se_sd_log_* {
	egen m_`var' = mean(`var'), by(age)
	drop `var'
	ren m_`var' `var'
}

sort age
drop if age==age[_n-1]

ren k1 kcu1
ren k2 kcu2
ren k3 kcv1
ren k4 kcv2

ren k5 ky1u1
ren k6 ky1u2
ren k7 ky1v1
ren k8 ky1v2

ren k9 ky2u1
ren k10 ky2u2
ren k11 ky2v1
ren k12 ky2v2


***********CALCULATE FIT OVER CROSS-SECTIONS OF AGE **************
*** Two first years
foreach var of varlist k* {
	replace `var' = `var'[3] in 1/2
}
*** Replace negative variances with zero
foreach var of varlist s_* {
	replace `var' = 0 if `var'<0
}

sort age
merge age using "$do/init_cond"
drop if _m==2
drop _merge

drop if age>`maxage'

*****HOURLY WAGES
g 		permw=s_v1 				if age==`minage'
replace permw=s_v1+permw[_n-1] 	if age>`minage'

gen fit_wage_psid=s_u1+permw+lwpl+s_me_w-0.23

*****EARNINGS
g 		permy=ky1v1^2*s_v1+ky1v2^2*s_v2+2*ky1v1*ky1v2*r_v1v2 				if age==`minage'
replace permy=ky1v1^2*s_v1+ky1v2^2*s_v2+2*ky1v1*ky1v2*r_v1v2+permy[_n-1] 	if age>`minage'
g trany	 	 =ky1u1^2*s_u1+ky1u2^2*s_u2+2*ky1u1*ky1u2*r_u1u2	

g fit_earn_psid=permy+trany+lypl+s_me_y-0.40

*****HOURS
g kh1v1=ky1v1-1
g kh1u1=ky1u1-1

g 		permh=kh1v1^2*s_v1+ky1v2^2*s_v2+2*kh1v1*ky1v2*r_v1v2 				if age==`minage'
replace permh=kh1v1^2*s_v1+ky1v2^2*s_v2+2*kh1v1*ky1v2*r_v1v2+permh[_n-1] 	if age>`minage'
g tranh	 	 =kh1u1^2*s_u1+ky1u2^2*s_u2+2*kh1u1*ky1u2*r_u1u2	

g fit_hour_psid=permh+tranh+lhpl+s_me_h-0.18

*****CONSUMPTION
g 		permc=kcv1^2*s_v1+kcv2^2*s_v2+2*kcv1*kcv2*r_v1v2 				if age==`minage'
replace permc=kcv1^2*s_v1+kcv2^2*s_v2+2*kcv1*kcv2*r_v1v2+permc[_n-1] 	if age>`minage'
g tranc	 	 =kcu1^2*s_u1+kcu2^2*s_u2+2*kcu1*kcu2*r_u1u2	

g fit_cons =permc+tranc+lcl-0.11

keep age fit*psid fit_cons sd_log_* se_sd_log_*

save fit_data_se, replace

*===========
* Figure 9
*===========
u fit_data_se, clear
* 95% confidence band for variance in the data
foreach var of varlist  sd_*  {
	gen `var'_hi = `var' + 1.96*se_`var'
	gen `var'_lw = `var' - 1.96*se_`var'
}

sort age

*** Wages
lab var fit_wage_psid "Fit PSID init. cond."
lab var sd_log_w "Var(log(W))"
lab var age "Age"
twoway rarea sd_log_w_lw sd_log_w_hi age, sort color(gs14) legend(order(2 3 4 5)) || scatter sd_log_w fit_wage_psid age, c(. l) ///
			lp(dash dash) ms(oh i s +) graphregion(color(white)) saving(fitw,replace) xlabel(`minage'(5)`maxage')

*** Earnings
lab var sd_log_y "Var(log(Y))"
lab var fit_earn_psid "Fit PSID init. cond."
twoway rarea sd_log_y_lw sd_log_y_hi age, sort color(gs14) legend(order(2 3 4 5)) || scatter sd_log_y fit_earn_psid age, c(. l l) ms(oh i) ///
			lp(dash dash dash dash_dot) graphregion(color(white)) saving(fity,replace) xlabel(`minage'(5)`maxage')

*** hours
lab var sd_log_h "Var(log(H))"
lab var fit_hour_psid "Fit PSID init. cond."

twoway rarea sd_log_h_lw sd_log_h_hi age, sort color(gs14) legend(order(2 3 4 5)) || scatter sd_log_h fit_hour_psid age, c(. l l) ms(oh i) ///
			lp(dash dash dash dash_dot) graphregion(color(white)) saving(fith,replace) xlabel(`minage'(5)`maxage')


*** consumption
lab var sd_log_c "Var(log(C))"
lab var fit_cons "Fit CEX init. cond."
twoway rarea sd_log_c_lw sd_log_c_hi age, sort color(gs14) legend(order(2 3 4 5)) || scatter sd_log_c fit_cons age, c(. l l) ms(oh i) ///
			lp(dash dash dash dash_dot) graphregion(color(white)) saving(fitc,replace) xlabel(`minage'(5)`maxage')

graph combine fitc.gph fity.gph fith.gph fitw.gph, iscale(.5) saving(fit_overall,replace) graphregion(color(white))
graph export "$graphs/Figure9_external_fit.png", as(png) width(1600) replace
graph export "$graphs/Figure9_external_fit.eps", as(eps) replace
graph save "$graphs/Figure9_external_fit.gph", replace  

cd "$do"
