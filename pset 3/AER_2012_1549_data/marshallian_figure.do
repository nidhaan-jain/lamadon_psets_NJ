/* Marshallian responses and decomposition graphs by Age */

graph set print fontface times

/* 
Note:
====
This file generates the figures for Marshallian elasticities by age as well as for the decomposition of total 
consumption smoothing to the different smoothing mechanisms. 

The last part of this file generates the transmission coefficients for the sample of FS recipients. This is
used in the text discussion insurance decomposition. 

*/ 

cd "$directory"

*********************
* Prepare variables *
*********************
u kAT_0_data, clear
sort person year
gen marsh_h1_w1 = k7_AT-1
gen marsh_h2_w2 = k12_AT-1
keep person year marsh_h1_w1 marsh_h2_w2
save temp, replace

u data_het12pi1fv10cons5_rep0.dta, clear
sort person year
merge 1:1 person year using temp
drop _m

keep age agew pi s_hat k* alpha eta* marsh_h1_w1 marsh_h2_w2 tau
gen beta=1-alpha
drop alpha
replace pi=1-pi
ren s_hat s

*** Response of total earnings of the household.
gen dy_dv1=(1-tau)*(s*k7+(1-s)*k11)
gen dy_dv2=(1-tau)*(s*k8+(1-s)*k12)

**********
* Graphs *
**********
drop if age<32

*** Collapse by age categories
collapse k3 k4 k7 k8 k11 k12 dy_dv1 dy_dv2 s tau marsh_h1_w1 marsh_h2_w2, by(age)
gen s2=1-s
replace s=s*(1-tau)
replace s2=s2*(1-tau)

*** Normalize variables to be the response to a 10 percent negative shock
foreach var of varlist k3 k4 dy_dv1 dy_dv2 s s2 {
	replace `var'=`var'*(-10)
}

foreach var of varlist k7 k12 {
	replace `var'=`var'-1
}


*** Figure 6
#delimit;
tw 	(lowess k3 age, lwidth(medthick) ms(T) msize(large) lc(blue) lp(solid))
	(lowess s age, lwidth(medthick) ms(D) msize(large) lc(red) lp(-)) 
	(lowess dy_dv1 age, lwidth(medthick) ms(O) msize(large) lc(green) lp(-.))
	, graphregion(color(white)) xlabel(35(5)55)
	legend(rows(3)) legend(order(2 3 1))
	legend(label(1 with family labor supply adjustment and other insurance)
		   label(2 fixed labor supply and no insurance)
		   label(3 with family labor supply adjustment))
	xtitle("Age of household head")
	name(smoothing_graph_all, replace)
	title("Response of Consumption to a 10% Permanent" "Decrease in the Male’s Wage Rate");
#delimit cr
graph export "$graphs\Figure6_response_by_age_v1.png", as(png) width(1600) replace
graph export "$graphs\Figure6_response_by_age_v1.eps", as(eps) replace
graph save "$graphs\Figure6_response_by_age_v1.gph", replace

*** Figure 5
#delimit:
tw 	(lowess marsh_h1_w1 age, lwidth(medthick) lp(solid) lc(blue))
	(lowess marsh_h2_w2 age, lwidth(medthick) lp(-) lc(red))
	, graphregion(color(white)) xlabel(35(5)55) 
	legend(rows(3)) legend(order(2 3 1))
	xtitle("Age of household head")
	title("After-tax Marshallian Elasticities")
	legend(label(1 Head's Marshallian Elasticity)
		   label(2 Wife's Marshallian Elasticity));
#delimit cr
graph export "$graphs\Figure5_marshallian_age.png", as(png) width(1600) replace
graph export "$graphs\Figure5_marshallian_age.eps", as(eps) replace
graph save "$graphs\Figure5_marshallian_age.gph", replace


*************************************
* Export kappas for FSB recipients  *
*************************************
u data_het12pi1fv10cons5_rep0.dta, clear
keep if FSB_pay>0 & FSB_pay!=.

keep age agew pi s_hat k* alpha eta* tau
keep if k1!=.
gen beta=1-alpha
drop alpha
replace pi=1-pi
ren s_hat s

collapse k1-k12 s tau
xpose, varname clear
order _varname 
outsheet using "$graphs/kappas_FS_eligible.csv", replace comma

cd "$do"
