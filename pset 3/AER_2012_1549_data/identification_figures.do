/* 	This file generates two figures by changing one parameter at a time holding all others constant:
	(1) Figure 1, changing eta_c_w1 and then eta_c_w2
	(2) Appendix figure, changing beta and eta_c_p.
	At all cases we hold all other parameters at their levels at the baseline 
	specification (T4C1). 
*/

clear all
cd "$output"

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

*============================
* Figure 1: Change eta_c_w1 and eta_c_w2
*============================
*** Prepare eta_c_w1 figure
local j=1
forvalues i=-25(2.5)25 {

	u "$directory/data_het12pi1fv10cons5_rep0.dta", clear
	cap drop _*
	
	gen k3_temp_0 = k3
	local scale = `i'/10
	replace eta_c_w1=`scale'*eta_c_w1
	replace eta_h1_p=`scale'*eta_h1_p

	global eta_c_p	= eta_c_p
	global eta_c_w1	= eta_c_w1
	global eta_h1_w1= eta_h1_w1	 
	global eta_c_w2	= eta_c_w2
	global eta_h1_w2= eta_h1_w2
	global eta_h2_w2= eta_h2_w2
	global a0		= a0
	global eta_h1_p	= eta_h1_p
	global eta_h2_p	= eta_h2_p
	global eta_h2_w1= eta_h2_w1

	do "$do/identification_figures_sub"
	gen scale = `scale'
	save temp`j', replace
	local j=`j'+1
}

u temp1, clear
forvalues i=2/21 {
	append using temp`i'
	erase temp`i'.dta
}
erase temp1.dta

sort eta_c_w1
label var eta_c_w1 "{&eta}{subscript:c,w1}"
label var k3_temp "{&kappa}{subscript:c,v1}"
su k3_temp_0 
local yline = r(mean)
#delimit;
tw 	line k3_temp eta_c_w1,
	graphregion(color(white))
	xline(0, lp(_))
	yline(`yline', lp(_))
	name(eta_c_w1, replace);
#delimit cr


gen diff = (k3_temp_0-k3_temp)/k3_temp
cap log close
log using "$graphs/Figure_7_numbers", replace t
l diff if scale==0
log close
drop diff

*** Prepare eta_c_w2 figure
local j=1
forvalues i=-75(7.5)75 {

	u "$directory/data_het12pi1fv10cons5_rep0.dta", clear
	cap drop _*

	gen k4_temp_0 = k4
	local scale = `i'/10
	replace eta_c_w2=`scale'*eta_c_w2
	replace eta_h2_p=`scale'*eta_h2_p

	global eta_c_p	= eta_c_p
	global eta_c_w1	= eta_c_w1
	global eta_h1_w1= eta_h1_w1	 
	global eta_c_w2	= eta_c_w2
	global eta_h1_w2= eta_h1_w2
	global eta_h2_w2= eta_h2_w2
	global a0		= a0
	global eta_h1_p	= eta_h1_p
	global eta_h2_p	= eta_h2_p
	global eta_h2_w1= eta_h2_w1

	do "$do/identification_figures_sub"
	gen scale = `scale'
	save temp`j', replace
	local j=`j'+1
}

u temp1, clear
forvalues i=2/21 {
	append using temp`i'
	erase temp`i'.dta
}
erase temp1.dta

sort eta_c_w2
label var eta_c_w2 "{&eta}{subscript:c,w2}"
label var k4_temp "{&kappa}{subscript:c,v2}"
su k4_temp_0 
local yline = r(mean)
#delimit;
tw 	line k4_temp eta_c_w2,
	graphregion(color(white))
	xline(0, lp(_))
	yline(`yline', lp(_))
	name(eta_c_w2, replace);
#delimit cr

gen diff = (k4_temp_0-k4_temp)/k4_temp
cap log close
log using "$graphs/Figure_7_numbers", append t
l diff if scale==0
log close
drop diff

*** Combine the two figures
window manage close graph _all
graph combine eta_c_w1 eta_c_w2, ycommon graphregion(color(white))
graph save "$graphs/Figure_7_no_annotate", replace	/* Annotations are manually added */ 


*============================
* Appendix figure: Change beta and eta_c_p
*============================
*** Prepare eta_c_p 
local j=1
forvalues i=0(0.5)10 {

	u "$directory/data_het12pi1fv10cons5_rep0.dta", clear
	cap drop _*
	
	local scale = `i'/10
	replace eta_c_p=`scale'

	global eta_c_p	= eta_c_p
	global eta_c_w1	= eta_c_w1
	global eta_h1_w1= eta_h1_w1	 
	global eta_c_w2	= eta_c_w2
	global eta_h1_w2= eta_h1_w2
	global eta_h2_w2= eta_h2_w2
	global a0		= a0
	global eta_h1_p	= eta_h1_p
	global eta_h2_p	= eta_h2_p
	global eta_h2_w1= eta_h2_w1

	do "$do/identification_figures_sub"
	save temp`j', replace
	local j=`j'+1
}

u temp1, clear
forvalues i=2/21 {
	append using temp`i'
	erase temp`i'.dta
}
erase temp1.dta

sort eta_c_p
replace k7_temp = k7_temp -1

label var eta_c_p "{&eta}{subscript:c,p}"
label var k3_temp "{&kappa}{sub:c,v1} Response of c to perm. shock to w{sub:1}"
label var k7_temp "{&kappa}{sub:h1,v1} Response of h{sub:1} to perm. shock to w{sub:1}"

#delimit;
tw 	line k3_temp k7_temp eta_c_p,
	lc(blue red) lp(solid .-) lwidth(medthick)
	graphregion(color(white))
	name(eta_c_p, replace)
	legend(rows(2));
#delimit cr

*** Prepare beta  
local j=1
forvalues i=0(0.5)10 {

	u "$directory/data_het12pi1fv10cons5_rep0.dta", clear
	cap drop _*
	
	local scale = `i'/10
	replace a0=`scale'

	global eta_c_p	= eta_c_p
	global eta_c_w1	= eta_c_w1
	global eta_h1_w1= eta_h1_w1	 
	global eta_c_w2	= eta_c_w2
	global eta_h1_w2= eta_h1_w2
	global eta_h2_w2= eta_h2_w2
	global a0		= a0
	global eta_h1_p	= eta_h1_p
	global eta_h2_p	= eta_h2_p
	global eta_h2_w1= eta_h2_w1

	do "$do/identification_figures_sub"
	save temp`j', replace
	local j=`j'+1
}

u temp1, clear
forvalues i=2/21 {
	append using temp`i'
	erase temp`i'.dta
}
erase temp1.dta

replace a0 = 1-a0
sort a0
replace k7_temp = k7_temp -1

label var a0 "{&beta}"
label var k3_temp "{&kappa}{sub:c,v1} Response of c to perm. shock to w{sub:1}"
label var k7_temp "{&kappa}{sub:h1,v1} Response of h{sub:1} to perm. shock to w{sub:1}"

#delimit;
tw 	line k3_temp k7_temp a0,
	lc(blue red) lp(solid .-) lwidth(medthick)
	graphregion(color(white))
	name(beta, replace)
	legend(rows(2));
#delimit cr

*** Combine the two figures
window manage close graph _all
grc1leg eta_c_p beta, ycommon graphregion(color(white)) 
graph save "$graphs/Figure_Appendix", replace	/* Annotations are manually added */ 

cd "$do"
