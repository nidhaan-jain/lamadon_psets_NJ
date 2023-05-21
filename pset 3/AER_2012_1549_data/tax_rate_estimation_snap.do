/* Tax rates from the US tax code are added to the data */

cap drop y
gen y=ly+wly

*************************
* Code the tax schedule *
*************************

/* Note: the EITC numbers are also adjusted for married filing jointly (phase out zone is shifted for these) */ 

*** 2008 tax schedule
local yr 108
gen tax=0.1*y 		if year==`yr'
replace tax=tax+(0.15-0.1)*(y-16000) if y>16000 & year==`yr'
replace tax=tax+(0.25-0.15)*(y-65100) if y>65100 & year==`yr'
replace tax=tax+(0.28-0.25)*(y-131400) if y>131400 & year==`yr'
replace tax=tax+(0.33-0.28)*(y-200300) if y>200300 & year==`yr'
replace tax=tax+(0.35-0.33)*(y-357700) if y>357700 & year==`yr'

* EITC 
local cr	0.0765	/* credit rate */
local mi	5720	/* Minimum income for maximum credit */
local bph	10160	/* Begin Phaseout */
local pr	0.0765	/* Phaseout rate */
local eph	15880	/* End Phaseout */

gen eitc=0 									if year==`yr'
replace eitc=`cr'*y 						if y<`mi' & kids==0 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids==0 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids==0 & year==`yr'

local cr	0.34	/* credit rate */
local mi	8580	/* Minimum income for maximum credit */
local bph	18740	/* Begin Phaseout */
local pr	0.1598	/* Phaseout rate */
local eph	36995	/* End Phaseout */

replace eitc=`cr'*y 						if y<`mi' & kids==1 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids==1 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids==1 & year==`yr'

local cr	0.4		/* credit rate */
local mi	12060	/* Minimum income for maximum credit */
local bph	18740	/* Begin Phaseout */
local pr	0.2106	/* Phaseout rate */
local eph	41646	/* End Phaseout */

replace eitc=`cr'*y 						if y<`mi' & kids>=2 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids>=2 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids>=2 & year==`yr'

*** 2006 tax schedule
local yr 106
replace tax=0.1*y 		if year==`yr'
replace tax=tax+(0.15-0.1)*(y-15100) if y>15100 & year==`yr'
replace tax=tax+(0.25-0.15)*(y-61300) if y>61300 & year==`yr'
replace tax=tax+(0.28-0.25)*(y-123700) if y>123700 & year==`yr'
replace tax=tax+(0.33-0.28)*(y-188450) if y>188450 & year==`yr'
replace tax=tax+(0.35-0.33)*(y-336550) if y>336550 & year==`yr'

* EITC 
local cr	0.0765	/* credit rate */
local mi	5380	/* Minimum income for maximum credit */
local bph	8740	/* Begin Phaseout */
local pr	0.0765	/* Phaseout rate */
local eph	14120	/* End Phaseout */

replace eitc=0 								if year==`yr'
replace eitc=`cr'*y 						if y<`mi' & kids==0 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids==0 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids==0 & year==`yr'

local cr	0.34	/* credit rate */
local mi	8080	/* Minimum income for maximum credit */
local bph	16810	/* Begin Phaseout */
local pr	0.1598	/* Phaseout rate */
local eph	34001	/* End Phaseout */

replace eitc=`cr'*y 						if y<`mi' & kids==1 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids==1 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids==1 & year==`yr'

local cr	0.4		/* credit rate */
local mi	11340	/* Minimum income for maximum credit */
local bph	16810	/* Begin Phaseout */
local pr	0.2106	/* Phaseout rate */
local eph	38348	/* End Phaseout */

replace eitc=`cr'*y 						if y<`mi' & kids>=2 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids>=2 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids>=2 & year==`yr'

*** 2004 tax schedule
local yr 104
replace tax=0.1*y 		if year==`yr'
replace tax=tax+(0.15-0.1)*(y-14300) if y>14300 & year==`yr'
replace tax=tax+(0.25-0.15)*(y-58100) if y>58100 & year==`yr'
replace tax=tax+(0.28-0.25)*(y-117250) if y>117250 & year==`yr'
replace tax=tax+(0.33-0.28)*(y-178650) if y>178650 & year==`yr'
replace tax=tax+(0.35-0.33)*(y-319100) if y>319100 & year==`yr'

* EITC 
local cr	0.0765	/* credit rate */
local mi	5100	/* Minimum income for maximum credit */
local bph	7390	/* Begin Phaseout */
local pr	0.0765	/* Phaseout rate */
local eph	12490	/* End Phaseout */

replace eitc=0 								if year==`yr'
replace eitc=`cr'*y 						if y<`mi' & kids==0 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids==0 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids==0 & year==`yr'

local cr	0.34	/* credit rate */
local mi	7660	/* Minimum income for maximum credit */
local bph	15040	/* Begin Phaseout */
local pr	0.1598	/* Phaseout rate */
local eph	31338	/* End Phaseout */

replace eitc=`cr'*y 						if y<`mi' & kids==1 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids==1 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids==1 & year==`yr'

local cr	0.4		/* credit rate */
local mi	10750	/* Minimum income for maximum credit */
local bph	15040	/* Begin Phaseout */
local pr	0.2106	/* Phaseout rate */
local eph	35458	/* End Phaseout */

replace eitc=`cr'*y 						if y<`mi' & kids>=2 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids>=2 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids>=2 & year==`yr'

*** 2002 tax schedule
local yr 102
replace tax=0.1*y 		if year==`yr'
replace tax=tax+(0.15-0.1)*(y-12000) if y>12000 & year==`yr'
replace tax=tax+(0.27-0.15)*(y-46700) if y>46700 & year==`yr'
replace tax=tax+(0.30-0.27)*(y-112850) if y>112850 & year==`yr'
replace tax=tax+(0.35-0.30)*(y-171950) if y>171950 & year==`yr'
replace tax=tax+(0.386-0.35)*(y-307050) if y>307050 & year==`yr'

* EITC 
local cr	0.0765	/* credit rate */
local mi	4910	/* Minimum income for maximum credit */
local bph	7150	/* Begin Phaseout */
local pr	0.0765	/* Phaseout rate */
local eph	12060	/* End Phaseout */

replace eitc=0 								if year==`yr'
replace eitc=`cr'*y 						if y<`mi' & kids==0 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids==0 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids==0 & year==`yr'

local cr	0.34	/* credit rate */
local mi	7370	/* Minimum income for maximum credit */
local bph	14520	/* Begin Phaseout */
local pr	0.1598	/* Phaseout rate */
local eph	30201	/* End Phaseout */

replace eitc=`cr'*y 						if y<`mi' & kids==1 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids==1 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids==1 & year==`yr'

local cr	0.4		/* credit rate */
local mi	10350	/* Minimum income for maximum credit */
local bph	14520	/* Begin Phaseout */
local pr	0.2106	/* Phaseout rate */
local eph	34178	/* End Phaseout */

replace eitc=`cr'*y 						if y<`mi' & kids>=2 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids>=2 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids>=2 & year==`yr'

*** 2000 tax schedule
local yr 100
replace tax=0.15*y 		if year==`yr'
replace tax=tax+(0.28-0.15)*(y-43050) if y>43050 & year==`yr'
replace tax=tax+(0.31-0.28)*(y-104050) if y>104050 & year==`yr'
replace tax=tax+(0.36-0.31)*(y-158550) if y>158550 & year==`yr'
replace tax=tax+(0.396-0.36)*(y-283150) if y>283150 & year==`yr'

* EITC 
local cr	0.0765	/* credit rate */
local mi	4610	/* Minimum income for maximum credit */
local bph	5770	/* Begin Phaseout */
local pr	0.0765	/* Phaseout rate */
local eph	10380	/* End Phaseout */

replace eitc=0 								if year==`yr'
replace eitc=`cr'*y 						if y<`mi' & kids==0 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids==0 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids==0 & year==`yr'

local cr	0.34	/* credit rate */
local mi	6920	/* Minimum income for maximum credit */
local bph	12690	/* Begin Phaseout */
local pr	0.1598	/* Phaseout rate */
local eph	27413	/* End Phaseout */

replace eitc=`cr'*y 						if y<`mi' & kids==1 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids==1 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids==1 & year==`yr'

local cr	0.4		/* credit rate */
local mi	9720	/* Minimum income for maximum credit */
local bph	12690	/* Begin Phaseout */
local pr	0.2106	/* Phaseout rate */
local eph	31152	/* End Phaseout */

replace eitc=`cr'*y 						if y<`mi' & kids>=2 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids>=2 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids>=2 & year==`yr'

*** 1998 tax schedule
local yr 98
replace tax=0.15*y 		if year==`yr'
replace tax=tax+(0.28-0.15)*(y-42350) if y>42350 & year==`yr'
replace tax=tax+(0.31-0.28)*(y-102300) if y>102300 & year==`yr'
replace tax=tax+(0.36-0.31)*(y-155950) if y>155950 & year==`yr'
replace tax=tax+(0.396-0.36)*(y-278450) if y>278450 & year==`yr'

* EITC 
local cr	0.0765	/* credit rate */
local mi	4460	/* Minimum income for maximum credit */
local bph	5570	/* Begin Phaseout */
local pr	0.0765	/* Phaseout rate */
local eph	10030	/* End Phaseout */

replace eitc=0 								if year==`yr'
replace eitc=`cr'*y 						if y<`mi' & kids==0 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids==0 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids==0 & year==`yr'

local cr	0.34	/* credit rate */
local mi	6680	/* Minimum income for maximum credit */
local bph	12260	/* Begin Phaseout */
local pr	0.1598	/* Phaseout rate */
local eph	26473	/* End Phaseout */

replace eitc=`cr'*y 						if y<`mi' & kids==1 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids==1 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids==1 & year==`yr'

local cr	0.4		/* credit rate */
local mi	9390	/* Minimum income for maximum credit */
local bph	12260	/* Begin Phaseout */
local pr	0.2106	/* Phaseout rate */
local eph	30095	/* End Phaseout */

replace eitc=`cr'*y 						if y<`mi' & kids>=2 & year==`yr'
replace eitc=`cr'*`mi' 						if y>`mi' & y<`bph' & kids>=2 & year==`yr'
replace eitc=`cr'*`mi' - `pr'*(y-`bph') 	if y>`bph' & y<`eph' & kids>=2 & year==`yr'


******************************
* Code foodstamps eligibility
******************************
replace year=year+1900
gen nny  = ly+wly
gen nnly = ly
gen nnwly= wly

egen countabley=rsum(nnly nnwly)

gen d2=0.2*(countabley/12)

gen d1=	134*(year<2004)+												///
		(134*(fsize<=4)+149*(fsize==5)+171*(fsize>=6))*(year==2004)+	///
		(134*(fsize<=4)+153*(fsize==5)+175*(fsize>=6))*(year==2006)+	///
		(134*(fsize<=3)+143*(fsize==4)+167*(fsize==5)+191*(fsize>=6))*(year==2008)
gen gy=max(0,(nny)/12)
gen N1=max(0,gy-d1-d2)

scalar def sc1=(115/386)
scalar def sc2=(212/386)
scalar def sc3=(304/386)
scalar def sc4=1
scalar def sc5=(459/386)
scalar def sc6=(550/386)
scalar def sc7=(608/386)
scalar def sc8=(695/386)

gen M=	(162*(fsize==1)+298*(fsize==2)+426*(fsize==3)+542*(fsize==4)+643*(fsize==5)+772*(fsize==6)+853*(fsize==7)+975*(fsize==8)+(975+122*(fsize-8))*(fsize>8))*(year==2008)+ 	///
		(152*(fsize==1)+278*(fsize==2)+399*(fsize==3)+506*(fsize==4)+601*(fsize==5)+722*(fsize==6)+798*(fsize==7)+912*(fsize==8)+(912+114*(fsize-8))*(fsize>8))*(year==2006)+	/// ///
		(141*(fsize==1)+259*(fsize==2)+371*(fsize==3)+471*(fsize==4)+560*(fsize==5)+672*(fsize==6)+743*(fsize==7)+849*(fsize==8)+(849+106*(fsize-8))*(fsize>8))*(year==2004)+	/// ///
		(135*(fsize==1)+248*(fsize==2)+356*(fsize==3)+452*(fsize==4)+537*(fsize==5)+644*(fsize==6)+712*(fsize==7)+814*(fsize==8)+(814+102*(fsize-8))*(fsize>8))*(year==2002)+	/// ///
		(127*(fsize==1)+234*(fsize==2)+335*(fsize==3)+426*(fsize==4)+506*(fsize==5)+607*(fsize==6)+671*(fsize==7)+767*(fsize==8)+(767+96*(fsize-8))*(fsize>8))*(year==2000)+	/// ///
		(122*(fsize==1)+224*(fsize==2)+321*(fsize==3)+408*(fsize==4)+485*(fsize==5)+582*(fsize==6)+643*(fsize==7)+735*(fsize==8)+(735+92*(fsize-8))*(fsize>8))*(year==1998)+	/// ///
		(119*(fsize==1)+218*(fsize==2)+313*(fsize==3)+397*(fsize==4)+472*(fsize==5)+566*(fsize==6)+626*(fsize==7)+716*(fsize>=8))*(year==1996)+	///
		(115*(fsize==1)+212*(fsize==2)+304*(fsize==3)+386*(fsize==4)+459*(fsize==5)+550*(fsize==6)+608*(fsize==7)+695*(fsize>=8))*(year==1995)+	///
		375*(sc1*(fsize==1)+sc2*(fsize==2)+sc3*(fsize==3)+sc4*(fsize==4)+sc5*(fsize==5)+sc6*(fsize==6)+sc7*(fsize==7)+sc8*(fsize>=8))*(year==1994|year==1993)+	///
		370*(sc1*(fsize==1)+sc2*(fsize==2)+sc3*(fsize==3)+sc4*(fsize==4)+sc5*(fsize==5)+sc6*(fsize==6)+sc7*(fsize==7)+sc8*(fsize>=8))*(year==1992)+	///
		352*(sc1*(fsize==1)+sc2*(fsize==2)+sc3*(fsize==3)+sc4*(fsize==4)+sc5*(fsize==5)+sc6*(fsize==6)+sc7*(fsize==7)+sc8*(fsize>=8))*(year==1991)+	///
		331*(sc1*(fsize==1)+sc2*(fsize==2)+sc3*(fsize==3)+sc4*(fsize==4)+sc5*(fsize==5)+sc6*(fsize==6)+sc7*(fsize==7)+sc8*(fsize>=8))*(year==1990)+	///
		round(331*(.79031229/.83301467))*(sc1*(fsize==1)+sc2*(fsize==2)+sc3*(fsize==3)+sc4*(fsize==4)+sc5*(fsize==5)+sc6*(fsize==6)+sc7*(fsize==7)+sc8*(fsize>=8))*(year==1989)+	///
		round(331*(.75398344/.83301467))*(sc1*(fsize==1)+sc2*(fsize==2)+sc3*(fsize==3)+sc4*(fsize==4)+sc5*(fsize==5)+sc6*(fsize==6)+sc7*(fsize==7)+sc8*(fsize>=8))*(year==1988)+	///
		round(331*(.72402805/.83301467))*(sc1*(fsize==1)+sc2*(fsize==2)+sc3*(fsize==3)+sc4*(fsize==4)+sc5*(fsize==5)+sc6*(fsize==6)+sc7*(fsize==7)+sc8*(fsize>=8))*(year==1987)+	///
		round(331*(.69853407/.83301467))*(sc1*(fsize==1)+sc2*(fsize==2)+sc3*(fsize==3)+sc4*(fsize==4)+sc5*(fsize==5)+sc6*(fsize==6)+sc7*(fsize==7)+sc8*(fsize>=8))*(year==1986)

gen fsizeold=fsize
gen kidsold=kids
replace fsize=9 if fsize>9
replace kids=8 if kids==9 & fsize==9
sort year fsize kids
merge year fsize kids using "$do/poverty_line2.dta"
drop if _merge==2
drop _merge
drop fsize kids nadults
ren fsizeold fsize
ren kidsold kids

gen 	FSB_pay=max(M-0.3*N1,10)

replace FSB_pay=0 if nny>1.3*poverty_line 	/*Eligibility gross income - ignore asset test (applies only to families without an elderly)*/
replace FSB_pay=0 if N1>poverty_line		/*Eligibility net income - ignore asset test*/
replace FSB_pay=0 if FSB_pay>0 & FSB_pay<10

*** Annualize
replace FSB_pay=12*FSB_pay
replace year=year-1900

*** Clearn
drop nny nnly nnwly countabley d1 d2 gy N1 M poverty_line


*******************************
* Predict tau and xsi by year *
*******************************
replace tax=tax-eitc-FSB_pay
gen at_y=y-tax

gen ycat = 1
replace ycat = 2 if FSB_pay!=0

gen l_y=log(y)
gen l_at_y=log(at_y)

gen tau=.
gen xsi=.
gen l_at_y_hat=.

*** We mark for drop those with benefits two times or larger than total labor income
gen psample = tax/y>-2 /* Note that this is idle when running in the estimation, and is only effective for the sample choice when called in "prepare_bootstrap" */

*** Predict tax function by year
forvalues yr=98(2)108 {
	reg l_at_y l_y 				if psample==1 & year==`yr' & ycat==1 
	replace tau=1-_b[l_y]		if psample==1 & year==`yr' & ycat==1 
	replace xsi=exp(_b[_cons])	if psample==1 & year==`yr' & ycat==1 
	predict temp 
	replace l_at_y_hat = temp	if year==`yr' & ycat==1 
	drop temp
}

*** Predict foodstamps by number of kids (not year since not enough observations)
gen kidstemp=kids
replace kidstemp=2 if kids>=2 
levelsof kidstemp, local(kd)
foreach k in `kd' {
	reg l_at_y l_y 				if psample==1 & kidstemp==`k' & ycat==2
	replace tau=1-_b[l_y]		if psample==1 & kidstemp==`k' & ycat==2
	replace xsi=exp(_b[_cons])	if psample==1 & kidstemp==`k' & ycat==2
	predict temp 
	replace l_at_y_hat = temp	if kidstemp==`k' & ycat==2
	drop temp
}
drop kidstemp

*** Check accuracy of the approximation (only if no bootstrap)
gen at_y_hat = exp(l_at_y_hat)
gen tax_bar = y-at_y_hat
gen tr = (y-at_y)/y
gen tr_bar = tax_bar/y

if $tax_graph==1 {
	cap log close
	log using "$graphs/tax_rate_fit", replace t

	reg tr_bar tr if psample==1
	
	log close

	*** Figure 2 in the paper
	label var tr "Actual"
	label var tr_bar "Predicted"
	label var y "Family labor earnings"

	#delimit;
	scatter tr tr_bar y if psample==1 & y<50000, 
			ms(+ Oh) mc(blue red) graphregion(color(white))
			yline(0, lp(-) lc(black))
			title(Average tax rate, color(black))
			legend(ring(0) position(4));
	#delimit cr
	graph export "$graphs\Figure1_tax_fit_all.eps", as(eps) replace
	graph export "$graphs\Figure1_tax_fit_all.png", as(png) width(1600) replace
	graph save "$graphs\Figure1_tax_fit_all.gph", replace

	window manage close graph _all
}

drop y l_y l_at_y at_y at_y_hat tax_bar tr tr_bar ycat l_at_y_hat 

*** If no taxes case is chosen for estimation, assign all tax rates to zero, and replace the global to the tax case so the rest runs the same
if $het_elasticities==10 {
	replace tau=0
	replace xsi=1
	global het_elasticities	= 12
}

