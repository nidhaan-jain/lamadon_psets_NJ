set more off
cap log close

cd "$output"

use "data4estimation.dta", clear

*========================
* GOOD BY GOOD (TABLE 9) 
*========================
egen utilities=rsum(electric heating water miscutils)
egen transport=rsum(carins carrepair gasoline parking busfare taxifare othertrans)

g u_sh=utilities/totcons
g t_sh=transport/totcons
g f_sh=fout/totcons
replace f_sh = 0 if f_sh==.	/* replacing the zeros with missing (makes sampe identical across columns, but does not matter for results) */ 

replace weduc_ind=. if weduc_ind>17

replace hours=hours/1000
replace hourw=hourw/1000

xtset person year

g white=race==1
g log_csq=log_c^2
replace smsa=. if smsa==0|smsa==9

replace year=year+1900
sort year

merge m:1 year using "$do\cpi_detailed"

drop if _merge==2
drop _merge

replace cpi_all=ln(cpi_all)
replace cpi_foodout=ln(cpi_foodout) 
replace cpi_transp=ln(cpi_transp) 
replace cpi_hoper=ln(cpi_hoper)

local prices "cpi_all cpi_foodout cpi_transp cpi_hoper"

xtset person year

xi:ivregress 2sls u_sh (hours hourw log_c=l2.log_w l2.log_ww l2.log_y asset) i.year i.educ i.weduc white fsize kid age agew i.smsa i.state `prices',cluster(person) 
estimates store upya
xi:ivregress 2sls t_sh (hours hourw log_c=l2.log_w l2.log_ww l2.log_y asset) i.year i.educ i.weduc white fsize kid age agew i.smsa i.state  `prices',cluster(person) 
estimates store tpya
xi:ivregress 2sls f_sh (hours hourw log_c=l2.log_w l2.log_ww l2.log_y asset) i.year i.educ i.weduc white fsize kid age agew i.smsa i.state `prices',cluster(person) 
estimates store fpya
gen sample=e(sample)

xi:reg hours l2.log_w l2.log_ww l2.log_y asset i.year i.educ i.weduc white fsize kid age agew i.smsa i.state `prices' if sample==1,cluster(person) 
testparm l2.log_w l2.log_ww l2.log_y asset
matrix first=J(3,1,0)
matrix first[1,1]=r(F)
xi:reg hourw l2.log_w l2.log_ww l2.log_y asset i.year i.educ i.weduc white fsize kid age agew i.smsa i.state `prices' if sample==1,cluster(person) 
testparm l2.log_w l2.log_ww l2.log_y asset 
matrix first[2,1]=r(F)
xi:reg log_c l2.log_w l2.log_ww l2.log_y asset i.year i.educ i.weduc white fsize kid age agew i.smsa i.state `prices' if sample==1,cluster(person) 
testparm l2.log_w l2.log_ww l2.log_y asset
matrix first[3,1]=r(F)

log using "$graphs/Table9", replace t
estimates table upya tpya fpya,keep(hours hourw log_c) b(%7.4f) se(%7.4f) t stats(N)
matlist first
log close

*========================
* EULER EQUATION (TABLE 8)
*========================
gen partw=log_ww!=.

egen partwm=mean(partw),by(weduc_ind agew year )

g logc=log(totcons)
g dlogc =logc -l2.logc

g dloghw=2*(hourw-l2.hourw)/(hourw+l2.hourw)
g dloghh=2*(hours-l2.hours)/(hours+l2.hours)

gen 	educde 	= 1	if yearschl!=.					
replace educde 	= 2 if yearschl>4 & yearschl!=.
replace educde 	= 3 if yearschl>8 & yearschl!=.
replace educde 	= 4 if yearschl==12 & yearschl!=.
replace educde 	= 5 if yearschl>12 & yearschl!=.
replace educde 	= 6 if yearschl>14 & yearschl!=.
replace educde 	= 7 if yearschl==16 & yearschl!=.
replace educde 	= 8 if yearschl>16 & yearschl!=.

gen 	weducde 	= 1	if wyearschl!=.
replace weducde 	= 2 if wyearschl>4 & wyearschl!=.
replace weducde 	= 3 if wyearschl>8 & wyearschl!=.
replace weducde 	= 4 if wyearschl==12 & wyearschl!=.
replace weducde 	= 5 if wyearschl>12 & wyearschl!=.
replace weducde 	= 6 if wyearschl>14 & wyearschl!=.
replace weducde 	= 7 if wyearschl==16 & wyearschl!=.
replace weducde 	= 8 if wyearschl>16 & wyearschl!=.

egen wagewm=mean(log_ww),by(weducde agew year )
egen wagehm=mean(log_w),by(educde age year )

g dpartw=partw-l2.partw
g dwagewm=wagewm-l2.wagewm
g dwagehm=wagehm-l2.wagehm
g dfsize=fsize-l2.fsize
g dkids=kids-l2.kids

g age2=age^2
g agew2=agew^2

g l2wagehm=l2.wagehm 
g l2wagewm=l2.wagewm 
g l2partwm=l2.partwm

* Column 2
xi:ivregress 2sls dlogc (dloghw dloghh dpartw=wagewm wagehm partwm l2wagehm l2wagewm l2partwm) i.state white age agew age2 agew2 i.year dfsize dkids i.educde i.weducde, cluster(person) 
estimates store T8C2
drop sample
gen sample=e(sample)

matrix first=J(4,2,0)
xi:reg dloghh wagewm wagehm partwm l2wagehm l2wagewm l2partwm i.state white age agew age2 agew2 i.year dfsize dkids i.educde i.weducde if sample==1,cluster(person) 
testparm wagewm wagehm partwm l2wagehm l2wagewm l2partwm
matrix first[2,2]=r(F)

xi:reg dpartw wagewm wagehm partwm l2wagehm l2wagewm l2partwm i.state white age agew age2 agew2 i.year dfsize dkids i.educde i.weducde if sample==1,cluster(person) 
testparm wagewm wagehm partwm l2wagehm l2wagewm l2partwm
matrix first[3,2]=r(F)

xi:reg dloghw wagewm wagehm partwm l2wagehm l2wagewm l2partwm i.state white age agew age2 agew2 i.year dfsize dkids i.educde i.weducde if sample==1,cluster(person) 
testparm wagewm wagehm partwm l2wagehm l2wagewm l2partwm
matrix first[4,2]=r(F)

***and the same looking at head's participation as well
use "data_sample2.dta", clear

xtset person year
g white=race==1

g dfsize=fsize-l2.fsize
g dkids=kids-l2.kids
g age2=age^2
g agew2=agew^2
replace weduc_ind=. if weduc_ind>17

gen partw=log_ww!=.
gen parth=log_w!=.

egen partwm=mean(partw),by(weduc_ind agew year )
egen parthm=mean(parth),by(educ_ind age year )

g logc=log(totcons)
g dlogc =logc -l2.logc

g dloghw=2*(hourw-l2.hourw)/(hourw+l2.hourw)
g dloghh=2*(hours-l2.hours)/(hours+l2.hours)

g dhw=(hourw-l2.hourw)
g dhh=(hours-l2.hours)

g dpartw=partw-l2.partw
g dparth=parth-l2.parth

gen 	educde 	= 1	if yearschl!=.
replace educde 	= 2 if yearschl>4 & yearschl!=.
replace educde 	= 3 if yearschl>8 & yearschl!=.
replace educde 	= 4 if yearschl==12 & yearschl!=.
replace educde 	= 5 if yearschl>12 & yearschl!=.
replace educde 	= 6 if yearschl>14 & yearschl!=.
replace educde 	= 7 if yearschl==16 & yearschl!=.
replace educde 	= 8 if yearschl>16 & yearschl!=.

gen 	weducde 	= 1	if wyearschl!=.
replace weducde 	= 2 if wyearschl>4 & wyearschl!=.
replace weducde 	= 3 if wyearschl>8 & wyearschl!=.
replace weducde 	= 4 if wyearschl==12 & wyearschl!=.
replace weducde 	= 5 if wyearschl>12 & wyearschl!=.
replace weducde 	= 6 if wyearschl>14 & wyearschl!=.
replace weducde 	= 7 if wyearschl==16 & wyearschl!=.
replace weducde 	= 8 if wyearschl>16 & wyearschl!=.

egen wagewm=mean(log_ww),by(weducde agew year )
egen wagehm=mean(log_w),by(educde age year )

g dwagewm=wagewm-l2.wagewm
g dwagehm=wagehm-l2.wagehm

g l2wagehm=l2.wagehm 
g l2wagewm=l2.wagewm 
g l2partwm=l2.partwm
g l2parthm=l2.parthm

* Column 1
xi:ivreg2 		dlogc (dloghw dloghh dpartw dparth=wagewm wagehm partwm parthm l2wagehm l2wagewm l2partwm l2parthm ) white age agew age2 agew2 i.year dfsize dkids i.state i.educde i.weducde, ffirst cluster(person) partial(i.state)
xi:ivregress 2sls dlogc (dloghw dloghh dpartw dparth=wagewm wagehm partwm parthm l2wagehm l2wagewm l2partwm l2parthm ) white age agew age2 agew2 i.year dfsize dkids i.state i.educde i.weducde, cluster(person) 
estimates store T8C1
gen sample=e(sample)

xi:reg dparth wagewm wagehm partwm parthm l2wagehm l2wagewm l2partwm l2parthm white age agew age2 agew2 i.year dfsize dkids i.state i.educde i.weducde if sample==1,cluster(person) 
testparm wagewm wagehm partwm parthm l2wagehm l2wagewm l2partwm l2parthm
matrix first[1,1]=r(F)

xi:reg dloghh wagewm wagehm partwm parthm l2wagehm l2wagewm l2partwm l2parthm white age agew age2 agew2 i.year dfsize dkids i.state i.educde i.weducde if sample==1,cluster(person) 
testparm wagewm wagehm partwm parthm l2wagehm l2wagewm l2partwm l2parthm
matrix first[2,1]=r(F)

xi:reg dpartw wagewm wagehm partwm parthm l2wagehm l2wagewm l2partwm l2parthm white age agew age2 agew2 i.year dfsize dkids i.state i.educde i.weducde if sample==1,cluster(person) 
testparm wagewm wagehm partwm parthm l2wagehm l2wagewm l2partwm l2parthm
matrix first[3,1]=r(F)

xi:reg dloghw wagewm wagehm partwm parthm l2wagehm l2wagewm l2partwm l2parthm white age agew age2 agew2 i.year dfsize dkids i.state i.educde i.weducde if sample==1,cluster(person) 
testparm wagewm wagehm partwm parthm l2wagehm l2wagewm l2partwm l2parthm
matrix first[4,1]=r(F)

log using "$graphs/Table8", replace t
estimates table T8C1 T8C2,keep(dparth dloghh dpartw dloghw) b(%7.3f) se(%7.3f) t stats(N)
matlist first
log close

*================================
* ADDED WORKER EFFECT (TABLE 7)
*================================
use "data3_resid_cons5_part0.dta", clear

xtset person year
g duw=uw
lab var duw "Wage res in FD, husband"

gen partw=log_ww!=.

sort person year
gen dpart=partw-l2.partw
gen IV_tran=-f2.duw
gen IV_perm=(l2.duw+duw+f2.duw)
gen IV_perm2=IV_perm^2
gen IV_tran2=IV_tran^2
gen agew2=agew^2
g age2=age^2

sort person year
gen dage_y=age_youngest-l2.age_youngest

gen nb = age_y==1
gen dnb = nb-l2.nb

xi: ivregress 2sls dpart (duw=IV_perm*) i.year agew agew2 age age2 i.weduc i.educ dkids dage_y dnb i.state i.smsa,cluster(person)
estimates store T9C1
gen sample1 = e(sample)

xi: ivregress 2sls dpart (duw=IV_tran*) i.year agew agew2 age age2 i.weduc i.educ dkids dage_y dnb i.state i.smsa,cluster(person)
estimates store T9C2
gen sample2 = e(sample)

matrix first=J(1,2,0)
xi: reg duw IV_perm* i.year agew agew2 age age2 i.weduc i.educ dkids dage_y dnb i.state i.smsa if sample1==1,cluster(person)
testparm IV*
matrix first[1,1]=r(F)

xi: reg duw IV_tran* i.year agew agew2 age age2 i.weduc i.educ dkids dage_y dnb i.state i.smsa if sample1==1,cluster(person)
testparm IV*
matrix first[1,2]=r(F)

log using "$graphs/Table7", replace t
estimates table T9C1 T9C2,keep(duw dkids dage_y) b(%7.3f) se(%7.3f) t stats(N)
matlist first
log close
 
