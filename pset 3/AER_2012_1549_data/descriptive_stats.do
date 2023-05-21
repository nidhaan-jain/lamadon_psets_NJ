/* Descriptive Stats - Table 1 */ 

/*
*********************************************************************************
This file generates the descriptive statistics for 3 samples:
(1) The sample used in estimation 
(2) The baseline sample + non-participating males aged 30-57
(3) All households with male aged 30-57 in hh, ever married during our sample
All variables in nominal values. In addition, for samples 2-3 we do perform the 
standard validity checks on the data, of dropping households with missing basic information
(education etc.), and not using probably measurement error observations in particular 
variables when reporting volatility (wage much lower than min wage, extreme changes). 
Note that the table generates also descriptive stats by year (from earlier versions of the paper)
*********************************************************************************
*/

clear all
cd "$output" 

u data4estimation, clear

*============================
* Sample 1 - Baseline sample
*============================
cap postclose table1
postfile table1 str64 var y1998 y2000 y2002 y2004 y2006 y2008 allyears allyears_med using descriptive_stats, replace

*** Consumption
gen rent_all = rent
replace rent_all= renteq if rent==.
egen health_ser = rsum(nurse doctor prescription), missing
egen utilities 	= rsum(electric heating water miscutils), missing
egen transport 	= rsum(carins carrepair parking busfare taxifare othertrans), missing
egen educ_ser	= rsum(tuition otherschool), missing
gen equiv = sqrt(fsize) 
gen totcons_eq = totcons/equiv
gen ndcons_eq = ndcons/equiv
gen services_eq = services/equiv

label var totcons 			"Consumption" 
label var totcons_eq 		"Consumption (equivalence scale)" 
label var ndcons 			"    Nondurable Consumption"
label var ndcons_eq 		"    Nondurable Consumption (equivalence scale)"
label var food 				"        Food (at home)"
label var gasoline 			"        Gasoline" 
label var services 			"    Services"
label var services_eq 		"    Services (equivalence scale)"
label var fout				"        Food (out)"
label var hinsurance 		"        Health Insurance"
label var homeinsure	 	"        Home Insurance"
label var health_ser 		"        Health Services"
label var utilities			"        Utilities"
label var transport			"        Transportation" 
label var educ_ser			"        Education"
label var childcare			"        Child Care"
label var rent_all			"        Rent (or rent equivalent)" 


foreach var in 	totcons totcons_eq ndcons ndcons_eq food gasoline ///
				services services_eq fout hinsurance health_ser utilities transport educ_ser childcare homeinsure rent_all {
	replace `var' = 0 if `var'==.
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y'
		local mean`y' = round(r(mean),1)
	}
	su `var',de
	local mean_all = round(r(mean),1)
	local med_all  = round(r(p50),1)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}

local blank = .
post table1 ("   ") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')

*** Assets
egen assets 	= rsum(cash bonds stocks busval penval house real_estate carval), missing
egen house_re	= rsum(house real_estate), missing
egen assets_ot	= rsum(cash bonds stocks busval penval carval), missing
egen tot_debt 	= rsum(other_debt mortgage1 mortgage2), missing
egen mortgage	= rsum(mortgage1 mortgage2), missing

gen tot_debt_temp = -tot_debt
egen networth = rsum(tot_debt_temp assets)
egen netfinworth = rsum(cash bonds stocks penval)

label var assets 		"Total assets"
label var house_re		"    Housing and RE assets"
label var assets_ot 	"    Financial assets"
label var tot_debt		"Total debt"
label var mortgage		"    Mortgage"
label var other_debt	"    Other debt" 
label var networth 		"Total Net Worth"
label var netfinworth 		"Total Net Financial Worth"

foreach var in 	assets house_re assets_ot tot_debt mortgage other_debt networth netfinworth {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y' & networth>=-1000000 & networth<20000000	/* There are 6  observations with extreme negative values for assets_ot which look like an outlier */
		local mean`y' = round(r(mean),1)
	}
	su `var' , de
	local mean_all = round(r(mean),1) 
	local med_all = round(r(p50),1) 
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
post table1 ("   ") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')

*** Wages and earnings first earner
label var ly		"    Earnings"
label var hours		"    Hours worked" 

gen ba = educ>=3
label var ba		"    Share with some college"
gen wba = weduc>=3 & weduc!=.
label var wba		"    Share with some college"
gen head_employed = 1

post table1 ("First earner (head)") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')
foreach var in head_employed {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y'
		local mean`y' = round(r(mean),0.01)
	}
	su `var', de
	local mean_all = round(r(mean),0.01)
	local med_all = round(r(p50),0.01)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
foreach var in ly hours {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y'
		local mean`y' = round(r(mean),1)
	}
	su `var', de
	local mean_all = round(r(mean),1)
	local med_all = round(r(p50),1)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
foreach var in ba  {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y'
		local mean`y' = round(r(mean),0.01)
	}
	su `var', de
	local mean_all = round(r(mean),0.01)
	local med_all = round(r(p50),0.01)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
post table1 ("   ") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')

*** Wages and earnings second earner
label var wly		"    Earnings (conditional on participation)"
label var hourw		"    Hours worked (conditional on participation)" 
label var wife_employed "    Participation rate"

post table1 ("Second earner (wife)") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')
foreach var in wife_employed {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y'
		local mean`y' = round(r(mean),0.01)
	}
	su `var', de
	local mean_all = round(r(mean),0.01)
	local med_all = round(r(p50),0.01)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
foreach var in wly hourw {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y' & wife_employed==1
		local mean`y' = round(r(mean),1)
	}
	su `var' if wife_employed==1,de
	local mean_all = round(r(mean),1)
	local med_all = round(r(p50),1)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
foreach var in wba  {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y'
		local mean`y' = round(r(mean),0.01)
	}
	su `var', de
	local mean_all = round(r(mean),0.01)
	local med_all = round(r(p50),0.01)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
post table1 ("   ") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')

*** Volatility
tsset person year
foreach var in log_w log_ww log_y log_yw log_c {
	gen d`var'=`var'-l2.`var'
	label var d`var' d`var'
	local vlab: variable label d`var'
	forvalues y=98(2)108 {
		su d`var' if year==`y'
		local mean`y' = round(r(Var),0.001)
	}
	su d`var'
	local mean_all = round(r(Var),0.001)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`blank')
}	


*** Count number of observations
forvalues y=98(2)108 {
	count if year==`y'
	local mean`y' = r(N)
}
count 
local mean_all = r(N)
post table1 ("Observarions") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`blank')

postclose table1

u descriptive_stats, clear
format %-40s var
outsheet using "$graphs\Table1_C1C2_descriptive_stats.csv", comma replace



*===============================================
* Sample 2: Baseline + non-participating males
*===============================================
cd "$output"
u data4estimation_nopart, clear

*** Apply the cleaning as before (minimum wage already applied, so just the percentile of jumps)
cap drop log_w
gen log_w 	= log(ly/hours) - log(price)
cap drop log_ww
gen log_ww	= log(wly/hourw) - log(price)
cap drop log_y
gen log_y 	= log(ly) - log(price)
cap drop log_yw
gen log_yw 	= log(wly) - log(price)
cap drop log_c
gen log_c	= log(totcons) - log(price)
gen head_employed = log_w!=.

global pec_drop     = 0.25
tsset person year
local npec = 100/${pec_drop}
di `npec'

foreach var of varlist log_* {
	gen d_`var' = `var' - l2.`var'
	gen d_`var'_lag = d_`var'*l2.d_`var'
}

foreach var of varlist d_*_lag {
	egen pec_`var'=  xtile(`var'), by(year) n(`npec') 
}

foreach var of varlist log_*  {
	replace `var'=. if f2.pec_d_`var'_lag==1 	
}

foreach var of varlist log_* {
	gen d_`var'_trunc = `var' - l2.`var'
	gen d_`var'_trunc_lag = d_`var'_trunc*l2.d_`var'_trunc
}
su d_*_lag
drop d_* pec_*

replace log_w = . if log_y==. 
replace log_y = . if log_w==. 

replace log_ww = . if log_yw==. 
replace log_yw = . if log_ww==. 

gen wife_employed = log_ww!=.
cap drop _m

save data_sample2, replace

*** Generate the Descriptive Statistics table
cap postclose table1
postfile table1 str64 var y1998 y2000 y2002 y2004 y2006 y2008 allyears allyears_med using descriptive_stats_nopart, replace

*** Consumption
gen rent_all = rent
replace rent_all= renteq if rent==.
egen health_ser = rsum(nurse doctor prescription), missing
egen utilities 	= rsum(electric heating water miscutils), missing
egen transport 	= rsum(carins carrepair parking busfare taxifare othertrans), missing
egen educ_ser	= rsum(tuition otherschool), missing
gen equiv = sqrt(fsize) 
gen totcons_eq = totcons/equiv
gen ndcons_eq = ndcons/equiv
gen services_eq = services/equiv

label var totcons 			"Consumption" 
label var totcons_eq 		"Consumption (equivalence scale)" 
label var ndcons 			"    Nondurable Consumption"
label var ndcons_eq 		"    Nondurable Consumption (equivalence scale)"
label var food 				"        Food (at home)"
label var gasoline 			"        Gasoline" 
label var services 			"    Services"
label var services_eq 		"    Services (equivalence scale)"
label var fout				"        Food (out)"
label var hinsurance 		"        Health Insurance"
label var homeinsure	 	"        Home Insurance"
label var health_ser 		"        Health Services"
label var utilities			"        Utilities"
label var transport			"        Transportation" 
label var educ_ser			"        Education"
label var childcare			"        Child Care"
label var rent_all			"        Rent (or rent equivalent)" 


foreach var in 	totcons totcons_eq ndcons ndcons_eq food gasoline ///
				services services_eq fout hinsurance health_ser utilities transport educ_ser childcare homeinsure rent_all {
	replace `var' = 0 if `var'==.
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y'
		local mean`y' = round(r(mean),1)
	}
	su `var',de
	local mean_all = round(r(mean),1)
	local med_all  = round(r(p50),1)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}

local blank = .
post table1 ("   ") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')

*** Assets
egen assets 	= rsum(cash bonds stocks busval penval house real_estate carval), missing
egen house_re	= rsum(house real_estate), missing
egen assets_ot	= rsum(cash bonds stocks busval penval carval), missing
egen tot_debt 	= rsum(other_debt mortgage1 mortgage2), missing
egen mortgage	= rsum(mortgage1 mortgage2), missing

gen tot_debt_temp = -tot_debt
egen networth = rsum(tot_debt_temp assets)
egen netfinworth = rsum(cash bonds stocks penval)

label var assets 		"Total assets"
label var house_re		"    Housing and RE assets"
label var assets_ot 	"    Financial assets"
label var tot_debt		"Total debt"
label var mortgage		"    Mortgage"
label var other_debt	"    Other debt" 
label var networth 		"Total Net Worth"
label var netfinworth 		"Total Net Financial Worth"

foreach var in 	assets house_re assets_ot tot_debt mortgage other_debt networth netfinworth {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y' & networth>=-1000000 & networth<20000000	/* There are 6  observations with extreme negative values for assets_ot which look like an outlier */
		local mean`y' = round(r(mean),1)
	}
	su `var' , de
	local mean_all = round(r(mean),1) 
	local med_all = round(r(p50),1) 
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
post table1 ("   ") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')

*** Wages and earnings first earner
label var ly		"    Earnings"
label var hours		"    Hours worked" 

gen ba = educ>=3
label var ba		"    Share with some college"
gen wba = weduc>=3 & weduc!=.
label var wba		"    Share with some college"

post table1 ("First earner (head)") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')
foreach var in head_employed {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y'
		local mean`y' = round(r(mean),0.01)
	}
	su `var', de
	local mean_all = round(r(mean),0.01)
	local med_all = round(r(p50),0.01)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
foreach var in ly hours {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y' & ly!=0 & hours!=0
		local mean`y' = round(r(mean),1)
	}
	su `var' if ly!=0 & hours!=0, de
	local mean_all = round(r(mean),1)
	local med_all = round(r(p50),1)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
foreach var in ba  {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y'
		local mean`y' = round(r(mean),0.01)
	}
	su `var', de
	local mean_all = round(r(mean),0.01)
	local med_all = round(r(p50),0.01)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
post table1 ("   ") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')

*** Wages and earnings second earner
label var wly		"    Earnings (conditional on participation)"
label var hourw		"    Hours worked (conditional on participation)" 
label var wife_employed "    Participation rate"

post table1 ("Second earner (wife)") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')
foreach var in wife_employed {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y'
		local mean`y' = round(r(mean),0.01)
	}
	su `var', de
	local mean_all = round(r(mean),0.01)
	local med_all = round(r(p50),0.01)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
foreach var in wly hourw {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y' & wife_employed==1
		local mean`y' = round(r(mean),1)
	}
	su `var' if wife_employed==1,de
	local mean_all = round(r(mean),1)
	local med_all = round(r(p50),1)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
foreach var in wba  {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y'
		local mean`y' = round(r(mean),0.01)
	}
	su `var', de
	local mean_all = round(r(mean),0.01)
	local med_all = round(r(p50),0.01)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}

post table1 ("   ") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')

*** Volatility 
tsset person year
foreach var in log_w log_ww log_y log_yw log_c {
	gen d`var'=`var'-l2.`var'
	label var d`var' d`var'
	local vlab: variable label d`var'
	forvalues y=98(2)108 {
		su d`var' if year==`y'
		local mean`y' = round(r(Var),0.001)
	}
	su d`var'
	local mean_all = round(r(Var),0.001)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`blank')
}	


*** Count number of observations
forvalues y=98(2)108 {
	count if year==`y'
	local mean`y' = r(N)
}
count 
local mean_all = r(N)
post table1 ("Observarions") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`blank')

postclose table1

u descriptive_stats_nopart, clear
format %-40s var
outsheet using "$graphs\Table1_C3C4_descriptive_stats_nopart.csv", comma replace


*============================================
* Sample 3: All households with males in hh ever married during our sample
*============================================
cd "$output"
u "data4sample3",clear	

*** Apply the cleaning as before 

/* drop if state is missing */ 
drop if state==.|state==0|state==99

/* Drop female household heads*/
drop if sex==2

/* We drop observations with total net worth higher than 20M$ */ 
drop if tot_assets3>=20000000&tot_assets3!=.

/* Sample age 30-57*/ 
drop if age<30 | age>57 

sort person year

/* 	Finally, we do want to focus only on those who had been married at 
	least for one year during the sample, and we do not want to include 
	the SEO sample
	*/
	
gen married=marit==1
egen max_married=max(married), by(person)
drop if max_married==0
keep if seo==0

/* Apply minimum wage cleaning */ 
sort state
merge state using "$do\psid_states"
drop _merge

sort state_st year
merge state_st year using "$do\min_wage"
drop if _merge!=3
drop _merge

sort year
merge year using "$do\price_indices"
keep if _m==3
drop _m
gen price=p_totcons

gen oly = ly
gen owly = wly
replace ly=.  if (ly/hours)<0.5*min_wage
replace wly=. if (wly/hourw)<0.5*min_wage

cap drop log_w
gen log_w 	= log(ly/hours) - log(price)
cap drop log_ww
gen log_ww	= log(wly/hourw) - log(price)
cap drop log_y
gen log_y 	= log(ly) - log(price)
cap drop log_yw
gen log_yw 	= log(wly) - log(price)
cap drop log_c
gen log_c	= log(totcons) - log(price)
gen wife_employed = log_ww!=.
cap drop _m
gen head_employed = log_w!=.


* Apply the cleaning of percentile of jumps 
global pec_drop     = 0.25
tsset person year
local npec = 100/${pec_drop}
di `npec'

foreach var of varlist log_* {
	gen d_`var' = `var' - l2.`var'
	gen d_`var'_lag = d_`var'*l2.d_`var'
}

foreach var of varlist d_*_lag {
	egen pec_`var'=  xtile(`var'), by(year) n(`npec') 
}

foreach var of varlist log_*  {
	replace `var'=. if f2.pec_d_`var'_lag==1 	/* 	Note that we are only assinging missing values to the year with the jump */ 
}

foreach var of varlist log_* {
	gen d_`var'_trunc = `var' - l2.`var'
	gen d_`var'_trunc_lag = d_`var'_trunc*l2.d_`var'_trunc
}
su d_*_lag
drop d_* pec_*

replace log_w = . if log_y==. 
replace log_y = . if log_w==. 

replace log_ww = . if log_yw==. 
replace log_yw = . if log_ww==. 

drop if educ==.

save data_noselection,replace


*** Generate the Descriptive Statistics table
cap postclose table1
postfile table1 str64 var y1998 y2000 y2002 y2004 y2006 y2008 allyears allyears_med using descriptive_stats_noselect, replace

*** Consumption
gen rent_all = rent
replace rent_all= renteq if rent==.
egen health_ser = rsum(nurse doctor prescription), missing
egen utilities 	= rsum(electric heating water miscutils), missing
egen transport 	= rsum(carins carrepair parking busfare taxifare othertrans), missing
egen educ_ser	= rsum(tuition otherschool), missing
gen equiv = sqrt(fsize) 
gen totcons_eq = totcons/equiv
gen ndcons_eq = ndcons/equiv
gen services_eq = services/equiv

label var totcons 			"Consumption" 
label var totcons_eq 		"Consumption (equivalence scale)" 
label var ndcons 			"    Nondurable Consumption"
label var ndcons_eq 		"    Nondurable Consumption (equivalence scale)"
label var food 				"        Food (at home)"
label var gasoline 			"        Gasoline" 
label var services 			"    Services"
label var services_eq 		"    Services (equivalence scale)"
label var fout				"        Food (out)"
label var hinsurance 		"        Health Insurance"
label var homeinsure	 	"        Home Insurance"
label var health_ser 		"        Health Services"
label var utilities			"        Utilities"
label var transport			"        Transportation" 
label var educ_ser			"        Education"
label var childcare			"        Child Care"
label var rent_all			"        Rent (or rent equivalent)" 


foreach var in 	totcons totcons_eq ndcons ndcons_eq food gasoline ///
				services services_eq fout hinsurance health_ser utilities transport educ_ser childcare homeinsure rent_all {
	replace `var' = 0 if `var'==.
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y'
		local mean`y' = round(r(mean),1)
	}
	su `var',de
	local mean_all = round(r(mean),1)
	local med_all  = round(r(p50),1)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}

local blank = .
post table1 ("   ") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')

*** Assets
egen assets 	= rsum(cash bonds stocks busval penval house real_estate carval), missing
egen house_re	= rsum(house real_estate), missing
egen assets_ot	= rsum(cash bonds stocks busval penval carval), missing
egen tot_debt 	= rsum(other_debt mortgage1 mortgage2), missing
egen mortgage	= rsum(mortgage1 mortgage2), missing

gen tot_debt_temp = -tot_debt
egen networth = rsum(tot_debt_temp assets)
egen netfinworth = rsum(cash bonds stocks penval)

label var assets 		"Total assets"
label var house_re		"    Housing and RE assets"
label var assets_ot 	"    Financial assets"
label var tot_debt		"Total debt"
label var mortgage		"    Mortgage"
label var other_debt	"    Other debt" 
label var networth 		"Total Net Worth"
label var netfinworth 		"Total Net Financial Worth"

foreach var in 	assets house_re assets_ot tot_debt mortgage other_debt networth netfinworth {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y' & networth>=-1000000 & networth<20000000	/* There are 6  observations with extreme negative values for assets_ot which look like an outlier */
		local mean`y' = round(r(mean),1)
	}
	su `var' , de
	local mean_all = round(r(mean),1) 
	local med_all = round(r(p50),1) 
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
post table1 ("   ") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')

*** Wages and earnings first earner
label var ly		"    Earnings"
label var hours		"    Hours worked" 

gen ba = educ>=3
label var ba		"    Share with some college"
gen wba = weduc>=3 & weduc!=.	if weduc!=.	/* This is important for this sample as some households don't have a wife */ 
label var wba		"    Share with some college"

post table1 ("First earner (head)") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')
foreach var in head_employed {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y'
		local mean`y' = round(r(mean),0.01)
	}
	su `var', de
	local mean_all = round(r(mean),0.01)
	local med_all = round(r(p50),0.01)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
foreach var in ly hours {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y' & ly!=0 & hours!=0
		local mean`y' = round(r(mean),1)
	}
	su `var' if ly!=0 & hours!=0, de
	local mean_all = round(r(mean),1)
	local med_all = round(r(p50),1)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
foreach var in ba  {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y'
		local mean`y' = round(r(mean),0.01)
	}
	su `var', de
	local mean_all = round(r(mean),0.01)
	local med_all = round(r(p50),0.01)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
post table1 ("   ") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')

*** Wages and earnings second earner
label var wly		"    Earnings (conditional on participation)"
label var hourw		"    Hours worked (conditional on participation)" 
label var wife_employed "    Participation rate"

post table1 ("Second earner (wife)") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')
foreach var in wife_employed {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y'
		local mean`y' = round(r(mean),0.01)
	}
	su `var', de
	local mean_all = round(r(mean),0.01)
	local med_all = round(r(p50),0.01)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
foreach var in wly hourw {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y' & wife_employed==1
		local mean`y' = round(r(mean),1)
	}
	su `var' if wife_employed==1,de
	local mean_all = round(r(mean),1)
	local med_all = round(r(p50),1)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}
foreach var in wba  {
	local vlab: variable label `var'
	forvalues y=98(2)108 {
		su `var' if year==`y'
		local mean`y' = round(r(mean),0.01)
	}
	su `var', de
	local mean_all = round(r(mean),0.01)
	local med_all = round(r(p50),0.01)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`med_all')
}

post table1 ("   ") (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank') (`blank')

*** Volatility 
tsset person year
foreach var in log_w log_ww log_y log_yw log_c {
	gen d`var'=`var'-l2.`var'
	label var d`var' d`var'
	local vlab: variable label d`var'
	forvalues y=98(2)108 {
		su d`var' if year==`y'
		local mean`y' = round(r(Var),0.001)
	}
	su d`var'
	local mean_all = round(r(Var),0.001)
	post table1 ("`vlab'") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`blank')
}	


*** Count number of observations
forvalues y=98(2)108 {
	count if year==`y'
	local mean`y' = r(N)
}
count 
local mean_all = r(N)
post table1 ("Observarions") (`mean98') (`mean100') (`mean102') (`mean104') (`mean106') (`mean108') (`mean_all') (`blank')

postclose table1
u descriptive_stats_noselect, clear
format %-40s var
outsheet using "$graphs\Table1_C5C6_descriptive_stats_noselect.csv", comma replace

cd "$do"
