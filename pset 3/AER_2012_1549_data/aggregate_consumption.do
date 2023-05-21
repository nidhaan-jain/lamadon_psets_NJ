/* PSID Consumption to compare with NIPA */ 

/* 	Note: Since NIPA is national, need to use the entire PSID sample with the weights */ 

cd "$output"
clear all
set mem 100m
set more off
cap log close

****************
* Prepare PSID *
****************
u psid_cons,clear

foreach var in totcons ndcons services {
	gen weighted_`var' = weight*`var'
	egen agg_`var' = sum(weighted_`var'), by(year)
	replace agg_`var'=agg_`var'/1000000
}

collapse agg*, by(year)

ren agg_totcons agg_cons 
ren agg_ndcons agg_nd 
ren agg_services agg_s

label var agg_cons "PSID: Nondurables and Services (Millions of Dollars)"
label var agg_nd "PSID: Nondurables (Millions of Dollars)"
label var agg_s "PSID: Services (Millions of Dollars)"

foreach var of varlist agg* {
	ren `var' psid_`var'
}

replace year=year+1900
sort year
label data "psid_agg_consumption - No sampling"
save psid_agg_consumption, replace


*****************************
* Prepare NIPA Table 2.3.5 **
*****************************

/* 1: Read Nipa Data */ 
insheet using "$do/table_2_3_5_annual.csv", clear
ren v1 year
ren nondurablegoods ndur_level
ren services services_level

keep year *level

ren ndur_level nipa_agg_nd
ren services_level nipa_agg_s
gen nipa_agg_cons = nipa_agg_nd+nipa_agg_s

keep nipa* year

sort year
label data "NIPA consumption based on 2.3.5 (nominal NOT per capita)"
save nipa_cons_annual, replace

****************************************
* Merge nipa and PSID and make a table *
****************************************
u nipa_cons_annual, clear
merge year using psid_agg_consumption
keep if _m==3
drop _m

gen years="y" + string(year)
drop year
ren years _varname

xpose, clear varname
order _varname

gen sortst1=reverse(substr(reverse(_varname),1,6))
gen sortst2=substr(_varname,3,1)
sort  sortst1 sortst2
drop sortst*

outsheet using "$graphs/Table2_psid_nipa_compare.csv", comma replace

cd "$do"
