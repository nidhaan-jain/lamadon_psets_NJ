/* 	Generate consumption and assets measures at the household level
	Notes: 
	(1) we assume that missing for a category is zero for this category. 
	(2) we add food stamps to the nd data 
	(3) Rent equivalent is set to 0.06 of house price
	*/ 

cd "$output"
u fam9808,clear
	
/* Rent equivalent */
gen renteq 		= 0
replace renteq 	= 0.06*house if rent==.

/* Consumption */
egen ndcons 	= 	rsum(food fstmp gasoline), missing 
egen services 	= 	rsum(hinsurance nurse doctor prescription homeinsure electric heating water miscutils ///
					carins carrepair parking busfare taxifare othertrans tuition otherschool childcare rent renteq fout), missing /* note that totalhealth is not used since the sub components are used */ 
egen totcons	= 	rsum(ndcons services), missing
egen tempr 		= 	rsum(rent renteq)
gen totcons_nh  =   totcons - tempr
drop tempr

label var ndcons "nondurable consumption, missing categories set to zeros"
label var services "services consumption, missing categories set to zeros"
label var totcons "nondurable and services consumption"
label var totcons_nh "same as totcons, excluding housing"

/* Assets */ 
gen other_debt_temp = -other_debt
gen mortgage1_temp 	= - mortgage1
gen mortgage2_temp 	= - mortgage2

su other_debt* mortgage1* mortgage2*

egen tot_assets3 = rsum(cash bonds stocks busval penval other_debt_temp house real_estate carval mortgage1_temp mortgage2_temp)

label var tot_assets3 "cash + bonds + stocks - other_debt + house + real_estate + carval - mortgage1 - mortgage2"

drop *_temp

save psid_cons, replace
 
cd "$do" 

erase "$output/fam9808.dta"
