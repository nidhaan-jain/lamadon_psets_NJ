*************************************************
*** Test for advanced information ***
*************************************************

cd "$output"
cap log close
log using "${graphs}\advanced_information_test", replace t
u estimation_input_all5_rhs1_part0, clear

/* Implemented as moment conditions */ 

*** Test 1: Consumption growth from t-2 to t should be orthogonal to wages growth from t+2 to t+4 and t4 to t+6
gen duc_duw_ff = duc*f4.duw
gen duc_duw_fff = duc*f6.duw
gen duc_duww_ff = duc*f4.duww
gen duc_duww_fff = duc*f6.duww

#delimit;
gmm 	(duc_duw_ff 	- {a1})
		(duc_duw_fff 	- {a2})
		(duc_duww_ff 	- {a3})
		(duc_duww_fff 	- {a4}),
		nocommonesample vce(cluster person)
		winitial(identity);
#delimit cr
test ([a1]_cons [a2]_cons [a3]_cons [a4]_cons)

log close

cd "$do"
