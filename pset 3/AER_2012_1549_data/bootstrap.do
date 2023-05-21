
/* Estimation including bootstrap of the main tables in the paper (4 and 6)   

Notes:
-----
The same bootstrap code is used to wrap all specifications in Tables 4 and 6. 
to replicate these tables, the columns are ran one by one. This is done by
choosing Table and Column for:
Table 4 columns 1 through 5
Table 6 columns 2 through 6 (note that column 1 of Table 6 is replicating column 1 of Table 4) 

This is done by setting the parameters at the top of the file. Setting the Table and Column
global parameter will call the correct paramfile located in the 'paramfiles' folder. 
The results will be stored in a directory in the output folder with the naming convention TiCj_bs for table i and column j.

To get the GMM standard errors for the variance estimates, need to run Table 4 columns 1 and 4 without bootstrap. 
This is done choosing global table  = 4 and:
global column = 11 	(for column 1)
global column = 41 	(for column 4)
The results will be stored in a directory in the output folder with the naming convention TiCj_gmm

Sophisticated users who wish to change parameters need to access directly the files in the 'paramfiles' folder.

*/

*===================================
* Set globals for Table and Column
*===================================
global table  = 4 	
global column = 1	
*===================================

clear 
clear matrix
clear mata
set matsize 11000
cd "$output"
set seed 1000		

*** Reads the parameter file and generates the directory for results (path defined in the parameters file)
global paramsfile 	"$do/paramfiles/T${table}C${column}" 
qui do "$paramsfile"
cap mkdir $directory

*=================
* Point estimates
*=================
global bootstraprep = 0

* The point estimates as well as the number of observations are from the rep0 (true) sample
global rep 0

u estimation_input_all${consumption}_rhs1_part${participation}, replace
qui do "$do/tax_rate_estimation_snap"
qui do "$do/pi_s_estimation"

$samplebs

qui do "$do/variance_estimation"
qui do "$do/frisch_estimation"

keep _varname value
xpose, clear
gen rep=0

if ${bn}>0 {
	save "$output/$directory/reps_${poststring}", replace
	u "$output/$directory/data_$poststring", clear
	save "$output/$directory/data_${poststring}_rep0", replace

	*===========
	* Bootstrap
	*===========
	*** Define the program bsrep 
	cap program drop bsrep
	program define bsrep
		cd "$output" 
		qui do "$paramsfile"
		u estimation_input_all${consumption}_rhs1_part${participation}, replace
		ren person idpsid
		bsample, cluster(idpsid) idcluster(person) 
		do "$do/tax_rate_estimation_snap"
		do "$do/pi_s_estimation"
		$samplebs

		do "$do/variance_estimation"
		do "$do/frisch_estimation"

		keep _varname value
		xpose, clear
		gen rep=$rep
	end 

	*** Run the program
	global rep=1
	while $rep<=$bn {
		di $rep
		global bootstraprep = 1
		cap bsrep
		local rc=_rc
		di `rc'
		if `rc'!=0 {
			keep in 1
			cap gen rep=.
			replace rep=$rep
			keep rep
		}
		qui append using "$output/$directory/reps_${poststring}"
		qui save "$output/$directory/reps_${poststring}", replace
		if ${savereps}==1 {
			qui u "$output/$directory/data_$poststring", clear
			qui save "$output/$directory/data_${rep}", replace
		}
		global rep=$rep+1
	}


	*=============================
	* Generate a table of results
	*=============================
	*** Prepare estimates table 
	cd "$output/$directory"
	u "reps_${poststring}", replace
	cap drop _*
	sort rep
	drop if pi==.
	local kr = $bskeep + 1
	keep in 1/`kr'

	foreach var of varlist  mean_k1-  s_omega {
		egen iqr_`var'_temp = iqr(`var') if rep!=0
		egen iqr_`var' = max(iqr_`var'_temp)
		gen se_`var'_iqr= iqr_`var'/1.349
		drop iqr_`var'*
	}

	if $sep!=1 {
		*===============================
		* Test for all non-separability 
		*===============================
		corr eta_c_w1 eta_c_w2 eta_h2_w1 if rep!=0, cov
		matrix vhat 	= r(C)
		matrix vhat_iqr = vhat

		local i=1
		foreach var in eta_c_w1 eta_c_w2 eta_h2_w1 {
			matrix vhat[`i',`i'] = (se_`var'_iqr[1])^2
			local i=`i'+1
		}

		scalar nrow = rowsof(vhat)
		matrix bhat = J(nrow,1,0)

		local i=1
		foreach var in eta_c_w1 eta_c_w2 eta_h2_w1 {
			su `var' if rep==0
			matrix bhat[`i',1] = r(mean)
			local i=`i'+1
		}

		cap log close
		log using "$output/$directory/test_nonsep", replace t

		matrix waldstat = bhat'*syminv(vhat)*bhat
		scalar waldstat = waldstat[1,1]
		scalar list waldstat
		display chi2tail(nrow, waldstat) /* p-value */
		display invchi2tail(nrow, .05) /* critical value */

		log close
		*********************************************************************
		}
		
	drop rep
	xpose, clear varname
	keep _v v1
	order _v

	gen index=1
	replace index = sum(index)

	gen seiqrtag=_v=="se_mean_k1_iqr"
	su index if seiqrtag==1
	gen se_iqr=v1[_n+r(mean)-1]

	ren v1 point_estimate
	su index if seiqrtag==1
	local  n=r(mean)
	drop in `n'/l

	drop index *tag

	save "$output/$directory/bootstrap_${poststring}", replace
	outsheet using "$output/$directory/bootstrap_${poststring}.csv", replace comma
	
	*These files are chaged each replication
	erase "$output/$directory/data_$poststring.dta"
	erase "$output/$directory/estimates_$poststring.dta"
	erase "$output/$directory/estimates_$poststring.csv"

}

cd "$do"
