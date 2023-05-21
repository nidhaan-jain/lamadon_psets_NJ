/* graph pi estimated vs pi projected */ 

clear all
cd "$directory"

*************************************
* Graph pi and s by head's age categories *
*************************************
u data_het12pi1fv10cons5_rep0.dta, clear
drop _*
keep if pi!=.
replace pi=1-pi
ren s_hat s

foreach p in 5 10 25 50 75 90 95 {
	egen pi_p_`p'=pctile(pi), p(`p') by(age)
}

foreach p in 5 10 25 50 75 90 95 {
	egen s_p_`p'=pctile(s), p(`p') by(age)
}

collapse pi s pi_p* s_p* tot_assets3, by(age)
replace tot_assets3=tot_assets3/1000
label var age "Age of household head"
label var tot_assets3 "Total Assets (Thousands of Dollars)"
label var pi "Pi"
label var s "s"

#delimit:
tw 	(lowess s age, lwidth(thick) lcolor(black)) 
	, graphregion(color(white)) 
	xtitle(Age of household head) xlabel(30(5)55)
	ytitle(Average s);
#delimit cr
graph export "$graphs\Figure4_s_age.eps", as(eps) replace
graph export "$graphs\Figure4_s_age.png", as(png) width(1600) replace
graph save "$graphs\Figure4_s_age.gph", replace

#delimit:
tw 	(lowess pi_p_50 age, lwidth(thick) lcolor(black))
	(lowess pi_p_5 age, lwidth(medium) lcolor(black) lp(-))
	(lowess pi_p_10 age, lwidth(medium) lcolor(black) lp(-))
	(lowess pi_p_25 age, lwidth(medium) lcolor(black) lp(-))
	(lowess pi_p_75 age, lwidth(medium) lcolor(black) lp(-))
	(lowess pi_p_90 age, lwidth(medium) lcolor(black) lp(-))
	(lowess pi_p_95 age, lwidth(medium) lcolor(black) lp(-))
	(lowess tot_assets3 age, yaxis(2) lwidth(medthick) lp(-) lcolor(blue) ylabel(100(200)900, axis(2)))
	, graphregion(color(white)) xlabel(30(5)55)
	xtitle(Age of household head)
	ytitle("", axis(1))
	ytitle("", axis(2))
	name(panelB,replace)
	legend(label(1 {&pi} (5th to 95th Percentiles) - Left axis)
		   label(8 Total Assets (X1000) - Right axis)
		   order(1 8)
		   rows(2)
		   position(10) ring(0));
#delimit cr

#delimit:
tw 	(lowess pi age, lwidth(thick) lcolor(black))
	(lowess tot_assets3 age, yaxis(2) lwidth(medthick) lp(-) lcolor(blue))
	, graphregion(color(white)) xlabel(30(5)55)
	name(panelA,replace)
	ytitle("", axis(1))
	ytitle("", axis(2))
	xtitle(Age of household head)
	legend(label(1 {&pi} (Mean) - Left axis)
		   label(8 Total Assets (X1000) - Right axis)
		   order(1 8)
		   rows(2)
		   position(10) ring(0));
#delimit cr

graph combine panelA panelB, graphregion(color(white)) rows(2) ysize(12) xsize(9)

graph export "$graphs\Figure3_pi_age_panels.eps", as(eps) replace
graph export "$graphs\Figure3_pi_age_panels.png", as(png) width(1600) replace
graph save "$graphs\Figure3_pi_age_panels.gph", replace

cd "$do"
