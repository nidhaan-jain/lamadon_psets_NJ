/* 	This file performs the construction of the sample used in the paper */

cd "$output"
u psid_cons,clear

count

/* Harmonize variables */ 
* Race: 1 is white, 2 black, 3 others
forvalues i=1/4 {
	gen     rc=race`i' if race`i'<=2
	replace rc=3    if race`i'>=3 & race`i'<=7
	drop race`i'
	ren rc race`i'

	gen     wrc=wrace`i' if wrace`i'<=2
	replace wrc=3    if wrace`i'>=3 & wrace`i'<=7
	drop wrace`i'
	ren wrc wrace`i'
}

/*
Note: Reported switch in race across panel years is common. At least one reason is that there 
----  is an option to mention more than one race, and the respondents reverse the order of the mentions.
      This is especially true when there is a slight change in the categories. In 2005 survey for example, 
	  the 3rd category was changed from "Native American" to "American Indian or Alaska Native". We deal 
	  with that by assuming that the clearest (and most stable) categories are white (1) and african-
	  american(2). Whenever a respondent mentions in any year an additional category he is assigned to the 
	  third category (other). 

*/

gen race=race1
replace race=3 if race2!=.		/* Any case of more than one reported race is also considered "other" */ 
egen race_temp = max(race), by(person)
egen race_sd = sd(race), by(person)
replace race_temp = 3 if race_sd>0 & race_sd!=.
replace race=race_temp

drop *temp race1 race2 race3 race4 race_sd 

/*
Replaces educ with educ from individual file when missing. 
Makes the variable education consistent over the years
	1 is 0-11 grades (includes those with no schooling, i.e. can't read/write people); 
    2 is 12 grades, i.e. high school w or w/o nonacademic training;
    3 is college dropout, college degree, or collage & advanced/professional degree;
    missing denotes na/dk.
*/

replace educ_ind =. if educ_ind==99
replace weduc_ind=. if weduc_ind==99 | weduc_ind==98
replace educ = educ_ind if educ==.
replace weduc = weduc_ind if weduc==.

gen sc=1 if educ>=0  & educ<=11           		/* 0-11  grades*/
replace sc=2 if educ==12                        /* High school or 12 grades+nonacademic training*/
replace sc=3 if educ>=13 & educ<=17          	/* College dropout, BA degree, or collage & adv./prof. degree*/
replace sc=. if educ>17                    		/* Missing, NA, DK*/
ren educ yearschl
ren sc educ

gen sc=1 if weduc>=0  & weduc<=11           	/* 0-11  grades*/
replace sc=2 if weduc==12                       /* High school or 12 grades+nonacademic training*/
replace sc=3 if weduc>=13 & weduc<=17          	/* College dropout, BA degree, or collage & adv./prof. degree*/
replace sc=. if weduc>17                    	/* Missing, NA, DK*/
ren weduc wyearschl
ren sc weduc

egen maxed=max(educ),by(person)
replace educ=maxed
drop maxed 

egen maxed=max(yearschl),by(person)
replace yearschl=maxed
drop maxed 	

save data4sample3, replace

*=======================================================
* Construct household with same husband-wife over time
*=======================================================
* Design a demographically unstable household as one where some family composition change (apart from changes in
* people other than head-wife takes place). Then split to new household following the change

sort person year
egen miny 		= min(year),by(person)
gen break_d 	= (fchg>1)
replace break_d	= 0 if year==miny & break_d==1
drop if break_d == 1
drop miny break_d

/* dropped the non-married heads */ 
drop if marit!=1 | agew==0

/* drop if state is missing */ 
drop if state==.|state==0|state==99

/* Drop female household heads (should be redundant because dropped by married criteria above, and for married head is always husband) */
drop if sex==2	

/* Recode age so that there is no gap or jump */
/* First drop those with always missing info on age */

egen n=sum(person!=.),by(person)
egen na=sum(age==.),by(person)
drop if n==na
drop n na

egen n=sum(person!=.),by(person)
egen na=sum(agew==.),by(person)
drop if n==na
drop n na

egen lasty=max(year) if age!=.,by(person)
gen lastage=age if year==lasty
gen b=year-lastage
replace b=0 if b==.
egen yb=sum(b),by(person)
replace age=year-yb
drop lasty lastage b

* Takes into account the retrospective nature of the data
replace age=age-1
replace agew=agew-1

/*  For intermittent "headship" (or missing from the panel from other reasons) for the year after the change, bring back the family with new id */
sort person year
qby person: gen dyear	= year-year[_n-1]
qby person: gen break_d = (dyear>2 & dyear!=.)
qby person: gen b_year 	= year if break_d == 1
egen break_year 		= min(b_year), by(person)

gen long_sample=0 			/* long_sample is equal to one for families which were broken and than put back in to the sample */ 
local ind=1
while `ind'>0 {
	sum person
	local max_id	=	r(max)
	count if year==break_year   /* for tracking */
	di r(N)						/* for tracking */ 
	replace person = person + `max_id' if year>=break_year
	replace long_sample = 1 if year>break_year
	drop dyear break_d b_year break_year
	sort person year
	qby person: gen dyear	= year-year[_n-1]
	qby person: gen break_d = (dyear>2 & dyear!=.)
	qby person: gen b_year 	= year if break_d == 1
	egen break_year 		= min(b_year), by(person)
	count if break_d == 1
	local ind=r(N)
	di `ind'					/* for tracking */
}

/* 	Final corrections for education and race for the split households (now 
	a person id is consistent with a unique spouse over time */ 

gen wrace=wrace1
replace wrace=3 if wrace2!=.		
egen wrace_temp = max(wrace), by(person)
egen wrace_sd = sd(wrace), by(person)
replace wrace_temp = 3 if wrace_sd>0 & wrace_sd!=.
replace wrace=wrace_temp

drop *temp wrace1 wrace2 wrace3 wrace4 wrace_sd 

egen maxed=max(weduc),by(person)
replace weduc=maxed
drop maxed 

egen maxed=max(wyearschl),by(person)
replace wyearschl=maxed
drop maxed 

egen lasty=max(year) if agew!=.,by(person)
gen lastage=agew if year==lasty
gen b=year-lastage
replace b=0 if b==.
egen wyb=sum(b),by(person)
replace agew=year-wyb
drop lasty lastage b

/*  Drop observations with total net worth higher than 20M$ */ 
drop if tot_assets3>=20000000&tot_assets3!=.

/* 	Before sampling by age, prepare the initial conditions for 
	variables where lagged values will be used in estimation */
tsset person year
foreach var of varlist tot_assets* {
	gen l2_`var'=l2.`var'
}

/* keep our age sample of 30-57 */
keep if age>=30 & age<=57
	
/* Remove SEO sample */ 
keep if seo==0

/* Keep only valid education data */ 	
drop if educ==. | weduc==.

save data3,replace

cd "$do"

