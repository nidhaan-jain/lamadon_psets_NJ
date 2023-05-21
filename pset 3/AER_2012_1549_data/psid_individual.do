/* 	This file has three parts each extracting a different set of variables from the individual file */ 
	
*========================
* Part 1: Panel and head
*========================
/*
The construction of the panel that includes only heads of household. Notes: 
(1) the 'person' id generated in the end is unique per household but is arbitrary
	(in the sense that cannot be used to merge other PSID data). 
(2) we also use this file to complete individual's missing education.
(3) this file also drops the latino sample 
*/

clear all

u $PSID\ind6809,clear

#delimit;
keep  	ER30001	/* Id 1968 */
		ER33501 ER33601 ER33701 ER33801 ER33901	ER34001	/* Interview number */
        ER33502 ER33602 ER33702 ER33802 ER33902 ER34002	/* Sequence number */
      	ER33503 ER33603 ER33703 ER33803 ER33903	ER34003 /* Relationship to Head */
		ER33516 ER33616 ER33716 ER33817 ER33917	ER34020	/* Education variables */;
#delimit cr

*** We take education also from the individual file to check is it has non missing values in cases where family file had missing;
gen educ98 = ER33516 if ER33503==10
gen educ100 = ER33616 if ER33603==10
gen educ102 = ER33716 if ER33703==10
gen educ104 = ER33817 if ER33803==10
gen educ106 = ER33917 if ER33903==10
gen educ108 = ER34020 if ER34003==10

drop if ER30001>=7001 & ER30001<=9308 /*Drop latino households*/
gen seo=ER30001>=5000 & ER30001<=7000

drop if ER33503!=10 & ER33603!=10 & ER33703!=10 & ER33803!=10 & ER33903!=10 & ER34003!=10	/*drop those who are never heads*/

ren ER33501 id98
ren ER33601 id100
ren ER33701 id102
ren ER33801 id104
ren ER33901 id106
ren ER34001 id108

gen pid98=ER33503==10 & ER33502>=1 & ER33502<=20
gen pid100=ER33603==10 & ER33602>=1 & ER33602<=20
gen pid102=ER33703==10 & ER33702>=1 & ER33702<=20
gen pid104=ER33803==10 & ER33802>=1 & ER33802<=20
gen pid106=ER33903==10 & ER33902>=1 & ER33902<=20
gen pid108=ER34003==10 & ER34002>=1 & ER34002<=20

keep id* pid* seo educ* 

gen person=1
replace person=sum(person)
reshape long id pid educ,i(person) j(year)

* drop people who are no heads in that year*
drop if pid==0
ren educ educ_ind
replace educ=. if educ>=98 | educ==0 	
label var educ_ind "head's education from the individual file"

compress
sort id year
save "$output\person",replace

merge 1:1 id year using "$output\fam"
drop if _merge!=3
drop _merge

label var person "person id (not from PSID). Note that only heads are left" 
drop pid

sort id year
save "$output\fam9808_head", replace


*==============
* Part 2: Wife education from individual file
*==============
/* 	This part is used for getting the wife education from individual file, in case it is
	missing in the family file. */

clear all
u $PSID\ind6809,clear

#delimit;
keep  	ER30001 
		ER33501 ER33601 ER33701 ER33801 ER33901	ER34001	/* Interview number */
        ER33502 ER33602 ER33702 ER33802 ER33902 ER34002	/* Sequence number */
      	ER33503 ER33603 ER33703 ER33803 ER33903	ER34003 /* Relationship to Head */
		ER33516 ER33616 ER33716 ER33817 ER33917	ER34020	/* Education variables */;
#delimit cr
	
*** We take education also from the individual file to check is it has non missing values in cases where family file had missing
gen educ98 = ER33516
gen educ100 = ER33616 
gen educ102 = ER33716 
gen educ104 = ER33817 
gen educ106 = ER33917 
gen educ108 = ER34020 

drop if ER30001>=7001 & ER30001<=9308 

#delimit cr
ren ER33501 id98
ren ER33601 id100
ren ER33701 id102
ren ER33801 id104
ren ER33901 id106
ren ER34001 id108

gen pid98=ER33503==20 | ER33503==22 
gen pid100=ER33603==20 | ER33603==22 
gen pid102=ER33703==20 | ER33703==22 
gen pid104=ER33803==20 | ER33803==22 
gen pid106=ER33903==20 | ER33903==22 
gen pid108=ER34003==20 | ER34003==22 

keep id* pid* educ* 

gen person=1
replace person=sum(person)

reshape long id pid educ,i(person) j(year)
replace educ=. if educ>=98 | educ==0	/* In the individual file most zeros stand for INAP, so not using 0s */ 
egen weduc_ind = max(educ), by(person)

* drop people who are no wife in that year*
drop if pid==0
label var weduc_ind "wife's education from the individual file"
keep id year weduc_ind
sort id year
duplicates tag id year, gen(tag)	/* drop very few observations with two wives marked for same year (this is just for the education variable) */
drop if tag>0
compress
save "$output\wives",replace

sort id year
merge 1:1 id year using "$output\fam9808_head"
tab _merge
drop if _merge==1
drop _merge

save "$output\fam9808_wife", replace

*===========================
* Part 3: Age of young kids
*===========================
/* 	This part is used to recover the age of the oungest child in the family at each year */
u $PSID\ind6809,clear

#delimit;
keep  	ER30001 
		ER33501 ER33601 ER33701 ER33801 ER33901	ER34001	/* Interview number */
        ER33502 ER33602 ER33702 ER33802 ER33902 ER34002	/* Sequence number */
      	ER33503 ER33603 ER33703 ER33803 ER33903	ER34003 /* Relationship to Head */
		ER33504 ER33604 ER33704 ER33804 ER33904 ER34004; /* Age */ 

#delimit cr

foreach x of varlist  ER33504 ER33604 ER33704 ER33804 ER33904 ER34004 {
		replace `x'=. if `x'==999|`x'==0
		}
		
drop if ER30001>=7001 & ER30001<=9308 

ren ER33501 id98
ren ER33601 id100
ren ER33701 id102
ren ER33801 id104
ren ER33901 id106
ren ER34001 id108

ren ER33504 age98
ren ER33604 age100
ren ER33704 age102
ren ER33804 age104
ren ER33904 age106
ren ER34004 age108       

gen pid98=ER33503>=30 & ER33503<40 
gen pid100=ER33603>=30 & ER33603<40 
gen pid102=ER33703>=30 & ER33703<40 
gen pid104=ER33803>=30 & ER33803<40 
gen pid106=ER33903>=30 & ER33903<40 
gen pid108=ER34003>=30 & ER34003<40 

keep id* age* pid* 

gen person=1
replace person=sum(person)
reshape long id age pid,i(person) j(year)

drop if age==.

egen age_youngest = min(age) if pid==1, by(id year)

collapse age_youngest,by(id year)

replace age_youngest=19 if age_youngest>19
sort id year
save "$output/age_youngest",replace

merge 1:1 id year using "$output\fam9808_wife"
tab _merge
drop if _merge==1
drop _merge

save "$output\fam9808", replace

cd "$do"

erase "$output\fam.dta"
erase "$output\fam9808_head.dta"
erase "$output\fam9808_wife.dta"
erase "$output\person.dta"
erase "$output\wives.dta"
erase "$output\age_youngest.dta"
