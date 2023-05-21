/* 	This file reads PSID variables from 1999 to 2009. */
clear all
set mem 400m
set more off

****************************************************************************************************
****************************************** 1999 ************************************************
********************************************************************************************;
use $PSID\f99, clear
ren ER13002 id

** Demographic**
ren ER13009 fsize
ren ER13010 age
ren ER13011 sex
ren ER13012 agew
ren ER13013 kids
ren ER13021 marit
ren ER16423 marit_generated
ren ER15928 race1
ren ER15929 race2
ren ER15930 race3
ren ER15931 race4	 
ren ER15836 wrace1
ren ER15837 wrace2
ren ER15838 wrace3
ren ER15839 wrace4
ren ER14976 outkid
ren ER16432 smsa 
ren ER13041 house
ren ER16518 weight 
ren ER16519 weight_CS

/* Home insurance */
ren ER13043 homeinsure
replace homeinsure=. if homeinsure>9997

/* Asset Stock */ 
ren ER15020 cash
replace cash=. if cash>=999999998

ren ER15026 bonds
replace bonds=. if bonds>=999999998

ren ER15007 stocks
replace stocks=. if stocks>=999999998

ren ER14993 real_estate
replace real_estate=. if real_estate>=999999998

ren ER14997 carval
replace carval=. if carval>=999999998

ren ER15002 busval
replace busval=. if busval>=999999998

ren ER15014 penval
replace penval=. if penval>=999999998

ren ER13047 mortgage1
replace mortgage1=. if mortgage1>=9999998

ren ER13056 mortgage2
replace mortgage2=. if mortgage2>=9999998

ren ER15031 other_debt
replace other_debt=. if other_debt>=999999998

/* health expenditures */ 
ren ER15780 hinsurance
ren ER15781 nurse
ren ER15787 doctor
ren ER15793 prescription
ren ER15799 totalhealth

replace hinsurance=. if hinsurance>999997
replace nurse=. if nurse>999997
replace doctor=. if doctor>9999997
replace prescription=. if prescription>9999997
replace totalhealth=. if totalhealth>9999997

replace hinsurance=hinsurance/2 	/* all medical care variables are reported for a period of two years */ 
replace nurse=nurse/2
replace doctor=doctor/2
replace prescription=prescription/2
replace totalhealth=totalhealth/2

/* Utilites */
ren ER13086 electric
replace electric=. if electric>9997
replace electric=. if (ER13087<3 | ER13087>6 | ER13087==4) & electric!=0
replace electric=52*electric if ER13087==3	/* 	note that 3 (weekly) rarely appears but kept in all 
												categories not to have to many special cases */ 
replace electric=12*electric if ER13087==5

ren ER13088 heating
replace heating=. if heating>9997
replace heating=. if ( ER13089<3 | ER13089>6 | ER13089==4) & heating!=0 
replace heating=52*heating if ER13089==3
replace heating=12*heating if ER13089==5

ren ER13090 water
replace water=. if water>9997
replace water=. if (ER13091<3 | ER13091>6 | ER13091==4) & water!=0 
replace water=52*water if ER13091==3
replace water=12*water if ER13091==5

ren ER13097 miscutils
replace miscutils=. if miscutils>997
replace miscutils=. if (ER13098<3 & ER13098>0 | ER13098>6 | ER13098==4) & miscutils!=0 
replace miscutils=52*miscutils if ER13098==3
replace miscutils=12*miscutils if ER13098==5

/* Car expenditure */
ren ER13191 carins
replace carins=. if carins>999997
replace carins=. if (ER13192<5 | ER13192>6) & carins!=0
replace carins=12*carins if ER13192==5

ren ER13195 carrepair
replace carrepair=. if carrepair>99997
replace carrepair=12*carrepair

ren ER13196 gasoline
replace gasoline=. if gasoline>99997
replace gasoline=12*gasoline

ren ER13197 parking
replace parking=. if parking>99997
replace parking=12*parking

ren ER13198 busfare
replace busfare=. if busfare>99997
replace busfare=12*busfare

ren ER13199 taxifare
replace taxifare=. if taxifare>99997
replace taxifare=12*taxifare

ren ER13200 othertrans
replace othertrans=. if othertrans>99997
replace othertrans=12*othertrans

/* Education expenditures */ 

ren ER13202 tuition
replace tuition=. if tuition>999997

ren ER13204 otherschool
replace otherschool=. if otherschool>999997

/* Child Care */ 
ren ER14232 childcare
replace childcare=. if childcare>999997

/* Merge mentions for empst for head */ 
ren ER13205 empst1
ren ER13206 empst2
ren ER13207 empst3

forvalues i=1(1)3 {
gen empst`i'_ordered =. 		
replace empst`i'_ordered = 11 if empst`i'==2
replace empst`i'_ordered = 12 if empst`i'==1
replace empst`i'_ordered = 13 if empst`i'==3
replace empst`i'_ordered = 14 if empst`i'==4
replace empst`i'_ordered = 15 if empst`i'==5  
replace empst`i'_ordered = 16 if empst`i'==7
replace empst`i'_ordered = 17 if empst`i'==6
replace empst`i'_ordered = 18 if empst`i'==8
replace empst`i'_ordered = 19 if empst`i'==9
}

gen empst_ordered=min(empst1_ordered, empst2_ordered, empst3_ordered)

gen empst = . 			
replace empst = 2 if empst_ordered==11
replace empst = 1 if empst_ordered==12
replace empst = 3 if empst_ordered==13
replace empst = 4 if empst_ordered==14
replace empst = 5 if empst_ordered==15
replace empst = 7 if empst_ordered==16
replace empst = 6 if empst_ordered==17
replace empst = 8 if empst_ordered==18
replace empst = 9 if empst_ordered==19

drop empst*_ordered

/* Merge mentions for empst for wife. Note - I construct the variable such that "no wife" will have a "." and not a "0" in this variable */ 
ren ER13717 wempst1
ren ER13718 wempst2
ren ER13719 wempst3

forvalues i=1(1)3 {
gen wempst`i'_ordered =. 		
replace wempst`i'_ordered = 11 if wempst`i'==2
replace wempst`i'_ordered = 12 if wempst`i'==1
replace wempst`i'_ordered = 13 if wempst`i'==3
replace wempst`i'_ordered = 14 if wempst`i'==4
replace wempst`i'_ordered = 15 if wempst`i'==5  
replace wempst`i'_ordered = 16 if wempst`i'==7
replace wempst`i'_ordered = 17 if wempst`i'==6
replace wempst`i'_ordered = 18 if wempst`i'==8
replace wempst`i'_ordered = 19 if wempst`i'==9
}

gen wempst_ordered=min(wempst1_ordered, wempst2_ordered, wempst3_ordered)

gen wempst = . 			
replace wempst = 2 if wempst_ordered==11
replace wempst = 1 if wempst_ordered==12
replace wempst = 3 if wempst_ordered==13
replace wempst = 4 if wempst_ordered==14
replace wempst = 5 if wempst_ordered==15
replace wempst = 7 if wempst_ordered==16
replace wempst = 6 if wempst_ordered==17
replace wempst = 8 if wempst_ordered==18
replace wempst = 9 if wempst_ordered==19

drop wempst*_ordered

** Education ** 
ren ER16516 educ
replace educ=. if educ==99
ren ER16517 weduc 
replace weduc=. if weduc==99 | agew==0 /* note that weduc==0 can be zero grades but is most likely no wife the no wife case is addressed by agew==0 */ 

/* Food Related variables */ 
gen       fstmp=0
cap program drop doit
program def doit
        local i=14258
        while `i' < 14270 {
            replace ER`i'=. if ER`i'>1
        	local i=`i'+1
       }
end
qui doit
egen nm=rsum(ER14258-ER14269)
replace fstmp=ER14256              if ER14257==6 
replace fstmp=ER14256*nm           if ER14257==5 
replace fstmp=ER14256*((26/12)*nm) if ER14257==4  
replace fstmp=ER14256*((52/12)*nm) if ER14257==3
replace fstmp=.					 if ER14256>999997 | ER14257>6 
drop nm

replace ER14288=.  if ER14288 >= 99998     
replace ER14291=.  if ER14291 >= 99998 
replace ER14293=.  if ER14293 >= 99998 
replace ER14295=.  if ER14295 >= 99998 
replace ER14298=.  if ER14298 >= 99998 
replace ER14300=.  if ER14300 >= 99998 

gen fdhm1     	= ER14288
gen fdhm2		= ER14295        
gen fdhmper1	= ER14289
gen fdhmper2	= ER14296        
gen fddel1    	= ER14291
gen fddel2		= ER14298        
gen fddelper1 	= ER14292 
gen fddelper2	= ER14299   
gen fout1    	= ER14293
gen fout2      = ER14300	  
gen foutper1 	= ER14294
gen foutper2   = ER14301     

gen fdhm        = 	fdhm1    	if fdhm1 !=. & fdhm2 == 0	
gen fdhmper     = 	fdhmper1 	if fdhm1 !=. & fdhm2 == 0	
replace fdhm    = 	fdhm2    	if fdhm==.
replace fdhmper = 	fdhmper2 	if fdhmper==.

gen fddel 		= 	fddel1 		if fddel1!=. & fddel2 == 0		
gen fddelper 	= 	fddelper1 	if fddel1!=. & fddel2 == 0
replace fddel 	= 	fddel2 		if fddel==.
replace fddelper = 	fddelper2 	if fddelper==.

gen fout		= 	fout1 		if fout1!=. & fout2 == 0
gen foutper 	= 	foutper1 	if fout1!=. & fout2 == 0 
replace fout 	= 	fout2 		if fout==.
replace foutper = 	foutper2 	if foutper==. 

replace fdhm=. 					if fdhmper == 2 | fdhmper>= 7  /* Missing if Wild Code (2), Other,DK,RF (7,8,9)*/
replace fdhm=fdhm*52 			if fdhmper == 3
replace fdhm=fdhm*26 			if fdhmper == 4
replace fdhm=fdhm*12 			if fdhmper == 5

replace fddel=. 				if fddelper == 2 | fddelper>= 7
replace fddel=fddel*52 			if fddelper == 3
replace fddel=fddel*26 			if fddelper == 4
replace fddel=fddel*12 			if fddelper == 5

replace fout=.  				if foutper >= 7
replace fout=fout*365   		if foutper == 2  		
replace fout=fout*52 			if foutper == 3
replace fout=fout*26 			if foutper == 4
replace fout=fout*12 			if foutper == 5

egen food=rsum(fdhm fddel)
replace food=. if fdhm==. & fddel==.

/* Asset Income variables */
gen gardeninc=.
cap program drop doit
program def doit
      local i=14447
        while `i' < 14459 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER14447-ER14458)
replace gardeninc=ER14445    	if ER14446==6
replace gardeninc=ER14445*nm  	if ER14446==5
replace gardeninc=.   			if ER14445>9999997
drop nm


gen roomerinc=.
cap program drop doit
program def doit
      local i=14463
        while `i' < 14475 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER14463-ER14474)
replace roomerinc=ER14461               if ER14462==6
replace roomerinc=ER14461*nm            if ER14462==5
replace roomerinc=ER14461*((26/12)*nm)  if ER14462==4
replace roomerinc=ER14461*((52/12)*nm)  if ER14462==3
replace roomerinc=. 					if ER14461>9999997
drop nm

gen rentinc=.
cap program drop doit
program def doit
      local i=14481
        while `i' < 14493 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER14481-ER14492)
replace rentinc=ER14479                 if ER14480==6
replace rentinc=ER14479*nm              if ER14480==5
replace rentinc=ER14479*((26/12)*nm)    if ER14480==4
replace rentinc=ER14479*((52/12)*nm)    if ER14480==3
replace rentinc=.          				if ER14479>999997
drop nm

gen divinc=.
cap program drop doit
program def doit
      local i= 14496
        while `i' < 14508 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER14496-ER14507)
replace divinc=ER14494          		if ER14495==6
replace divinc=ER14494*nm       		if ER14495==5
replace divinc=. 						if ER14494>999997
drop nm

gen intinc=.
cap program drop doit
program def doit
      local i=14511
        while `i' < 14523 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER14511-ER14522)
replace intinc=ER14509      			if ER14510==6
replace intinc=ER14509*nm   			if ER14510==5
replace intinc=. 						if ER14509>999997
drop nm

gen trustinc=.
cap program drop doit
program def doit
      local i=14526
        while `i' < 14538 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER14526-ER14537)
replace trustinc=ER14524        		if ER14525==6
replace trustinc=ER14524*nm     		if ER14525==5
replace trustinc=. 						if ER14524>999997
drop nm

gen aliminc=.
cap program drop doit
program def doit
      local i=14696
        while `i' < 14708 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER14696-ER14707)
replace aliminc=ER14694                 if ER14695==6
replace aliminc=ER14694*nm              if ER14695==5
replace aliminc=ER14694*((26/12)*nm)    if ER14695==4
replace aliminc=ER14694*((52/12)*nm)    if ER14695==3
replace aliminc=. 	    		    if ER14694>99997
drop nm

gen wdivinc=.
cap program drop doit
program def doit
      local i= 14792
        while `i' < 14804 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER14792-ER14803)
replace wdivinc=ER14790         if ER14791==6
replace wdivinc=ER14790*nm      if ER14791==5
replace wdivinc=. 				if ER14790>999997
drop nm

gen wintinc=.
cap program drop doit
program def doit
      local i=14807
        while `i' < 14819 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER14807-ER14818)
replace wintinc=ER14805         if ER14806==6
replace wintinc=ER14805*nm      if ER14806==5
replace wintinc=. 				if ER14805>999997
drop nm

gen wtrustinc=.
cap program drop doit
program def doit
      local i=14822
        while `i' < 14834 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER14822-ER14833)
replace wtrustinc=ER14820       if ER14821==6
replace wtrustinc=ER14820*nm    if ER14821==5
replace wtrustinc=. 			if ER14820>999997
drop nm

gen wothinc=.
cap program drop doit
program def doit
      local i=14837
        while `i' < 14849 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER14837-ER14848)
replace wothinc=ER14835         if ER14836==6
replace wothinc=ER14835*nm      if ER14836==5
replace wothinc=. 				if ER14835>999997
drop nm

egen asset_part1=rsum(gardeninc roomerinc rentinc divinc intinc trustinc aliminc wdivinc wintinc wtrustinc wothinc)

*** Rent ***
gen rent=.
replace rent=ER13065        	if ER13066==6
replace rent=ER13065*12     	if ER13066==5
replace rent=ER13065*26   	    if ER13066==4
replace rent=ER13065*52     	if ER13066==3
replace rent=. 					if ER13065>99997

*** generate labor and asset part of farm income
gen farmlabor=0
replace farmlabor=0.5*ER14348 	if ER14348>0
replace farmlabor=. 			if ER14348>9999997

gen farmasset=0
replace farmasset=0.5*ER14348 	if ER14348>0
replace farmasset=ER14348 		if ER14348<0
replace farmasset=. 			if ER14348>9999997 | ER14348==-5000 /* -5000 is a wild code */ 

ren ER16471 hours
ren ER16482 hourw

ren ER13004 state

#delimit;
keep id hours hourw state food fout fstmp rent educ weduc fsize age sex agew kids marit race* wrace* house  
outkid empst wempst asset_part1 farmlabor farmasset smsa weight* divinc wdivinc 
gardeninc roomerinc rentinc intinc trustinc aliminc wintinc wtrustinc wothinc 
electric heating water miscutils homeinsure hinsurance nurse doctor prescription totalhealth 
carins carrepair gasoline parking busfare taxifare othertrans 
tuition otherschool childcare cash* bonds* stocks* busval* penval* other_debt* real_estate* carval* mortgage1* mortgage2*;
#delimit cr

sort id
save "$output\newfam98", replace

*****************************************************************************************************************;
******************************* Append 1999 Income variables from File*************************************************
*****************************************************************************************************************;
use $PSID\faminc99, clear

ren ID99 id

keep id FAMINC99 HDEARN99 HDBUSY99 WFEARN99 WFBUSY99 TXHW99 TXOFM99 TRHW99 TROFM99 HDBUSK99 WFBUSK99

sort id
merge id using "$output\newfam98.dta"
tab _merge
keep if _merge==3
drop _merge

* Income *
gen truncy=FAMINC99<1
ren FAMINC99 y
replace y=1 if y<1  
egen ly=rsum(HDEARN99 HDBUSY99 farmlabor)
gen wages=HDEARN99
egen wly=rsum(WFEARN99 WFBUSY99)
gen wagesw=WFEARN99
ren TXOFM99 tyoth

egen asset =rsum(farmasset HDBUSK99 WFBUSK99 asset_part1)
gen trunca=0

#delimit;
keep id food fout fstmp rent educ fsize age sex weduc agew kids marit race* wrace* house 
outkid empst wempst asset_part1 hours hourw weight* state y 
ly wages wagesw wly tyoth asset smsa truncy trunca 
electric heating water miscutils homeinsure hinsurance nurse doctor prescription totalhealth 
carins carrepair gasoline parking busfare taxifare othertrans tuition otherschool childcare 
cash* bonds* stocks* busval* penval* other_debt* real_estate* carval* mortgage1* mortgage2*; 
#delimit cr

gen year=98
save "$output\fam98", replace
erase "$output\newfam98.dta"


****************************************************************************************************
**************************************2001******************************************************
********************************************************************************************;

use $PSID\f01, clear

ren ER17002 id
ren ER17012 fsize
ren ER17013 age
ren ER17014 sex
ren ER17015 agew
ren ER17016 kids
ren ER17024 marit
ren ER19989 race1
ren ER19990 race2
ren ER19991 race3
ren ER19992 race4
ren ER19897 wrace1
ren ER19898 wrace2
ren ER19899 wrace3
ren ER19900 wrace4
ren ER19172 outkid
ren ER17044 house
ren ER20378 smsa
ren ER20394 weight 
ren ER17004 state

ren ER17007 fchg

/* Home insurance */
ren ER17048 homeinsure
replace homeinsure=. if homeinsure>9997

/* Asset Stock */ 
ren ER19216 cash
replace cash=. if cash>=999999998

ren ER19222 bonds
replace bonds=. if bonds>=999999998

ren ER19203 stocks
replace stocks=. if stocks>=999999998

ren ER19189 real_estate
replace real_estate=. if real_estate>=999999998

ren ER19193 carval
replace carval=. if carval>=999999998

ren ER19198 busval
replace busval=. if busval>=999999998

ren ER19210 penval
replace penval=. if penval>=999999998

ren ER17052 mortgage1
replace mortgage1=. if mortgage1>=9999998

ren ER17063 mortgage2
replace mortgage2=. if mortgage2>=9999998

ren ER19227 other_debt
replace other_debt=. if other_debt>=999999998

/* health expenditures */ 
ren ER19841 hinsurance
ren ER19842 nurse
ren ER19848 doctor
ren ER19854 prescription
ren ER19860 totalhealth

replace hinsurance=. if hinsurance>999997
replace nurse=. if nurse>999997
replace doctor=. if doctor>9999997
replace prescription=. if prescription>9999997
replace totalhealth=. if totalhealth>99999997

replace hinsurance=hinsurance/2 	
replace nurse=nurse/2
replace doctor=doctor/2
replace prescription=prescription/2
replace totalhealth=totalhealth/2

/* Utilites */
ren ER17097 electric
replace electric=. if electric>9997 
replace electric=. if (ER17098<3 | ER17098>6 | ER17098==4) & electric!=0
replace electric=52*electric if ER17098==3
replace electric=12*electric if ER17098==5

ren ER17099 heating
replace heating=. if heating>9997
replace heating=. if (ER17100<3 | ER17100>6 | ER17100==4) & heating!=0 
replace heating=52*heating if ER17100==3
replace heating=12*heating if ER17100==5

ren ER17101 water
replace water=. if water>9997
replace water=. if (ER17102<3 | ER17102>6 | ER17102==4) & water!=0
replace water=52*water if ER17102==3
replace water=12*water if ER17102==5

ren ER17108 miscutils
replace miscutils=. if miscutils>997
replace miscutils=. if (ER17109<3 & ER17109>0| ER17109>6 | ER17109==4) & miscutils!=0
replace miscutils=52*miscutils if ER17109==3
replace miscutils=12*miscutils if ER17109==5

/* Car expenditure */
ren ER17202 carins
replace carins=. if carins>999997
replace carins=. if (ER17203<5 | ER17203>6) & carins!=0
replace carins=12*carins if ER17203==5

ren ER17206 carrepair
replace carrepair=. if carrepair>99997
replace carrepair=12*carrepair

ren ER17207 gasoline
replace gasoline=. if gasoline>99997
replace gasoline=12*gasoline

ren ER17208 parking
replace parking=. if parking>99997
replace parking=12*parking

ren ER17209 busfare
replace busfare=. if busfare>99997
replace busfare=12*busfare

ren ER17210 taxifare
replace taxifare=. if taxifare>99997
replace taxifare=12*taxifare

ren ER17211 othertrans
replace othertrans=. if othertrans>99997
replace othertrans=12*othertrans

/* Education expenditures */ 
ren ER17213 tuition
replace tuition=. if tuition>999997

ren ER17215 otherschool
replace otherschool=. if otherschool>999997

/* Child Care */ 
ren ER18362 childcare
replace childcare=. if childcare>999997

/* Merge mentions for empst for head */ 
ren ER17216 empst1
ren ER17217 empst2
ren ER17218 empst3

forvalues i=1(1)3 {
gen empst`i'_ordered =. 		
replace empst`i'_ordered = 11 if empst`i'==2
replace empst`i'_ordered = 12 if empst`i'==1
replace empst`i'_ordered = 13 if empst`i'==3
replace empst`i'_ordered = 14 if empst`i'==4
replace empst`i'_ordered = 15 if empst`i'==5  
replace empst`i'_ordered = 16 if empst`i'==7
replace empst`i'_ordered = 17 if empst`i'==6
replace empst`i'_ordered = 18 if empst`i'==8
replace empst`i'_ordered = 19 if empst`i'==9
}

gen empst_ordered=min(empst1_ordered, empst2_ordered, empst3_ordered)

gen empst = . 			
replace empst = 2 if empst_ordered==11
replace empst = 1 if empst_ordered==12
replace empst = 3 if empst_ordered==13
replace empst = 4 if empst_ordered==14
replace empst = 5 if empst_ordered==15
replace empst = 7 if empst_ordered==16
replace empst = 6 if empst_ordered==17
replace empst = 8 if empst_ordered==18
replace empst = 9 if empst_ordered==19

ren ER17786 wempst1
ren ER17787 wempst2
ren ER17788 wempst3

forvalues i=1(1)3 {
gen wempst`i'_ordered =. 		
replace wempst`i'_ordered = 11 if wempst`i'==2
replace wempst`i'_ordered = 12 if wempst`i'==1
replace wempst`i'_ordered = 13 if wempst`i'==3
replace wempst`i'_ordered = 14 if wempst`i'==4
replace wempst`i'_ordered = 15 if wempst`i'==5  
replace wempst`i'_ordered = 16 if wempst`i'==7
replace wempst`i'_ordered = 17 if wempst`i'==6
replace wempst`i'_ordered = 18 if wempst`i'==8
replace wempst`i'_ordered = 19 if wempst`i'==9
}

gen wempst_ordered=min(wempst1_ordered, wempst2_ordered, wempst3_ordered)

gen wempst = . 			
replace wempst = 2 if wempst_ordered==11
replace wempst = 1 if wempst_ordered==12
replace wempst = 3 if wempst_ordered==13
replace wempst = 4 if wempst_ordered==14
replace wempst = 5 if wempst_ordered==15
replace wempst = 7 if wempst_ordered==16
replace wempst = 6 if wempst_ordered==17
replace wempst = 8 if wempst_ordered==18
replace wempst = 9 if wempst_ordered==19

** Education ** 
ren ER20457 educ
replace educ=. if educ==99
ren ER20458 weduc
replace weduc=. if weduc==99 | agew==0

/* Food Related variables */ 
gen fstmp=0
cap program drop doit
program def doit
        local i=18390
        while `i' < 18402 {
            replace ER`i'=. if ER`i'>1
        	local i=`i'+1
       }
end
qui doit
egen nm=rsum(ER18390-ER18401)
replace fstmp=ER18387              if ER18388==6 
replace fstmp=ER18387*nm           if ER18388==5 
replace fstmp=ER18387*((26/12)*nm) if ER18388==4  
replace fstmp=ER18387*((52/12)*nm) if ER18388==3
replace fstmp=.					  if ER18387>999997 | ER18388>6 
drop nm

replace ER18421=.  if ER18421 >= 99998    
replace ER18425=.  if ER18425 >= 99998 
replace ER18428=.  if ER18428 >= 99998 
replace ER18431=.  if ER18431 >= 99998 
replace ER18435=.  if ER18435 >= 99998  
replace ER18438=.  if ER18438 >= 99998 

gen fdhm1     	= ER18421
gen fdhm2		= ER18431        
gen fdhmper1	= ER18422
gen fdhmper2	= ER18432        
gen fddel1    	= ER18425
gen fddel2		= ER18435        
gen fddelper1 	= ER18426 
gen fddelper2	= ER18436   
gen fout1    	= ER18428
gen fout2      	= ER18438	  
gen foutper1 	= ER18429
gen foutper2   	= ER18439  

gen fdhm        = 	fdhm1    	if fdhm1 !=. & fdhm2 == 0	
gen fdhmper     = 	fdhmper1 	if fdhm1 !=. & fdhm2 == 0	
replace fdhm    = 	fdhm2    	if fdhm==.
replace fdhmper = 	fdhmper2 	if fdhmper==.

gen fddel 		= 	fddel1 		if fddel1!=. & fddel2 == 0		
gen fddelper 	= 	fddelper1 	if fddel1!=. & fddel2 == 0
replace fddel 	= 	fddel2 		if fddel==.
replace fddelper = 	fddelper2 	if fddelper==.

gen fout 		= 	fout1 		if fout1!=. & fout2 == 0
gen foutper 	= 	foutper1 	if fout1!=. & fout2 == 0 
replace fout 	= 	fout2 		if fout==.
replace foutper = 	foutper2 	if foutper==. 

replace fdhm=. 					if fdhmper == 2 | fdhmper>= 7
replace fdhm=fdhm*52 			if fdhmper == 3
replace fdhm=fdhm*26 			if fdhmper == 4
replace fdhm=fdhm*12 			if fdhmper == 5

replace fddel=. 				if fddelper == 2 | fddelper>= 7
replace fddel=fddel*52 			if fddelper == 3
replace fddel=fddel*26 			if fddelper == 4
replace fddel=fddel*12 			if fddelper == 5

replace fout=.  				if foutper >= 7
replace fout=fout*365   		if foutper == 2
replace fout=fout*52 			if foutper == 3
replace fout=fout*26 			if foutper == 4
replace fout=fout*12 			if foutper == 5

egen food=rsum(fdhm fddel)
replace food=. if fdhm==. & fddel==.

/* Asset Income variables */
gen gardeninc=.
cap program drop doit
program def doit
      local i=18600
        while `i' < 18612 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER18600-ER18611)
replace gardeninc=ER18597 			if ER18598==6
replace gardeninc=ER18597*nm   		if ER18598==5
replace gardeninc=. 				if ER18597>9999997
drop nm

gen roomerinc=.
cap program drop doit
program def doit
      local i=18617
        while `i' < 18629 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER18617-ER18628)
replace roomerinc=ER18614               if ER18615==6
replace roomerinc=ER18614*nm            if ER18615==5
replace roomerinc=ER18614*((26/12)*nm)  if ER18615==4
replace roomerinc=ER18614*((52/12)*nm)  if ER18615==3
replace roomerinc=. 					if ER18614>9999997
drop nm

gen rentinc=.
cap program drop doit
program def doit
      local i=18637
        while `i' < 18649 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER18637-ER18648)
replace rentinc=ER18634                 if ER18635==6
replace rentinc=ER18634*nm              if ER18635==5
replace rentinc=ER18634*((26/12)*nm)    if ER18635==4
replace rentinc=ER18634*((52/12)*nm)    if ER18635==3
replace rentinc=.          				if ER18634>999997
drop nm

gen divinc=.
cap program drop doit
program def doit
      local i=18653
        while `i' < 18665 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER18653-ER18664)
replace divinc=ER18650          if ER18651==6
replace divinc=ER18650*nm       if ER18651==5
replace divinc=. 				if ER18650>999997
drop nm

gen intinc=.
cap program drop doit
program def doit
      local i=18669
        while `i' < 18681 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER18669-ER18680)
replace intinc=ER18666      	if ER18667==6
replace intinc=ER18666*nm   	if ER18667==5
replace intinc=. 				if ER18666>999997
drop nm

gen trustinc=.
cap program drop doit
program def doit
      local i=18685
        while `i' < 18697 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER18685-ER18696)
replace trustinc=ER18682        if ER18683==6
replace trustinc=ER18682*nm     if ER18683==5
replace trustinc=. 				if ER18682>999997
drop nm

gen aliminc=.
cap program drop doit
program def doit
      local i=18866
        while `i' < 18878 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER18866-ER18877)
replace aliminc=ER18863                 if ER18864==6
replace aliminc=ER18863*nm              if ER18864==5
replace aliminc=ER18863*((26/12)*nm)    if ER18864==4
replace aliminc=ER18863*((52/12)*nm)    if ER18864==3
replace aliminc=. 						if ER18863>99997
drop nm

gen wdivinc=.
cap program drop doit
program def doit
      local i=18969
        while `i' < 18981 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER18969-ER18980)
replace wdivinc=ER18966         if ER18967==6
replace wdivinc=ER18966*nm      if ER18967==5
replace wdivinc=. 				if ER18966>999997
drop nm

gen wintinc=.
cap program drop doit
program def doit
      local i=18985
        while `i' < 18997 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER18985-ER18996)
replace wintinc=ER18982         if ER18983==6
replace wintinc=ER18982*nm      if ER18983==5
replace wintinc=. 				if ER18982>999997
drop nm

gen wtrustinc=.
cap program drop doit
program def doit
      local i=19001
        while `i' < 19013{
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER19001-ER19012)
replace wtrustinc=ER18998       if ER18999==6
replace wtrustinc=ER18998*nm    if ER18999==5
replace wtrustinc=. 			if ER18998>999997
drop nm

gen wothinc=.
cap program drop doit
program def doit
      local i=19017
        while `i' < 19029 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER19017-ER19028)
replace wothinc=ER19014         if ER19015==6
replace wothinc=ER19014*nm      if ER19015==5
replace wothinc=. 				if ER19014>999997
drop nm

egen asset_part1=rsum(gardeninc roomerinc rentinc divinc intinc trustinc aliminc wdivinc wintinc wtrustinc wothinc)
*** Rent ***
gen rent=.
replace rent=ER17074        	if ER17075==6
replace rent=ER17074*12     	if ER17075==5
replace rent=ER17074*26  	if ER17075==4
replace rent=ER17074*52     	if ER17075==3
replace rent=ER17074*365  	if ER17075==2
replace rent=. 					if ER17074>99997

*** Income variables ***
*generate labor and asset part of farm income*
gen farmlabor=0
replace farmlabor=0.5*ER18487 	if ER18487>0
replace farmlabor=. 			if ER18487>9999997

gen farmasset=0
replace farmasset=0.5*ER18487 	if ER18487>0
replace farmasset=ER18487 		if ER18487<0
replace farmasset=. 			if ER18487>9999997

/* Income and hours variables */
ren ER20399 hours
ren ER20410 hourw

** Income **
egen ly= 				rsum(ER20443 farmlabor ER20422) 
egen wly=				rsum(ER20444 ER20447)
gen wages = ER20443
gen wagesw = ER20447
ren ER20453 tyoth
gen truncy=ER20456<1
ren ER20456 y
replace y=1 if y<1

egen asset= rsum(farmasset ER20423 ER20445 asset_part1)
gen trunca=0

#delimit;
keep id food fout fstmp rent educ fsize age sex weduc agew kids marit race* wrace* house 
outkid empst wempst asset_part1 hours hourw weight* state y 
ly wages wagesw wly tyoth asset smsa truncy trunca 
electric heating water miscutils homeinsure hinsurance nurse doctor prescription totalhealth 
carins carrepair gasoline parking busfare taxifare othertrans tuition otherschool childcare 
cash* bonds* stocks* busval* penval* other_debt* real_estate* carval* mortgage1* mortgage2* fchg; 
#delimit cr

gen year=100

save "$output\fam100", replace

*****************************************************************************************************************;
****************************************** 2003 **************************************************
*****************************************************************************************************************;
use $PSID\f03, clear

ren ER21002 id

ren ER21016 fsize
ren ER21017 age
ren ER21018 sex
ren ER21019 agew
ren ER21020 kids
ren ER21023 marit
ren ER23426 race1
ren ER23427 race2
ren ER23428 race3
ren ER23429 race4
ren ER23334 wrace1 
ren ER23335 wrace2
ren ER23336 wrace3
ren ER23337 wrace4 
ren ER22537 outkid
ren ER21043 house
ren ER24145 smsa
ren ER21007 fchg
ren ER21003 state
ren ER24179 weight
ren ER24180 weight_CS 

/* Home insurance */
ren ER21047 homeinsure
replace homeinsure=. if homeinsure>9997

/* Asset Stock */ 
ren ER22596 cash
replace cash=. if cash>=999999998

ren ER22617 bonds
replace bonds=. if bonds>=999999998

ren ER22568 stocks
replace stocks=. if stocks>=999999998

ren ER22554 real_estate
replace real_estate=. if real_estate>=999999998

ren ER22558 carval
replace carval=. if carval>=999999998

ren ER22563 busval
replace busval=. if busval>=999999998

ren ER22590 penval
replace penval=. if penval>=999999998

ren ER21051 mortgage1
replace mortgage1=. if mortgage1>=9999998

ren ER21062 mortgage2
replace mortgage2=. if mortgage2>=9999998

ren ER22622 other_debt
replace other_debt=. if other_debt>=999999998

/* health expenditures */ 
ren ER23278 hinsurance
ren ER23279 nurse
ren ER23285 doctor
ren ER23291 prescription
ren ER23297 totalhealth

replace hinsurance=. if hinsurance>999997
replace nurse=. if nurse>999997
replace doctor=. if doctor>9999997
replace prescription=. if prescription>9999997
replace totalhealth=. if totalhealth>99999997

replace hinsurance=hinsurance/2 	 
replace nurse=nurse/2
replace doctor=doctor/2
replace prescription=prescription/2
replace totalhealth=totalhealth/2

/* Utilites */
ren ER21086 electric
replace electric=. if electric>9997 
replace electric=. if (ER21087<3 | ER21087>6 | ER21087==4) & electric!=0
replace electric=52*electric if ER21087==3
replace electric=12*electric if ER21087==5

ren ER21088 heating
replace heating=. if heating>9997
replace heating=. if (ER21089<3 | ER21089>6 | ER21089==4) & heating!=0
replace heating=52*heating if ER21089==3
replace heating=12*heating if ER21089==5

ren ER21090 water
replace water=. if water>9997
replace water=. if (ER21091<3 | ER21091>6 | ER21091==4) & water!=0
replace water=52*water if ER21091==3
replace water=12*water if ER21091==5

ren ER21097 miscutils
replace miscutils=. if miscutils>997
replace miscutils=. if (ER21098<3 & ER21098>0 | ER21098>6 | ER21098==4) & miscutils!=0
replace miscutils=52*miscutils if ER21098==3
replace miscutils=12*miscutils if ER21098==5

/* Car expenditure */
ren ER21838 carins
replace carins=. if carins>999997
replace carins=. if (ER21839<5 | ER21839>6) & carins!=0
replace carins=12*carins if ER21839==5

ren ER21842 carrepair
*gen carrepair_tc = (carrepair==99997)
replace carrepair=. if carrepair>99997
replace carrepair=12*carrepair

ren ER21843 gasoline
replace gasoline=. if gasoline>99997
replace gasoline=12*gasoline

ren ER21844 parking
replace parking=. if parking>99997
replace parking=12*parking

ren ER21845 busfare
replace busfare=. if busfare>99997
replace busfare=12*busfare

ren ER21846 taxifare
replace taxifare=. if taxifare>99997
replace taxifare=12*taxifare

ren ER21847 othertrans
replace othertrans=. if othertrans>99997
replace othertrans=12*othertrans

/* Education expenditures */ 

ren ER21849 tuition
replace tuition=. if tuition>999997

ren ER21851 otherschool
replace otherschool=. if otherschool>999997

/* Child Care */ 
ren ER21628 childcare
replace childcare=. if childcare>999997

/* Merge mentions for empst for head */ 
ren ER21123 empst1
ren ER21124 empst2
ren ER21125 empst3

forvalues i=1(1)3 {
gen empst`i'_ordered =. 		
replace empst`i'_ordered = 11 if empst`i'==2
replace empst`i'_ordered = 12 if empst`i'==1
replace empst`i'_ordered = 13 if empst`i'==3
replace empst`i'_ordered = 14 if empst`i'==4
replace empst`i'_ordered = 15 if empst`i'==5  
replace empst`i'_ordered = 16 if empst`i'==7
replace empst`i'_ordered = 17 if empst`i'==6
replace empst`i'_ordered = 18 if empst`i'==8
replace empst`i'_ordered = 19 if empst`i'==9
}

gen empst_ordered=min(empst1_ordered, empst2_ordered, empst3_ordered)

gen empst = . 			
replace empst = 2 if empst_ordered==11
replace empst = 1 if empst_ordered==12
replace empst = 3 if empst_ordered==13
replace empst = 4 if empst_ordered==14
replace empst = 5 if empst_ordered==15
replace empst = 7 if empst_ordered==16
replace empst = 6 if empst_ordered==17
replace empst = 8 if empst_ordered==18
replace empst = 9 if empst_ordered==19

ren ER21373 wempst1
ren ER21374 wempst2
ren ER21375 wempst3

forvalues i=1(1)3 {
gen wempst`i'_ordered =. 		
replace wempst`i'_ordered = 11 if wempst`i'==2
replace wempst`i'_ordered = 12 if wempst`i'==1
replace wempst`i'_ordered = 13 if wempst`i'==3
replace wempst`i'_ordered = 14 if wempst`i'==4
replace wempst`i'_ordered = 15 if wempst`i'==5  
replace wempst`i'_ordered = 16 if wempst`i'==7
replace wempst`i'_ordered = 17 if wempst`i'==6
replace wempst`i'_ordered = 18 if wempst`i'==8
replace wempst`i'_ordered = 19 if wempst`i'==9
}

gen wempst_ordered=min(wempst1_ordered, wempst2_ordered, wempst3_ordered)

gen wempst = . 			
replace wempst = 2 if wempst_ordered==11
replace wempst = 1 if wempst_ordered==12
replace wempst = 3 if wempst_ordered==13
replace wempst = 4 if wempst_ordered==14
replace wempst = 5 if wempst_ordered==15
replace wempst = 7 if wempst_ordered==16
replace wempst = 6 if wempst_ordered==17
replace wempst = 8 if wempst_ordered==18
replace wempst = 9 if wempst_ordered==19

** Education ** 
ren ER24148 educ
replace educ=. if educ==99
ren ER24149 weduc
replace weduc=. if weduc==99 | agew==0

/* Food Related variables */ 
gen       fstmp=0
cap program drop doit
program def doit
        local i=21656
        while `i' < 21668 {
            replace ER`i'=. if ER`i'>1
        	local i=`i'+1
       }
end
qui doit
egen nm=rsum(ER21656-ER21667)
replace fstmp=ER21653              if ER21654==6 
replace fstmp=ER21653*nm           if ER21654==5 
replace fstmp=ER21653*((26/12)*nm) if ER21654==4  
replace fstmp=ER21653*((52/12)*nm) if ER21654==3
replace fstmp=.					 if ER21653>999997  | ER21654>6 
drop nm

replace ER21686=. if ER21686 >= 99998        
replace ER21690=. if ER21690 >= 99998    
replace ER21693=. if ER21693 >= 99998     
replace ER21696=. if ER21696 >= 99998    
replace ER21700=. if ER21700 >= 99998       
replace ER21703=. if ER21703 >= 99998  

gen fdhm1     = ER21686 
gen fdhm2     = ER21696 
gen fdhmper1  = ER21687 
gen fdhmper2  = ER21697 
gen fddel1    = ER21690 
gen fddel2    = ER21700 
gen fddelper1 = ER21691 
gen fddelper2 = ER21701 
gen fout1    = ER21693 
gen fout2    = ER21703 
gen foutper1 = ER21694 
gen foutper2 = ER21704 

gen fdhm        = fdhm1    		if fdhm1 !=. & fdhm2 == 0
gen fdhmper     = fdhmper1 		if fdhm1 !=. & fdhm2 == 0
replace fdhm    = fdhm2    		if fdhm==.
replace fdhmper = fdhmper2 		if fdhmper==.

gen fddel 		= fddel1 		if fddel1!=. & fddel2 == 0
gen fddelper 	= fddelper1 	if fddel1!=. & fddel2 == 0
replace fddel 	= fddel2 		if fddel==.
replace fddelper = fddelper2 	if fddelper==.

gen fout 		= fout1 		if fout1!=. & fout2 == 0
gen foutper 	= foutper1 	if fout1!=. & fout2 == 0 
replace fout 	= fout2 		if fout==.
replace foutper = foutper2 	if foutper==. 

replace fdhm=. 					if fdhmper == 2 | fdhmper>= 7
replace fdhm=fdhm*52 			if fdhmper == 3
replace fdhm=fdhm*26 			if fdhmper == 4
replace fdhm=fdhm*12 			if fdhmper == 5

replace fddel=. 				if fddelper == 2 | fddelper>= 7
replace fddel=fddel*52 			if fddelper == 3
replace fddel=fddel*26 			if fddelper == 4
replace fddel=fddel*12 			if fddelper == 5

replace fout=.  				if foutper >= 7
replace fout=fout*365   		if foutper == 2
replace fout=fout*52 			if foutper == 3
replace fout=fout*26 			if foutper == 4
replace fout=fout*12 			if foutper == 5

egen food=rsum(fdhm fddel)
replace food=. if fdhm==. & fddel==.

/* Asset Income variables */
ren ER24129 gardeninc
replace gardeninc=. if gardeninc==0

gen roomerinc=.
cap program drop doit
program def doit
      local i=21986
        while `i' < 21998 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER21986-ER21997)
replace roomerinc=ER21983               if ER21984==6
replace roomerinc=ER21983*nm            if ER21984==5
replace roomerinc=ER21983*((26/12)*nm)  if ER21984==4
replace roomerinc=ER21983*((52/12)*nm)  if ER21984==3
replace roomerinc=. 					if ER21983>9999997
drop nm

gen rentinc=.
cap program drop doit
program def doit
      local i=22007
        while `i' < 22019 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER22007-ER22018)
replace rentinc=ER22003                 if ER22004==6
replace rentinc=ER22003*nm              if ER22004==5
replace rentinc=ER22003*((26/12)*nm)    if ER22004==4
replace rentinc=ER22003*((52/12)*nm)    if ER22004==3
replace rentinc=.          				if ER22003>999997
drop nm

gen divinc=.
cap program drop doit
program def doit
      local i=22024
        while `i' < 22036 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER22024-ER22035)
replace divinc=ER22020          if ER22021==6
replace divinc=ER22020*nm       if ER22021==5
replace divinc=. 				if ER22020>999997
drop nm

gen intinc=.
cap program drop doit
program def doit
      local i=22041
        while `i' < 22053 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER22041-ER22052)
replace intinc=ER22037      			if ER22038==6
replace intinc=ER22037*nm   			if ER22038==5
replace intinc=ER22037*((52/12)*nm)  	if ER22038==3
replace intinc=. 						if ER22037>999997
drop nm

gen trustinc=.
cap program drop doit
program def doit
      local i=22057
        while `i' < 22069{
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER22057-ER22068)
replace trustinc=ER22054        if ER22055==6
replace trustinc=ER22054*nm     if ER22055==5
replace trustinc=. 				if ER22054>999997
drop nm

gen aliminc=.
cap program drop doit
program def doit
      local i=22236
        while `i' < 22248 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER22236-ER22247)
replace aliminc=ER22233                 if ER22234==6
replace aliminc=ER22233*nm              if ER22234==5
replace aliminc=ER22233*((26/12)*nm)    if ER22234==4
replace aliminc=ER22233*((52/12)*nm)    if ER22234==3
replace aliminc=. 						if ER22233>99997
drop nm

/* 	NOTE: From 2003 on, questionnaire does not include question about wife's other assets.
	Instead they have a question about wife's rent income, so this is included into families 
	total asset income in 2003, 2005, 2007 and 2009 */

gen wrentinc=.
cap program drop doit
program def doit
      local i=22340
        while `i' < 22352 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER22340-ER22351)
replace wrentinc=ER22336                 if ER22337==6
replace wrentinc=ER22336*nm              if ER22337==5
replace wrentinc=ER22336*((26/12)*nm)    if ER22337==4
replace wrentinc=ER22336*((52/12)*nm)    if ER22337==3
replace wrentinc=.          			if ER22336>999997
drop nm

/* Starting in 2003, for Rent, dividend and interest income, while they ask the head and wife seperately, when they ask
	they ask the head "Does it include wife's share" and the wife if it "Is in addition to husbands income" therefore,
	to avoid double counting must subtract wife's from heads if he says it is joint, etc. */

gen rentincj = ER22006
gen wrentincadd = ER22339

replace rentinc  = rentinc-wrentinc if (rentincj==1 & wrentincadd ==1)
replace wrentinc = wrentinc if (rentincj==1 & wrentincadd ==1)

replace rentinc  = rentinc if (rentincj==5 & wrentincadd ==1)
replace wrentinc = wrentinc if (rentincj==5 & wrentincadd ==1)

replace rentinc  = rentinc if (rentincj==5 & wrentincadd ==5)
replace wrentinc = wrentinc-rentinc if (rentincj==5 & wrentincadd ==5)

/* If husband says it includes his wife's share, and wife says it is not in addition to husbands share, 
	choose the larger amount of the two values they said and set the smaller equal to zero.  */
replace rentinc  = 0 if (rentincj==1 & wrentincadd ==5 & wrentinc>rentinc) 
replace wrentinc = 0 if (rentincj==1 & wrentincadd ==5 & wrentinc<rentinc)

/* If says it is a combined asset but the other asset is missing, then leave asset the same. 
	I.e. only missing if orginal is missing.*/
replace rentinc=rentinc if wrentinc==.
replace wrentinc=wrentinc if rentinc==.
replace rentinc=0 if rentinc<0
replace wrentinc=0 if wrentinc<0 

gen wdivinc=.
cap program drop doit
program def doit
      local i=22357
        while `i' < 22369 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER22357-ER22368)
replace wdivinc=ER22353         if ER22354==6
replace wdivinc=ER22353*nm      if ER22354==5
replace wdivinc=. 				if ER22353>999997
drop nm

/* Same Process as for Rentinc */
gen divincj = ER22023
gen wdivincadd = ER22356

replace divinc  = divinc-wdivinc if (divincj==1 & wdivincadd ==1)
replace wdivinc = wdivinc if (divincj==1 & wdivincadd ==1)

replace divinc  = divinc if (divincj==5 & wdivincadd ==1)
replace wdivinc = wdivinc if (divincj==5 & wdivincadd ==1)

replace divinc  = divinc if (divincj==5 & wdivincadd ==5)
replace wdivinc = wdivinc-divinc if (divincj==5 & wdivincadd ==5)

**If both are combined assets, choose larger and make smaller=0**
replace divinc  = 0 if (divincj==1 & wdivincadd ==5 & wdivinc>divinc) 
replace wdivinc = 0 if (divincj==1 & wdivincadd ==5 & wdivinc<divinc)

replace divinc=divinc if wdivinc==.
replace wdivinc=wdivinc if divinc==.
replace divinc=0 if divinc<0
replace wdivinc=0 if wdivinc<0 
 
gen wintinc=.
cap program drop doit
program def doit
      local i=22374
        while `i' < 22386 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER22374-ER22385)
replace wintinc=ER22370         if ER22371==6
replace wintinc=ER22370*nm      if ER22371==5
replace wintinc=. 				if ER22370>999997
drop nm

/* Same Process as for Rentinc */
gen intincj = ER22040
gen wintincadd = ER22373

replace intinc  = intinc-wintinc if (intincj==1 & wintincadd ==1)
replace wintinc = wintinc if (intincj==1 & wintincadd ==1)

replace intinc  = intinc if (intincj==5 & wintincadd ==1)
replace wintinc = wintinc if (intincj==5 & wintincadd ==1)

replace intinc  = intinc if (intincj==5 & wintincadd ==5)
replace wintinc = wintinc-intinc if (intincj==5 & wintincadd ==5)

**If both are combined assets, choose larger and make smaller=0**
replace intinc  = 0 if (intincj==1 & wintincadd ==5 & wintinc>intinc) 
replace wintinc = 0 if (intincj==1 & wintincadd ==5 & wintinc<intinc)

replace intinc=intinc if wintinc==.
replace wintinc=wintinc if intinc==.
replace intinc=0 if intinc<0
replace wintinc=0 if wintinc<0  

gen wtrustinc=.
cap program drop doit
program def doit
      local i=22390
        while `i' < 22402 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER22390-ER22401)
replace wtrustinc=ER22387       if ER22388==6
replace wtrustinc=ER22387*nm    if ER22388==5
replace wtrustinc=. 			if ER22387>999997
drop nm

/* No Question about women's other asset income from 2003 on. */ 
gen wothinc=.

egen asset_part1=rsum(gardeninc roomerinc rentinc divinc intinc trustinc aliminc wrentinc wdivinc wintinc wtrustinc)

*** Rent ***
gen rent=.
replace rent=ER21072        if ER21073==6
replace rent=ER21072*12     if ER21073==5
replace rent=ER21072*26     if ER21073==4
replace rent=ER21072*52     if ER21073==3
replace rent=ER21072*365  	if ER21073==2
replace rent=. 				if ER21072>99997

*generate labor and asset part of farm income*
gen farmlabor=0
replace farmlabor=0.5*ER21855 	if ER21855>0
replace farmlabor=. 			if ER21855>9999997

gen farmasset=0
replace farmasset=0.5*ER21855 	if ER21855>0
replace farmasset=ER21855 		if ER21855<0
replace farmasset=. 			if ER21855>9999997


/* Income and hours variables */
** Hours Worked **
ren ER24080 hours
ren ER24077 weeks
ren ER24091 hourw
ren ER24088 weekw

** Income **
egen ly= 				rsum(ER24109 farmlabor ER24116) 
egen wly=				rsum(ER24111 ER24135)
gen wages = ER24116
gen wagesw = ER24135
ren ER24102 tyoth
gen truncy=ER24099<1
ren ER24099 y
replace y=1 if y<1

egen asset= rsum(farmasset ER24110 ER24112 asset_part1)
gen trunca=ER24110==-999999

#delimit;
keep id food fout fstmp rent educ fsize age sex weduc agew kids marit race* wrace* house 
outkid empst wempst asset_part1 hours hourw weight* state y 
ly wages wagesw wly tyoth asset smsa truncy trunca 
electric heating water miscutils homeinsure hinsurance nurse doctor prescription totalhealth 
carins carrepair gasoline parking busfare taxifare othertrans tuition otherschool childcare 
cash* bonds* stocks* busval* penval* other_debt* real_estate* carval* mortgage1* mortgage2* fchg; 
#delimit cr

#delimit cr

gen year=102
save "$output\fam102", replace

*****************************************************************************************************************;
************************************ 2005 ***********************************************************
*****************************************************************************************************************;

use $PSID\f05, clear

ren ER25002 id
ren ER25016 fsize
ren ER25017 age
ren ER25018 sex
ren ER25019 agew
ren ER25020 kids
ren ER25023 marit
ren ER27393 race1
ren ER27394 race2	
ren ER27395 race3	
ren ER27396 race4	
ren ER27297 wrace1
ren ER27298 wrace2
ren ER27299 wrace3
ren ER27300 wrace4
ren ER26518 outkid
ren ER25387 selfw
ren ER25029 house
ren ER28044 smsa
ren ER25007 fchg
ren ER28078 weight
ren ER25003 state

/* Home insurance */
ren ER25038 homeinsure
replace homeinsure=. if homeinsure>9997

/* Asset Stock */ 
ren ER26577 cash
replace cash=. if cash>=999999998 | cash==-400 /* -400 is a wild code */ 

ren ER26598 bonds
replace bonds=. if bonds>=999999998

ren ER26549 stocks
replace stocks=. if stocks>=999999998

ren ER26535 real_estate
replace real_estate=. if real_estate>=999999998

ren ER26539 carval
replace carval=. if carval>=999999998

ren ER26544 busval
replace busval=. if busval>=999999998

ren ER26571 penval
replace penval=. if penval>=999999998

ren ER25042 mortgage1
replace mortgage1=. if mortgage1>=9999998

ren ER25053 mortgage2
replace mortgage2=. if mortgage2>=9999998

ren ER26603 other_debt
replace other_debt=. if other_debt>=999999998

/* health expenditures */ 
ren ER27238 hinsurance
ren ER27239 nurse
ren ER27245 doctor
ren ER27251 prescription
ren ER27257 totalhealth

replace hinsurance=. if hinsurance>999997
replace nurse=. if nurse>999997
replace doctor=. if doctor>999997
replace prescription=. if prescription>999997
replace totalhealth=. if totalhealth>99999997

replace hinsurance=hinsurance/2 	 
replace nurse=nurse/2
replace doctor=doctor/2
replace prescription=prescription/2
replace totalhealth=totalhealth/2

/* Utilites */
ren ER25082 electric
replace electric=. if electric>9997 
replace electric=. if (ER25083<3 | ER25083>6 | ER25083==4) & electric!=0
replace electric=52*electric if ER25083==3
replace electric=12*electric if ER25083==5

ren ER25080 heating
replace heating=. if heating>9997
replace heating=. if (ER25081<3 | ER25081>6 | ER25081==4) & heating!=0
replace heating=52*heating if ER25081==3
replace heating=12*heating if ER25081==5

ren ER25084 water
replace water=. if water>9997
replace water=. if (ER25085<3 | ER25085>6 | ER25085==4) & water!=0
replace water=52*water if ER25085==3
replace water=12*water if ER25085==5

ren ER25090 miscutils_t
replace miscutils_t=. if miscutils_t>997
replace miscutils_t=. if (ER25091<3 | ER25091>6 | ER25091==4) & miscutils_t!=0
replace miscutils_t=52*miscutils_t if ER25091==3
replace miscutils_t=12*miscutils_t if ER25091==5
 
ren ER25086 telephone
replace telephone=. if telephone>9997
replace telephone=. if (ER25087<5 | ER25087>6) & telephone!=0
replace telephone=12*telephone if ER25087==5

egen miscutils		= rsum(miscutils_t telephone)
drop telephone miscutils_t 

/* Car expenditure */
ren ER25794 carins
replace carins=. if carins>999997
replace carins=. if (ER25795<5 | ER25795>6) & carins!=0
replace carins=12*carins if ER25795==5

ren ER25798 carrepair
replace carrepair=. if carrepair>99997
replace carrepair=12*carrepair

ren ER25799 gasoline
replace gasoline=. if gasoline>99997
replace gasoline=12*gasoline

ren ER25800 parking
replace parking=. if parking>99997
replace parking=12*parking

ren ER25801 busfare
replace busfare=. if busfare>99997
replace busfare=12*busfare

ren ER25802 taxifare
replace taxifare=. if taxifare>99997
replace taxifare=12*taxifare

ren ER25803 othertrans
replace othertrans=. if othertrans>99997
replace othertrans=12*othertrans

/* Education expenditures */ 
ren ER25805 tuition
replace tuition=. if tuition>999997

ren ER25807 otherschool
replace otherschool=. if otherschool>999997

/* Child Care */ 
ren ER25628 childcare
replace childcare=. if childcare>999997

/* Merge mentions for empst for head */ 
ren ER25104 empst1
ren ER25105 empst2
ren ER25106 empst3

forvalues i=1(1)3 {
gen empst`i'_ordered =. 		
replace empst`i'_ordered = 11 if empst`i'==2
replace empst`i'_ordered = 12 if empst`i'==1
replace empst`i'_ordered = 13 if empst`i'==3
replace empst`i'_ordered = 14 if empst`i'==4
replace empst`i'_ordered = 15 if empst`i'==5  
replace empst`i'_ordered = 16 if empst`i'==7
replace empst`i'_ordered = 17 if empst`i'==6
replace empst`i'_ordered = 18 if empst`i'==8
replace empst`i'_ordered = 19 if empst`i'==9
}

gen empst_ordered=min(empst1_ordered, empst2_ordered, empst3_ordered)

gen empst = . 			
replace empst = 2 if empst_ordered==11
replace empst = 1 if empst_ordered==12
replace empst = 3 if empst_ordered==13
replace empst = 4 if empst_ordered==14
replace empst = 5 if empst_ordered==15
replace empst = 7 if empst_ordered==16
replace empst = 6 if empst_ordered==17
replace empst = 8 if empst_ordered==18
replace empst = 9 if empst_ordered==19

ren ER25362 wempst1
ren ER25363 wempst2
ren ER25364 wempst3

forvalues i=1(1)3 {
gen wempst`i'_ordered =. 		
replace wempst`i'_ordered = 11 if wempst`i'==2
replace wempst`i'_ordered = 12 if wempst`i'==1
replace wempst`i'_ordered = 13 if wempst`i'==3
replace wempst`i'_ordered = 14 if wempst`i'==4
replace wempst`i'_ordered = 15 if wempst`i'==5  
replace wempst`i'_ordered = 16 if wempst`i'==7
replace wempst`i'_ordered = 17 if wempst`i'==6
replace wempst`i'_ordered = 18 if wempst`i'==8
replace wempst`i'_ordered = 19 if wempst`i'==9
}

gen wempst_ordered=min(wempst1_ordered, wempst2_ordered, wempst3_ordered)

gen wempst = . 			
replace wempst = 2 if wempst_ordered==11
replace wempst = 1 if wempst_ordered==12
replace wempst = 3 if wempst_ordered==13
replace wempst = 4 if wempst_ordered==14
replace wempst = 5 if wempst_ordered==15
replace wempst = 7 if wempst_ordered==16
replace wempst = 6 if wempst_ordered==17
replace wempst = 8 if wempst_ordered==18
replace wempst = 9 if wempst_ordered==19

** Education ** 
ren ER28047 educ
replace educ=. if educ==99
ren ER28048 weduc
replace weduc=. if weduc==99 | agew==0

/* Food Related variables */ 
gen       fstmp=0
cap program drop doit
program def doit
        local i=25658
        while `i' < 25670 {
            replace ER`i'=. if ER`i'>1
        	local i=`i'+1
       }
end
qui doit
egen nm=rsum(ER25658-ER25669)
replace fstmp=ER25655              if ER25656==6 
replace fstmp=ER25655*nm           if ER25656==5 
replace fstmp=ER25655*((26/12)*nm) if ER25656==4  
replace fstmp=ER25655*((52/12)*nm) if ER25656==3
replace fstmp=.					 if ER25655>999997  | ER25656>6 
drop nm

replace ER25688=. if ER25688 >= 99998
replace ER25692=. if ER25692 >= 99998
replace ER25695=. if ER25695 >= 99998   
replace ER25698=. if ER25698 >= 99998     
replace ER25702=. if ER25702 >= 99998
replace ER25705=. if ER25705 >= 99998


gen fdhm1     = ER25688 
gen fdhm2     = ER25698 
gen fdhmper1  = ER25689 
gen fdhmper2  = ER25699 
gen fddel1    = ER25692 
gen fddel2    = ER25702 
gen fddelper1 = ER25693 
gen fddelper2 = ER25703
gen fout1    = ER25695
gen fout2    = ER25705
gen foutper1 = ER25696
gen foutper2 = ER25706

gen fdhm        = 	fdhm1    	if fdhm1 !=. & fdhm2 == 0	
gen fdhmper     = 	fdhmper1 	if fdhm1 !=. & fdhm2 == 0	
replace fdhm    = 	fdhm2    	if fdhm==.
replace fdhmper = 	fdhmper2 	if fdhmper==.

gen fddel 		= 	fddel1 		if fddel1!=. & fddel2 == 0		
gen fddelper 	= 	fddelper1 	if fddel1!=. & fddel2 == 0
replace fddel 	= 	fddel2 		if fddel==.
replace fddelper = 	fddelper2 	if fddelper==.

gen fout 		= 	fout1 		if fout1!=. & fout2 == 0
gen foutper 	= 	foutper1 	if fout1!=. & fout2 == 0 
replace fout 	= 	fout2 		if fout==.
replace foutper = 	foutper2 	if foutper==. 

replace fdhm=. 					if fdhmper == 2 | fdhmper>= 7
replace fdhm=fdhm*52 			if fdhmper == 3
replace fdhm=fdhm*26 			if fdhmper == 4
replace fdhm=fdhm*12 			if fdhmper == 5

replace fddel=. 				if fddelper == 2 | fddelper>= 7
replace fddel=fddel*52 			if fddelper == 3
replace fddel=fddel*26 			if fddelper == 4
replace fddel=fddel*12 			if fddelper == 5

replace fout=.  				if foutper >= 7
replace fout=fout*365   		if foutper == 2
replace fout=fout*52 			if foutper == 3
replace fout=fout*26 			if foutper == 4
replace fout=fout*12 			if foutper == 5

egen food=rsum(fdhm fddel)
replace food=. if fdhm==. & fddel==.

/* Asset Income variables */
gen gardeninc=.
replace gardeninc=ER25947 			if ER25948==6
replace gardeninc=. 				if ER25947>9999997

gen roomerinc=.
cap program drop doit
program def doit
      local i=25967
        while `i' < 25979 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER25967-ER25978)
replace roomerinc=ER25964               if ER25965==6
replace roomerinc=ER25964*nm            if ER25965==5
replace roomerinc=ER25964*((26/12)*nm)  if ER25965==4
replace roomerinc=ER25964*((52/12)*nm)  if ER25965==3
replace roomerinc=. 					if ER25964>9999997
drop nm

gen rentinc=.
cap program drop doit
program def doit
      local i=25988
        while `i' < 26000 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER25988-ER25999)
replace rentinc=ER25984                 if ER25985==6
replace rentinc=ER25984*nm              if ER25985==5
replace rentinc=ER25984*((26/12)*nm)    if ER25985==4
replace rentinc=ER25984*((52/12)*nm)    if ER25985==3
replace rentinc=.          				if ER25984>999997
drop nm

gen divinc=.
cap program drop doit
program def doit
      local i=26005
        while `i' < 26017 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER26005-ER26016)
replace divinc=ER26001          		if ER26002==6
replace divinc=ER26001*nm       		if ER26002==5
replace divinc=. 						if ER26001>999997
drop nm

gen intinc=.
cap program drop doit
program def doit
      local i=26022
        while `i' < 26034 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER26022-ER26033)
replace intinc=ER26018      			if ER26019==6
replace intinc=ER26018*nm   			if ER26019==5
replace intinc=ER26018*((52/12)*nm) 	if ER26019==3
replace intinc=. 						if ER26018>999997
drop nm

gen trustinc=.
cap program drop doit
program def doit
      local i=26038
        while `i' < 26050 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER26038-ER26049)
replace trustinc=ER26035        		if ER26036==6
replace trustinc=ER26035*nm     		if ER26036==5
replace trustinc=. 						if ER26035>999997
drop nm

gen aliminc=.
cap program drop doit
program def doit
      local i=26217
        while `i' < 26229 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER26217-ER26228)
replace aliminc=ER26214                 if ER26215==6
replace aliminc=ER26214*nm              if ER26215==5
replace aliminc=ER26214*((52/12)*nm)    if ER26215==3
replace aliminc=. 						if ER26214>999997
drop nm

/* Since no Wife's Other Asset in 2005 (like 2003) include rent income*/
gen wrentinc=.
cap program drop doit
program def doit
      local i=26321
        while `i' < 26333 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER26321-ER26332)
replace wrentinc=ER26317                 if ER26318==6
replace wrentinc=ER26317*nm              if ER26318==5
replace wrentinc=ER26317*((26/12)*nm)    if ER26318==4
replace wrentinc=ER26317*((52/12)*nm)    if ER26318==3
replace wrentinc=.          			if ER26317>999997
drop nm

/* Prevent Double Counting for Wife and Head's combined rent income */
gen rentincj = ER25987
gen wrentincadd = ER26320

replace rentinc  = rentinc-wrentinc if (rentincj==1 & wrentincadd ==1)
replace wrentinc = wrentinc if (rentincj==1 & wrentincadd ==1)

replace rentinc  = rentinc if (rentincj==5 & wrentincadd ==1)
replace wrentinc = wrentinc if (rentincj==5 & wrentincadd ==1)

replace rentinc  = rentinc if (rentincj==5 & wrentincadd ==5)
replace wrentinc = wrentinc-rentinc if (rentincj==5 & wrentincadd ==5)

**If both are combined assets, choose larger and make smaller=0**
replace rentinc  = 0 if (rentincj==1 & wrentincadd ==5 & wrentinc>rentinc) 
replace wrentinc = 0 if (rentincj==1 & wrentincadd ==5 & wrentinc<rentinc)

replace rentinc=rentinc if wrentinc==.
replace wrentinc=wrentinc if rentinc==.
replace rentinc=0 if rentinc<0
replace wrentinc=0 if wrentinc<0 

gen wdivinc=.
cap program drop doit
program def doit
      local i=26338
        while `i' < 26350 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER26338-ER26349)
replace wdivinc=ER26334         		if ER26335==6
replace wdivinc=ER26334*nm      		if ER26335==5
replace wdivinc=. 						if ER26334>999997
drop nm

/* Prevent Double Counting for Wife and Head's combined dividend assets */
gen divincj = ER26004
gen wdivincadd = ER26337

replace divinc  = divinc-wdivinc if (divincj==1 & wdivincadd ==1)
replace wdivinc = wdivinc if (divincj==1 & wdivincadd ==1)

replace divinc  = divinc if (divincj==5 & wdivincadd ==1)
replace wdivinc = wdivinc if (divincj==5 & wdivincadd ==1)

replace divinc  = divinc if (divincj==5 & wdivincadd ==5)
replace wdivinc = wdivinc-divinc if (divincj==5 & wdivincadd ==5)

**If both are combined assets, choose larger and make smaller=0**
replace divinc  = 0 if (divincj==1 & wdivincadd ==5 & wdivinc>divinc) 
replace wdivinc = 0 if (divincj==1 & wdivincadd ==5 & wdivinc<divinc) 

replace divinc=divinc if wdivinc==.
replace wdivinc=wdivinc if divinc==.
replace divinc=0 if divinc<0
replace wdivinc=0 if wdivinc<0 

gen wintinc=.
cap program drop doit
program def doit
      local i=26355
        while `i' < 26367 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER26355-ER26366)
replace wintinc=ER26351         		if ER26352==6
replace wintinc=ER26351*nm      		if ER26352==5
replace wintinc=. 						if ER26351>999997
drop nm

/* Prevent Double Counting for Wife and Head's combined Interest income */
gen intincj = ER26021
gen wintincadd = ER26354

replace intinc  = intinc-wintinc if (intincj==1 & wintincadd ==1)
replace wintinc = wintinc if (intincj==1 & wintincadd ==1)

replace intinc  = intinc if (intincj==5 & wintincadd ==1)
replace wintinc = wintinc if (intincj==5 & wintincadd ==1)

replace intinc  = intinc if (intincj==5 & wintincadd ==5)
replace wintinc = wintinc-intinc if (intincj==5 & wintincadd ==5)

**If both are combined assets, choose larger and make smaller=0**
replace intinc  = 0 if (intincj==1 & wintincadd ==5 & wintinc>intinc) 
replace wintinc = 0 if (intincj==1 & wintincadd ==5 & wintinc<intinc) 

replace intinc=intinc if wintinc==.
replace wintinc=wintinc if intinc==.
replace intinc=0 if intinc<0
replace wintinc=0 if wintinc<0 

gen wtrustinc=.
cap program drop doit
program def doit
      local i=26371
        while `i' < 26383 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER26371-ER26382)
replace wtrustinc=ER26368       		if ER26369==6
replace wtrustinc=ER26368*nm    		if ER26369==5
replace wtrustinc=. 					if ER26368>999997
drop nm

/*No Wife Income from Other Assets like in 2003*/
gen wothinc=.

egen asset_part1=rsum(gardeninc roomerinc rentinc divinc intinc trustinc aliminc wdivinc wintinc wtrustinc)

*** Rent ***
gen rent=.
replace rent=ER25063        			if ER25064==6
replace rent=ER25063*12    			if ER25064==5
replace rent=ER25063*26     			if ER25064==4
replace rent=ER25063*52     			if ER25064==3
replace rent=ER25063*365  			if ER25064==2
replace rent=. 					if ER25063>99997

*** Income variables ***
gen farmlabor=0
replace farmlabor=0.5*ER25836 	if ER25836>0
replace farmlabor=. 			if ER25836>999997

gen farmasset=0
replace farmasset=0.5*ER25836 	if ER25836>0
replace farmasset=ER25836 		if ER25836<0
replace farmasset=. 			if ER25836>999997



/* Income and hours variables */
ren ER27886 hours
ren ER27897 hourw

** Income **
egen ly= 				rsum(ER27910 farmlabor ER27931) 
egen wly=				rsum(ER27940 ER27943)
gen wagesw=ER27943
gen wages= ER27931
ren ER28009 tyoth
gen truncy=ER28037<1
ren ER28037 y
replace y=1 if y<1

egen asset= rsum(farmasset ER27911 ER27941 asset_part1)
gen trunca=ER26001==999997

#delimit;
keep id food fout fstmp rent educ fsize age sex weduc agew kids marit race* wrace* house 
outkid empst wempst asset_part1 hours hourw weight* state y 
ly wages wagesw wly tyoth asset smsa truncy trunca 
electric heating water miscutils homeinsure hinsurance nurse doctor prescription totalhealth 
carins carrepair gasoline parking busfare taxifare othertrans tuition otherschool childcare 
cash* bonds* stocks* busval* penval* other_debt* real_estate* carval* mortgage1* mortgage2* fchg; 
#delimit cr

gen year=104
save "$output\fam104", replace


****************************************************************************************************
**************************************2007******************************************************
********************************************************************************************

use $PSID\f07, clear

ren ER36002 id
ren ER36016 fsize
ren ER36017 age
ren ER36018 sex
ren ER36019 agew
ren ER36020 kids
ren ER36023 marit
ren ER40565 race1
ren ER40566 race2
ren ER40567 race3
ren ER40568 race4
ren ER40472 wrace1
ren ER40473 wrace2
ren ER40474 wrace3
ren ER40475 wrace4
ren ER37536 outkid
ren ER36029 house
ren ER41034 smsa
ren ER36007 fchg
ren ER36003 state
ren ER41069 weight 

/* Home insurance */
ren ER36038 homeinsure
replace homeinsure=. if homeinsure>9997

/* Asset Stock */ 
ren ER37595 cash
replace cash=. if cash>=999999998  

ren ER37616 bonds
replace bonds=. if bonds>=999999998

ren ER37567 stocks
replace stocks=. if stocks>=999999998 | stocks==-4000 | stocks<=-99999999 /* -4,000 and -99,999,999 are wild codes */

ren ER37553 real_estate
replace real_estate=. if real_estate>=999999998 | real_estate<-99999999 /* variable is not defined below -99999999 */

ren ER37557 carval
replace carval=. if carval>=999999998

ren ER37562 busval
replace busval=. if busval>=999999998 | busval<-99999999 /* variable is not defined below -99999999 */

ren ER37589 penval
replace penval=. if penval>=999999998

ren ER36042 mortgage1
replace mortgage1=. if mortgage1>=9999998

ren ER36054 mortgage2
replace mortgage2=. if mortgage2>=9999998

ren ER37621 other_debt
replace other_debt=. if other_debt>=999999998

/* health expenditures */ 
ren ER40410 hinsurance
ren ER40414 nurse
ren ER40420 doctor
ren ER40426 prescription
ren ER40432 totalhealth

replace hinsurance=. if hinsurance>999997
replace nurse=. if nurse>999997
replace doctor=. if doctor>999997
replace prescription=. if prescription>999997
replace totalhealth=. if totalhealth>99999997

replace hinsurance=hinsurance/2 	
replace nurse=nurse/2
replace doctor=doctor/2
replace prescription=prescription/2
replace totalhealth=totalhealth/2

/* Utilites */
ren ER36085 electric
replace electric=. if electric>9997
replace electric=. if (ER36086<3 | ER36086>6 | ER36086==4) & electric!=0 
replace electric=52*electric if ER36086==3
replace electric=12*electric if ER36086==5

ren ER36083 heating
replace heating=. if heating>9997
replace heating=. if (ER36084<3 | ER36084>6 | ER36084==4) & heating!=0
replace heating=52*heating if ER36084==3
replace heating=12*heating if ER36084==5

ren ER36089 water
replace water=. if water>9997
replace water=. if (ER36090<3 | ER36090>6 | ER36090==4) & water!=0
replace water=52*water if ER36090==3
replace water=12*water if ER36090==5

ren ER36095 miscutils_t
replace miscutils_t=. if miscutils_t>997
replace miscutils_t=. if (ER36096<3 & ER36096>0 | ER36096>6 | ER36096==4) & miscutils_t!=0
replace miscutils_t=52*miscutils_t if ER36096==3
replace miscutils_t=12*miscutils_t if ER36096==5

ren ER36091 telephone
replace telephone=. if telephone>9997
replace telephone=. if (ER36092<5 | ER36092>6) & telephone!=0
replace telephone=12*telephone if ER36092==5

egen miscutils		= rsum(miscutils_t telephone)
drop telephone miscutils_t 

/* Car expenditure */
ren ER36812 carins
replace carins=. if carins>999997
replace carins=. if (ER36813<5 | ER36813>6) & carins!=0
replace carins=12*carins if ER36813==5

ren ER36816 carrepair
replace carrepair=. if carrepair>99997
replace carrepair=12*carrepair

ren ER36817 gasoline
replace gasoline=. if gasoline>99997
replace gasoline=12*gasoline

ren ER36818 parking
replace parking=. if parking>99997
replace parking=12*parking

ren ER36819 busfare
replace busfare=. if busfare>99997
replace busfare=12*busfare

ren ER36820 taxifare
replace taxifare=. if taxifare>99997
replace taxifare=12*taxifare

ren ER36821 othertrans
replace othertrans=. if othertrans>99997
replace othertrans=12*othertrans

/* Education expenditures */ 
ren ER36823 tuition
replace tuition=. if tuition>999997

ren ER36825 otherschool
replace otherschool=. if otherschool>999997

/* Child Care */ 
ren ER36633 childcare
replace childcare=. if childcare>999997

/* Merge mentions for empst for head */ 
ren ER36109 empst1
ren ER36110 empst2
ren ER36111 empst3

forvalues i=1(1)3 {
gen empst`i'_ordered =. 		
replace empst`i'_ordered = 11 if empst`i'==2
replace empst`i'_ordered = 12 if empst`i'==1
replace empst`i'_ordered = 13 if empst`i'==3
replace empst`i'_ordered = 14 if empst`i'==4
replace empst`i'_ordered = 15 if empst`i'==5  
replace empst`i'_ordered = 16 if empst`i'==7
replace empst`i'_ordered = 17 if empst`i'==6
replace empst`i'_ordered = 18 if empst`i'==8
replace empst`i'_ordered = 19 if empst`i'==9
}

gen empst_ordered=min(empst1_ordered, empst2_ordered, empst3_ordered)

gen empst = . 			
replace empst = 2 if empst_ordered==11
replace empst = 1 if empst_ordered==12
replace empst = 3 if empst_ordered==13
replace empst = 4 if empst_ordered==14
replace empst = 5 if empst_ordered==15
replace empst = 7 if empst_ordered==16
replace empst = 6 if empst_ordered==17
replace empst = 8 if empst_ordered==18
replace empst = 9 if empst_ordered==19

ren ER36367 wempst1
ren ER36368 wempst2
ren ER36369 wempst3

forvalues i=1(1)3 {
gen wempst`i'_ordered =. 		
replace wempst`i'_ordered = 11 if wempst`i'==2
replace wempst`i'_ordered = 12 if wempst`i'==1
replace wempst`i'_ordered = 13 if wempst`i'==3
replace wempst`i'_ordered = 14 if wempst`i'==4
replace wempst`i'_ordered = 15 if wempst`i'==5  
replace wempst`i'_ordered = 16 if wempst`i'==7
replace wempst`i'_ordered = 17 if wempst`i'==6
replace wempst`i'_ordered = 18 if wempst`i'==8
replace wempst`i'_ordered = 19 if wempst`i'==9
}

gen wempst_ordered=min(wempst1_ordered, wempst2_ordered, wempst3_ordered)

gen wempst = . 			
replace wempst = 2 if wempst_ordered==11
replace wempst = 1 if wempst_ordered==12
replace wempst = 3 if wempst_ordered==13
replace wempst = 4 if wempst_ordered==14
replace wempst = 5 if wempst_ordered==15
replace wempst = 7 if wempst_ordered==16
replace wempst = 6 if wempst_ordered==17
replace wempst = 8 if wempst_ordered==18
replace wempst = 9 if wempst_ordered==19


** Education ** 
ren ER41037 educ
replace educ=. if educ==99
ren ER41038 weduc
replace weduc=. if weduc==99 | agew==0

/* Food Related variables */ 
gen       fstmp=0
cap program drop doit
program def doit
        local i=36676
        while `i' < 36688 {
            replace ER`i'=. if ER`i'>1
        	local i=`i'+1
       }
end
qui doit
egen nm=rsum(ER36676-ER36687)
replace fstmp=ER36673              if ER36674==6 
replace fstmp=ER36673*nm           if ER36674==5 
replace fstmp=ER36673*((26/12)*nm) if ER36674==4  
replace fstmp=ER36673*((52/12)*nm) if ER36674==3
replace fstmp=.					 if ER36673>999997 | ER36674>6 | ER36674==1 
drop nm


replace ER36706=. if ER36706 >= 99998
replace ER36710=. if ER36710 >= 99998
replace ER36713=. if ER36713 >= 99998 
replace ER36716=. if ER36716 >= 99998   
replace ER36720=. if ER36720 >= 99998
replace ER36723=. if ER36723 >= 99998
 

gen fdhm1     = ER36706 
gen fdhm2     = ER36716 
gen fdhmper1  = ER36707 
gen fdhmper2  = ER36717 
gen fddel1    = ER36710 
gen fddel2    = ER36720 
gen fddelper1 = ER36711 
gen fddelper2 = ER36721
gen fout1    = ER36713
gen fout2    = ER36723
gen foutper1 = ER36714
gen foutper2 = ER36724


gen fdhm        	= fdhm1    		if fdhm1 !=. & fdhm2 == 0	
gen fdhmper     	= fdhmper1 		if fdhm1 !=. & fdhm2 == 0	
replace fdhm    	= fdhm2    		if fdhm==.
replace fdhmper 	= fdhmper2 		if fdhmper==.

gen fddel 			= fddel1 		if fddel1!=. & fddel2 == 0		
gen fddelper 		= fddelper1 	if fddel1!=. & fddel2 == 0
replace fddel 		= fddel2 		if fddel==.
replace fddelper 	= fddelper2 	if fddelper==.

gen fout 			= fout1 		if fout1!=. & fout2 == 0
gen foutper 		= foutper1 		if fout1!=. & fout2 == 0 
replace fout 		= fout2 		if fout==.
replace foutper 	= foutper2 		if foutper==. 

replace fdhm=. 					if fdhmper == 2 | fdhmper>= 7
replace fdhm=fdhm*52 			if fdhmper == 3
replace fdhm=fdhm*26 			if fdhmper == 4
replace fdhm=fdhm*12 			if fdhmper == 5

replace fddel=. 				if fddelper == 2 | fddelper>= 7
replace fddel=fddel*52 			if fddelper == 3
replace fddel=fddel*26 			if fddelper == 4
replace fddel=fddel*12 			if fddelper == 5

replace fout=.  				if foutper >= 7
replace fout=fout*365   		if foutper == 2
replace fout=fout*52 			if foutper == 3
replace fout=fout*26 			if foutper == 4
replace fout=fout*12 			if foutper == 5

egen food=rsum(fdhm fddel)
replace food=. if fdhm==. & fddel==.

/* Asset Income variables */
gen gardeninc=.
cap program drop doit
program def doit
      local i=36968
        while `i' < 36980 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER36968-ER36979)
replace gardeninc=ER36965               if ER36966==6
replace gardeninc=ER36965*nm            if ER36966==5
replace gardeninc=ER36965*((26/12)*nm)  if ER36966==4
replace gardeninc=ER36965*((52/12)*nm)  if ER36966==3
replace gardeninc=. 				if ER36965>9999997
drop nm

gen rentinc=.
cap program drop doit
program def doit
      local i=37006
        while `i' < 37018 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER37006-ER37017)
replace rentinc=ER37002                 if ER37003==6
replace rentinc=ER37002*nm              if ER37003==5
replace rentinc=ER37002*((26/12)*nm)    if ER37003==4
replace rentinc=ER37002*((52/12)*nm)    if ER37003==3
replace rentinc=.          			if ER37002>999997
drop nm

gen divinc=.
cap program drop doit
program def doit
      local i=37023
        while `i' < 37035 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER37023-ER37034)
replace divinc=ER37019          if ER37020==6
replace divinc=ER37019*nm       if ER37020==5
replace divinc=. 				if ER37019>999997
drop nm

gen intinc=.
cap program drop doit
program def doit
      local i=37040
        while `i' < 37052 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER37040-ER37051)
replace intinc=ER37036      			if ER37037==6
replace intinc=ER37036*nm   			if ER37037==5
replace intinc=ER37036*((26/12)*nm) 	if ER37037==4
replace intinc=ER37036*((52/12)*nm)  	if ER37037==3
replace intinc=. 					if ER37036>999997
drop nm

gen trustinc=.
cap program drop doit
program def doit
      local i=37056
        while `i' < 37068{
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER37056-ER37067)
replace trustinc=ER37053        if ER37054==6
replace trustinc=ER37053*nm     if ER37054==5
replace trustinc=. 				if ER37053>999997
drop nm

gen aliminc=.
cap program drop doit
program def doit
      local i=37235
        while `i' < 37247 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER37235-ER37246)
replace aliminc=ER37232                 if ER37233==6
replace aliminc=ER37232*nm              if ER37233==5
replace aliminc=ER37232*((26/12)*nm)    if ER37233==4
replace aliminc=ER37232*((52/12)*nm)    if ER37233==3
replace aliminc=. 						if ER37232>999997
drop nm

gen wrentinc=.
cap program drop doit
program def doit
      local i=37339
        while `i' < 37351 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER37339-ER37350)
replace wrentinc=ER37335                 if ER37336==6
replace wrentinc=ER37335*nm              if ER37336==5
replace wrentinc=ER37335*((26/12)*nm)    if ER37336==4
replace wrentinc=ER37335*((52/12)*nm)    if ER37336==3
replace wrentinc=.          			if ER37335>999997
drop nm

/* Prevent Double Counting for Wife and Head's combined Rent income */
gen rentincj = ER37005
gen wrentincadd = ER37338

replace rentinc  = rentinc-wrentinc if (rentincj==1 & wrentincadd ==1)
replace wrentinc = wrentinc if (rentincj==1 & wrentincadd ==1)

replace rentinc  = rentinc if (rentincj==5 & wrentincadd ==1)
replace wrentinc = wrentinc if (rentincj==5 & wrentincadd ==1)

replace rentinc  = rentinc if (rentincj==5 & wrentincadd ==5)
replace wrentinc = wrentinc-rentinc if (rentincj==5 & wrentincadd ==5)

**If both are combined assets, choose larger and make smaller=0**
replace rentinc  = 0 if (rentincj==1 & wrentincadd ==5 & wrentinc>rentinc) 
replace wrentinc = 0 if (rentincj==1 & wrentincadd ==5 & wrentinc<rentinc)

replace rentinc=rentinc if wrentinc==.
replace wrentinc=wrentinc if rentinc==.
replace rentinc=0 if rentinc<0
replace wrentinc=0 if wrentinc<0 

gen wdivinc=.
cap program drop doit
program def doit
      local i=37356
        while `i' < 37368 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER37356-ER37367)
replace wdivinc=ER37352         if ER37353==6
replace wdivinc=ER37352*nm      if ER37353==5
replace wdivinc=. 				if ER37352>999997
drop nm

/* Prevent Double Counting for Wife and Head's combined Dividend income */
gen divincj = ER37018
gen wdivincadd = ER37355

replace divinc  = divinc-wdivinc if (divincj==1 & wdivincadd ==1)
replace wdivinc = wdivinc if (divincj==1 & wdivincadd ==1)

replace divinc  = divinc if (divincj==5 & wdivincadd ==1)
replace wdivinc = wdivinc if (divincj==5 & wdivincadd ==1)

replace divinc  = divinc if (divincj==5 & wdivincadd ==5)
replace wdivinc = wdivinc-divinc if (divincj==5 & wdivincadd ==5)

**If both are combined assets, choose larger and make smaller=0**
replace divinc  = 0 if (divincj==1 & wdivincadd ==5 & wdivinc>divinc) 
replace wdivinc = 0 if (divincj==1 & wdivincadd ==5 & wdivinc<divinc) 

replace divinc=divinc if wdivinc==.
replace wdivinc=wdivinc if divinc==.
replace divinc=0 if divinc<0
replace wdivinc=0 if wdivinc<0 

gen wintinc=.
cap program drop doit
program def doit
      local i=37373
        while `i' < 37385 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER37373-ER37384)
replace wintinc=ER37369         if ER37370==6
replace wintinc=ER37369*nm      if ER37370==5
replace wintinc=. 				if ER37369>999997
drop nm

/* Prevent Double Counting for Wife and Head's combined Interest income */
gen intincj = ER37039
gen wintincadd = ER37372

replace intinc  = intinc-wintinc if (intincj==1 & wintincadd ==1)
replace wintinc = wintinc if (intincj==1 & wintincadd ==1)

replace intinc  = intinc if (intincj==5 & wintincadd ==1)
replace wintinc = wintinc if (intincj==5 & wintincadd ==1)

replace intinc  = intinc if (intincj==5 & wintincadd ==5)
replace wintinc = wintinc-intinc if (intincj==5 & wintincadd ==5)

**If both are combined assets, choose larger and make smaller=0**
replace intinc  = 0 if (intincj==1 & wintincadd ==5 & wintinc>intinc) 
replace wintinc = 0 if (intincj==1 & wintincadd ==5 & wintinc<intinc) 

replace intinc=intinc if wintinc==.
replace wintinc=wintinc if intinc==.
replace intinc=0 if intinc<0
replace wintinc=0 if wintinc<0 

gen wtrustinc=.
cap program drop doit
program def doit
      local i=37389
        while `i' < 37401 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER37389-ER37400)
replace wtrustinc=ER37386       if ER37387==6
replace wtrustinc=ER37386*nm    if ER37387==5
replace wtrustinc=. 			if ER37386>999997
drop nm


gen roomerinc=.
cap program drop doit
program def doit
      local i=36985
        while `i' < 36997 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER36985-ER36996)
replace roomerinc=ER36982               if ER36983==6
replace roomerinc=ER36982*nm            if ER36983==5
replace roomerinc=ER36982*((26/12)*nm)  if ER36983==4
replace roomerinc=ER36982*((52/12)*nm)  if ER36983==3
replace roomerinc=. 					if ER36982>9999997
drop nm

**No Wife's Other Income**
gen wothinc=.

egen asset_part1=rsum(gardeninc roomerinc rentinc divinc intinc trustinc aliminc wrentinc wdivinc wintinc wtrustinc wothinc)

** Rent **
gen rent=.
replace rent=ER36065        			if ER36066==6
replace rent=ER36065*12     			if ER36066==5
replace rent=ER36065*26  				if ER36066==4
replace rent=ER36065*52     			if ER36066==3
replace rent=ER36065*365    			if ER36066==2
replace rent=. 							if ER36065>99997

*generate labor and asset part of farm income*
gen farmlabor=0
replace farmlabor=0.5*ER36854 			if ER36854>0
replace farmlabor=. 					if ER36854>9999997

gen farmasset=0
replace farmasset=0.5*ER36854 			if ER36854>0
replace farmasset=ER36854 				if ER36854<0
replace farmasset=. 					if ER36854>9999997


/* Income and hours variables */
ren ER40876 hours
ren ER40887 hourw

** Income **
egen ly= 				rsum(ER40921 farmlabor ER40900) 
egen wly=				rsum(ER40930 ER40933)
gen wages 	= ER40921
gen wagesw 	= ER40933
ren ER40999 tyoth
gen truncy=ER41027<1
ren ER41027 y
replace y=1 if y<1

egen asset= rsum(farmasset ER40901 ER40931 asset_part1)
gen trunca=0

#delimit;
keep id food fout fstmp rent educ fsize age sex weduc agew kids marit race* wrace* house 
outkid empst wempst asset_part1 hours hourw weight* state y 
ly wages wagesw wly tyoth asset smsa truncy trunca 
electric heating water miscutils homeinsure hinsurance nurse doctor prescription totalhealth 
carins carrepair gasoline parking busfare taxifare othertrans tuition otherschool childcare 
cash* bonds* stocks* busval* penval* other_debt* real_estate* carval* mortgage1* mortgage2* fchg; 
#delimit cr

#delimit cr
gen year=106
save "$output\fam106", replace

*****************************************************************************************************************;
************************************ 2009 ***********************************************************
*****************************************************************************************************************;
use $PSID\f09, clear

ren ER42002 id
ren ER42016 fsize
ren ER42017 age
ren ER42018 sex
ren ER42019 agew
ren ER42020 kids
ren ER42023 marit
ren ER46543 race1
ren ER46544 race2
ren ER46545 race3
ren ER46546 race4
ren ER46449 wrace1
ren ER46450 wrace2
ren ER46451 wrace3
ren ER46452 wrace4
ren ER43527 outkid
ren ER42030 house
ren ER46976 smsa
ren ER42007 fchg
ren ER42003 state
ren ER47012 weight 

/* Home insurance */ 
ren ER42039 homeinsure
replace homeinsure=. if homeinsure>9997

/* Asset Stock */ 
ren ER43586 cash
replace cash=. if cash>=999999998  

ren ER43607 bonds
replace bonds=. if bonds>=999999998

ren ER43558 stocks
replace stocks=. if stocks>=999999998 | stocks<=-99999999  /* -99,999,999 stands for loss but don't know how much (from 2009) */

ren ER43544 real_estate
replace real_estate=. if real_estate>=999999998 | real_estate<=-99999999  /* -99,999,999 stands for loss but don't know how much (from 2009) */

ren ER43548 carval
replace carval=. if carval>=999999998 | carval<=-99999999  /* -99,999,999 stands for loss but don't know how much (from 2009) */

ren ER43553 busval
replace busval=. if busval>=999999998 | busval<=-99999999  /* -99,999,999 stands for loss but don't know how much (from 2009) */

ren ER43580 penval
replace penval=. if penval>=999999998

ren ER42043 mortgage1
replace mortgage1=. if mortgage1>=9999998

ren ER42062 mortgage2
replace mortgage2=. if mortgage2>=9999998

ren ER43612 other_debt
replace other_debt=. if other_debt>=999999998

/* health expenditures */ 
ren ER46383 hinsurance
ren ER46387 nurse
ren ER46393 doctor
ren ER46399 prescription
ren ER46404 totalhealth

replace hinsurance=. if hinsurance>999997
replace nurse=. if nurse>999997
replace doctor=. if doctor>999997
replace prescription=. if prescription>999997
replace totalhealth=. if totalhealth>99999997

replace hinsurance=hinsurance/2 	 
replace nurse=nurse/2
replace doctor=doctor/2
replace prescription=prescription/2
replace totalhealth=totalhealth/2

/* Utilites */
ren ER42114 electric
replace electric=. if electric>9997
replace electric=. if (ER42115<5 | ER42115>6) & electric!=0 
replace electric=12*electric if ER42115==5

ren ER42112 heating
replace heating=. if heating>9997
replace heating=. if (ER42113<3 | ER42113>6 | ER42113==4) & heating!=0
replace heating=52*heating if ER42113==3
replace heating=12*heating if ER42113==5

ren ER42118 water
replace water=. if water>9997
replace water=. if (ER42119<5 | ER42119>6) & water!=0
replace water=12*water if ER42119==5

ren ER42124 miscutils_t
replace miscutils_t=. if miscutils_t>997
replace miscutils_t=. if (ER42125<5 | ER42125>6) & miscutils_t!=0
replace miscutils_t=12*miscutils_t if ER42125==5

ren ER42120 telephone
replace telephone=. if telephone>9997
replace telephone=. if (ER42121<5 | ER42121>6) & telephone!=0
replace telephone=12*telephone if ER42121==5

egen miscutils		= rsum(miscutils_t telephone)
drop telephone miscutils_t 

/* Car expenditure */
ren ER42803 carins
replace carins=. if carins>999997
replace carins=. if (ER42804<5 | ER42804>6) & carins!=0
replace carins=12*carins if ER42804==5

ren ER42807 carrepair
replace carrepair=. if carrepair>99997
replace carrepair=12*carrepair

ren ER42808 gasoline
replace gasoline=. if gasoline>99997
replace gasoline=12*gasoline

ren ER42809 parking
replace parking=. if parking>99997
replace parking=12*parking

ren ER42810 busfare
replace busfare=. if busfare>99997
replace busfare=12*busfare

ren ER42811 taxifare
replace taxifare=. if taxifare>99997
replace taxifare=12*taxifare

ren ER42812 othertrans
replace othertrans=. if othertrans>99997
replace othertrans=12*othertrans


/* Education expenditures */ 
ren ER42814 tuition
replace tuition=. if tuition>999997

ren ER42816 otherschool
replace otherschool=. if otherschool>999997

/* Child Care */ 
ren ER42652 childcare
replace childcare=. if childcare>999997

/* Merge mentions for empst for head */ 
ren ER42140 empst1
ren ER42141 empst2
ren ER42142 empst3

forvalues i=1(1)3 {
gen empst`i'_ordered =. 		
replace empst`i'_ordered = 11 if empst`i'==2
replace empst`i'_ordered = 12 if empst`i'==1
replace empst`i'_ordered = 13 if empst`i'==3
replace empst`i'_ordered = 14 if empst`i'==4
replace empst`i'_ordered = 15 if empst`i'==5  
replace empst`i'_ordered = 16 if empst`i'==7
replace empst`i'_ordered = 17 if empst`i'==6
replace empst`i'_ordered = 18 if empst`i'==8
replace empst`i'_ordered = 19 if empst`i'==9
}

gen empst_ordered=min(empst1_ordered, empst2_ordered, empst3_ordered)

gen empst = . 			
replace empst = 2 if empst_ordered==11
replace empst = 1 if empst_ordered==12
replace empst = 3 if empst_ordered==13
replace empst = 4 if empst_ordered==14
replace empst = 5 if empst_ordered==15
replace empst = 7 if empst_ordered==16
replace empst = 6 if empst_ordered==17
replace empst = 8 if empst_ordered==18
replace empst = 9 if empst_ordered==19

ren ER42392 wempst1
ren ER42393 wempst2
ren ER42394 wempst3

forvalues i=1(1)3 {
gen wempst`i'_ordered =. 		
replace wempst`i'_ordered = 11 if wempst`i'==2
replace wempst`i'_ordered = 12 if wempst`i'==1
replace wempst`i'_ordered = 13 if wempst`i'==3
replace wempst`i'_ordered = 14 if wempst`i'==4
replace wempst`i'_ordered = 15 if wempst`i'==5  
replace wempst`i'_ordered = 16 if wempst`i'==7
replace wempst`i'_ordered = 17 if wempst`i'==6
replace wempst`i'_ordered = 18 if wempst`i'==8
replace wempst`i'_ordered = 19 if wempst`i'==9
}

gen wempst_ordered=min(wempst1_ordered, wempst2_ordered, wempst3_ordered)

gen wempst = . 			
replace wempst = 2 if wempst_ordered==11
replace wempst = 1 if wempst_ordered==12
replace wempst = 3 if wempst_ordered==13
replace wempst = 4 if wempst_ordered==14
replace wempst = 5 if wempst_ordered==15
replace wempst = 7 if wempst_ordered==16
replace wempst = 6 if wempst_ordered==17
replace wempst = 8 if wempst_ordered==18
replace wempst = 9 if wempst_ordered==19

** Education ** 
ren ER46981 educ
replace educ=. if educ==99
ren ER46982 weduc
replace weduc=. if weduc==99 | agew==0

/* Food Related variables */ 
gen       fstmp=0
cap program drop doit
program def doit
        local i=42695
        while `i' < 42707 {
            replace ER`i'=. if ER`i'>1
        	local i=`i'+1
       }
end
qui doit
egen nm=rsum(ER42695-ER42706)
replace fstmp=ER42692              if ER42693==6 
replace fstmp=ER42692*nm           if ER42693==5 
replace fstmp=ER42692*((26/12)*nm) if ER42693==4  
replace fstmp=ER42692*((52/12)*nm) if ER42693==3
replace fstmp=.					 if ER42692>999997 | ER42693>6 | ER42693==1 
drop nm


replace ER42712=. if ER42712 >= 99998
replace ER42716=. if ER42716 >= 99998
replace ER42719=. if ER42719 >= 99998 
replace ER42722=. if ER42722 >= 99998   
replace ER42726=. if ER42726 >= 99998
replace ER42729=. if ER42729 >= 99998
 

gen fdhm1     = ER42712 
gen fdhm2     = ER42722 
gen fdhmper1  = ER42713 
gen fdhmper2  = ER42723 
gen fddel1    = ER42716 
gen fddel2    = ER42726 
gen fddelper1 = ER42717 
gen fddelper2 = ER42727
gen fout1    = ER42719
gen fout2    = ER42729
gen foutper1 = ER42720
gen foutper2 = ER42730


gen fdhm        	= fdhm1    		if fdhm1 !=. & fdhm2 == 0	
gen fdhmper     	= fdhmper1 		if fdhm1 !=. & fdhm2 == 0	
replace fdhm    	= fdhm2    		if fdhm==.
replace fdhmper 	= fdhmper2 		if fdhmper==.

gen fddel 			= fddel1 		if fddel1!=. & fddel2 == 0		
gen fddelper 		= fddelper1 	if fddel1!=. & fddel2 == 0
replace fddel 		= fddel2 		if fddel==.
replace fddelper 	= fddelper2 	if fddelper==.

gen fout 			= fout1 		if fout1!=. & fout2 == 0
gen foutper 		= foutper1 		if fout1!=. & fout2 == 0 
replace fout 		= fout2 		if fout==.
replace foutper 	= foutper2 		if foutper==. 

replace fdhm=. 					if fdhmper == 2 | fdhmper>= 7
replace fdhm=fdhm*52 			if fdhmper == 3
replace fdhm=fdhm*26 			if fdhmper == 4
replace fdhm=fdhm*12 			if fdhmper == 5

replace fddel=. 				if fddelper == 2 | fddelper>= 7
replace fddel=fddel*52 			if fddelper == 3
replace fddel=fddel*26 			if fddelper == 4
replace fddel=fddel*12 			if fddelper == 5

replace fout=.  				if foutper >= 7
replace fout=fout*365   		if foutper == 2
replace fout=fout*52 			if foutper == 3
replace fout=fout*26 			if foutper == 4
replace fout=fout*12 			if foutper == 5

egen food=rsum(fdhm fddel)
replace food=. if fdhm==. & fddel==.

/* Asset Income variables */
gen gardeninc=.
cap program drop doit
program def doit
      local i=42959
        while `i' < 42971 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER42959-ER42970)
replace gardeninc=ER42956               if ER42957==6
replace gardeninc=ER42956*nm            if ER42957==5
replace gardeninc=ER42956*((26/12)*nm)  if ER42957==4
replace gardeninc=ER42956*((52/12)*nm)  if ER42957==3
replace gardeninc=. 				if ER42956>9999997
drop nm

gen rentinc=.
cap program drop doit
program def doit
      local i=42997
        while `i' < 43009{
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER42997-ER43008)
replace rentinc=ER42993                 if ER42994==6
replace rentinc=ER42993*nm              if ER42994==5
replace rentinc=ER42993*((26/12)*nm)    if ER42994==4
replace rentinc=ER42993*((52/12)*nm)    if ER42994==3
replace rentinc=.          			if ER42993>999997
drop nm

gen divinc=.
cap program drop doit
program def doit
      local i=43014
        while `i' < 43026 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER43014-ER43025)
replace divinc=ER43010          if ER43011==6
replace divinc=ER43010*nm       if ER43011==5
replace divinc=. 				if ER43010>999997
drop nm

gen intinc=.
cap program drop doit
program def doit
      local i=43031
        while `i' < 43043 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER43031-ER43042)
replace intinc=ER43027      			if ER43028==6
replace intinc=ER43027*nm   			if ER43028==5
replace intinc=ER43027*((26/12)*nm) 	if ER43028==4
replace intinc=ER43027*((52/12)*nm)  	if ER43028==3
replace intinc=. 					if ER43027>999997
drop nm

gen trustinc=.
cap program drop doit
program def doit
      local i=43047
        while `i' < 43059{
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER43047-ER43058)
replace trustinc=ER43044        if ER43045==6
replace trustinc=ER43044*nm     if ER43045==5
replace trustinc=. 				if ER43044>999997
drop nm

gen aliminc=.
cap program drop doit
program def doit
      local i=43226
        while `i' < 43238 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER43226-ER43237)
replace aliminc=ER43223                 if ER43224==6
replace aliminc=ER43223*nm              if ER43224==5
replace aliminc=ER43223*((26/12)*nm)    if ER43224==4
replace aliminc=ER43223*((52/12)*nm)    if ER43224==3
replace aliminc=. 						if ER43223>999997
drop nm

gen wrentinc=.
cap program drop doit
program def doit
      local i=43330
        while `i' < 43342 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER43330-ER43341)
replace wrentinc=ER43326                 if ER43327==6
replace wrentinc=ER43326*nm              if ER43327==5
replace wrentinc=ER43326*((26/12)*nm)    if ER43327==4
replace wrentinc=ER43326*((52/12)*nm)    if ER43327==3
replace wrentinc=.          			if ER43326>999997
drop nm

/* Prevent Double Counting for Wife and Head's combined Rent income */
gen rentincj = ER42996
gen wrentincadd = ER43329

replace rentinc  = rentinc-wrentinc if (rentincj==1 & wrentincadd ==1)
replace wrentinc = wrentinc if (rentincj==1 & wrentincadd ==1)

replace rentinc  = rentinc if (rentincj==5 & wrentincadd ==1)
replace wrentinc = wrentinc if (rentincj==5 & wrentincadd ==1)

replace rentinc  = rentinc if (rentincj==5 & wrentincadd ==5)
replace wrentinc = wrentinc-rentinc if (rentincj==5 & wrentincadd ==5)

**If both are combined assets, choose larger and make smaller=0**
replace rentinc  = 0 if (rentincj==1 & wrentincadd ==5 & wrentinc>rentinc) 
replace wrentinc = 0 if (rentincj==1 & wrentincadd ==5 & wrentinc<rentinc)

replace rentinc=rentinc if wrentinc==.
replace wrentinc=wrentinc if rentinc==.
replace rentinc=0 if rentinc<0
replace wrentinc=0 if wrentinc<0 

gen wdivinc=.
cap program drop doit
program def doit
      local i=43347
        while `i' < 43359 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER43347-ER43358)
replace wdivinc=ER43343         if ER43344==6
replace wdivinc=ER43343*nm      if ER43344==5
replace wdivinc=. 				if ER43343>999997
drop nm

/* Prevent Double Counting for Wife and Head's combined Dividend income */
gen divincj = ER43009
gen wdivincadd = ER43346

replace divinc  = divinc-wdivinc if (divincj==1 & wdivincadd ==1)
replace wdivinc = wdivinc if (divincj==1 & wdivincadd ==1)

replace divinc  = divinc if (divincj==5 & wdivincadd ==1)
replace wdivinc = wdivinc if (divincj==5 & wdivincadd ==1)

replace divinc  = divinc if (divincj==5 & wdivincadd ==5)
replace wdivinc = wdivinc-divinc if (divincj==5 & wdivincadd ==5)

**If both are combined assets, choose larger and make smaller=0**
replace divinc  = 0 if (divincj==1 & wdivincadd ==5 & wdivinc>divinc) 
replace wdivinc = 0 if (divincj==1 & wdivincadd ==5 & wdivinc<divinc) 

replace divinc=divinc if wdivinc==.
replace wdivinc=wdivinc if divinc==.
replace divinc=0 if divinc<0
replace wdivinc=0 if wdivinc<0 

gen wintinc=.
cap program drop doit
program def doit
      local i=43364
        while `i' < 43376 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER43364-ER43375)
replace wintinc=ER43360         if ER43361==6
replace wintinc=ER43360*nm      if ER43361==5
replace wintinc=. 				if ER43360>999997
drop nm

/* Prevent Double Counting for Wife and Head's combined Interest income */
gen intincj = ER43030
gen wintincadd = ER43363

replace intinc  = intinc-wintinc if (intincj==1 & wintincadd ==1)
replace wintinc = wintinc if (intincj==1 & wintincadd ==1)

replace intinc  = intinc if (intincj==5 & wintincadd ==1)
replace wintinc = wintinc if (intincj==5 & wintincadd ==1)

replace intinc  = intinc if (intincj==5 & wintincadd ==5)
replace wintinc = wintinc-intinc if (intincj==5 & wintincadd ==5)

**If both are combined assets, choose larger and make smaller=0**
replace intinc  = 0 if (intincj==1 & wintincadd ==5 & wintinc>intinc) 
replace wintinc = 0 if (intincj==1 & wintincadd ==5 & wintinc<intinc) 

replace intinc=intinc if wintinc==.
replace wintinc=wintinc if intinc==.
replace intinc=0 if intinc<0
replace wintinc=0 if wintinc<0 

gen wtrustinc=.
cap program drop doit
program def doit
      local i=43380
        while `i' < 43392 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER43380-ER43391)
replace wtrustinc=ER43377       if ER43378==6
replace wtrustinc=ER43377*nm    if ER43378==5
replace wtrustinc=. 			if ER43377>999997
drop nm


gen roomerinc=.
cap program drop doit
program def doit
      local i=42976
        while `i' < 42988 {
            replace ER`i'=. if ER`i'>1
        local i=`i'+1
                }
end
qui doit
egen nm=rsum(ER42976-ER42987)
replace roomerinc=ER42973               if ER42974==6
replace roomerinc=ER42973*nm            if ER42974==5
replace roomerinc=ER42973*((26/12)*nm)  if ER42974==4
replace roomerinc=ER42973*((52/12)*nm)  if ER42974==3
replace roomerinc=. 					if ER42973>9999997
drop nm

**No Wife's Other Income**
gen wothinc=.

egen asset_part1=rsum(gardeninc roomerinc rentinc divinc intinc trustinc aliminc wrentinc wdivinc wintinc wtrustinc wothinc)

** Rent **
gen rent=.
replace rent=ER42080        			if ER42081==6
replace rent=ER42080*12     			if ER42081==5
replace rent=ER42080*26  				if ER42081==4
replace rent=ER42080*52     			if ER42081==3
replace rent=ER42080*365    			if ER42081==2
replace rent=. 							if ER42080>99997

*generate labor and asset part of farm income*
gen farmlabor=0
replace farmlabor=0.5*ER42845 			if ER42845>0
replace farmlabor=. 					if ER42845>9999997

gen farmasset=0
replace farmasset=0.5*ER42845 			if ER42845>0
replace farmasset=ER42845 				if ER42845<0
replace farmasset=. 					if ER42845>9999997


/* Income and hours variables */
** Hours Worked **
ren ER46767 hours
ren ER46788 hourw

** Income **
egen ly= 				rsum(ER46829 farmlabor ER46808) 
egen wly=				rsum(ER46838 ER46841)
gen wages 	= ER46829
gen wagesw 	= ER46841
ren ER46907 tyoth
gen truncy=ER46935<1
ren ER46935 y
replace y=1 if y<1

egen asset= rsum(farmasset ER46809 ER46839 asset_part1)
gen trunca=0

#delimit;
keep id food fout fstmp rent educ fsize age sex weduc agew kids marit race* wrace* house 
outkid empst wempst asset_part1 hours hourw weight* state y 
ly wages wagesw wly tyoth asset smsa truncy trunca 
electric heating water miscutils homeinsure hinsurance nurse doctor prescription totalhealth 
carins carrepair gasoline parking busfare taxifare othertrans tuition otherschool childcare 
cash* bonds* stocks* busval* penval* other_debt* real_estate* carval* mortgage1* mortgage2* fchg; 
#delimit cr

#delimit cr
gen year=108
save "$output\fam108", replace

***************************
*** Compress and append ***
***************************
local i=98
while `i' < 109 {
	u "$output\fam`i'",clear
	sort id
	compress
	save,replace 
local i=`i'+2
}
clear

use "$output\fam98", clear
append using "$output\fam100"
append using "$output\fam102"
append using "$output\fam104"
append using "$output\fam106"
append using "$output\fam108"


/* For some variables replace DK and wild codes to missing */ 
replace house =. if house>9999997
replace state =. if state>51

replace age = . if age==999 & year>=98
replace agew = . if agew==999 & year>=98


forvalues i=1/4 {
	replace race`i' =. if race`i'>7 | race`i'==0 	
	replace wrace`i' =. if wrace`i'>7 | wrace`i'==0 
}
replace hourw = . if agew==0 |  (hourw > 5840) /* This variable is generated with 0 hours of work for the wife 
															when wife does not exist (agew==0). Also, the variable's range is up to 5840 */
replace hours = . if (hours > 5840) 			/* The variable's range is up to 5840 */ 
replace sex = . if sex == 0 					
replace marit =. if marit>7 | marit==0 
replace outkid =. if outkid>7 | outkid==0 /* Assigned "." for outkid==0 which is always a wild code */

/* label */
label var food "Food at home + food delivered"
label var fout "Food out"
label var fstmp "Food stamps"
label var rent "Monthly rent"
label var educ "Education of head (17 is 17 or more)" 
label var weduc "Education of wife (17 is 17 or more)"
label var fsize "Number of Persons in FU"
label var kids "Number of Persons Now in the FU Under 18 Years of Age" 
label var marit "(1)Married, (2)Never married, (3)widowed, (4)divorced, (5)separated"
label var wrace1 "wife: (1)White,(2)Black,(3)American Indian, (4)Asian, others change"
label var race1 "head: (1)White,(2)Black,(3)American Indian, (4)Asian, others change"
label var house "Self reported house value" 
label var outkid "Support someone who is not living in the FU"
label var empst "Head work status" 
label var wempst "wife work status" 
label var hinsurance "health insurance"
label var nurse "out of pocket for nursing home, hospital"
label var doctor "out of pocket for doctor, outpatient surge, dental"
label var prescription "out of pocket prescription, in-home mc"
label var totalhealth "total medicare cost inc Medicare and Medicaid"
label var homeinsure "home owner insurance"
label var electric "Electricity expenditure"
label var heating "Heating expenditure" 
label var water "water expenditure" 
label var miscutils "Other utilities (e.g. phone and cable)" 
label var carins "car insurance"
label var carrepair "car repair"
label var gasoline "gasoline cost"
label var parking "parking and car pool"
label var busfare "bus and train fares"
label var taxifare "taxi fares"
label var othertrans "other transportation expenditure"
label var tuition "tuition room and board (not day care)" 
label var otherschool "other school related expenses"
label var childcare "childcare expenses"
label var tyoth "Taxable Income Other Family Unit Members"
label var wages "Labor income of head"
label var wagesw "Labor income of wife"
label var ly "Labor + labor Part of Bus + labor part of farm income head"
label var wly "Labor + labor Part of Bus wife"
label var fchg "Family Composition Change from last int"
label var hours "Head's Annual Hours Worked (all jobs inc overtime)"
label var hourw "Wife's Annual Hours Worked (all jobs inc overtime)"
label var id "Interview Number"
label var cash "Value of cash and savings"
label var bonds "Value of bonds"
label var stocks "Value of stocks"
label var real_estate "NET value of real-estate excluding main house"
label var carval "NET value of cars"
label var busval "NET value of the share household has in business or form"
label var penval "Value of pensions and annuities"
label var mortgage1 "Remaining principal for the first mortgage on the main house"
label var mortgage2 "Remaining principal for the second mortgage on the main house"
label var other_debt "Debt other than mortgage, car loans and anything not reported elsewhere"
 
sort id year
save "$output\fam", replace

local i=98
while `i' < 109 {
	erase $output\fam`i'.dta
	local i=`i'+2
}

cd "$do"

