
*** User Parameters ***
tsset person year

scalar r	= 0.02	/* Interest rate for calculation of pi */ 
scalar tau	= 0		/* This allows for linear tax rate, but not used in the paper anymore, because it is less realistic than progressive tax rate implemented below */ 
scalar T 	= 65	/* Assumed retirement age */ 

*** Prepare variables 
gen age2 = age^2
gen age3 = age^3

gen agew2 = agew^2
gen agew3 = agew^3

* Note in this case we take 5 years birth cohorts to avoid omitted cohorts for the wife
gen wyb_coh=1
replace wyb_coh=2 if wyb>40 	
replace wyb_coh=3 if wyb>45
replace wyb_coh=4 if wyb>50
replace wyb_coh=5 if wyb>55
replace wyb_coh=6 if wyb>60
replace wyb_coh=7 if wyb>65
replace wyb_coh=8 if wyb>70
replace wyb_coh=9 if wyb>75
replace wyb_coh=10 if wyb>80 	

*===================
* Prepare matrices for prediction of husband and wife (for s_hat)
*===================
*** For the prediction of husband's earnings 
xi	i.educ*age i.educ*age2 i.educ*age3 i.race*age i.race*age2 i.race*age3 i.yb
reg log_y  age age2 age3 _I*

matrix beta_male = e(b) 

su yb
local yb_min = r(min)+1	/* Since first dummy is ommited */ 
local yb_max = r(max)
local jj_min = `yb_min'+1

mata
yb_string = " _Iyb_" + st_local("yb_min")
jj = strtoreal(st_local("jj_min"))

while (jj<=strtoreal(st_local("yb_max"))) {
	yb_string = yb_string + " _Iyb_" + strofreal(jj)
	jj = jj + 1
}

data 	= st_data(.,("age","age2","age3","_Ieduc_2","_Ieduc_3", ///
					"_IeduXage_2","_IeduXage_3","_IeduXage2_2","_IeduXage2_3","_IeduXage3_2", ///
					"_IeduXage3_3","_Irace_2","_Irace_3","_IracXage_2","_IracXage_3","_IracXage2_2", ///
					"_IracXage2_3","_IracXage3_2","_IracXage3_3",yb_string))
				
ones 	= J(rows(data),1,1)
data 	= data,ones
beta_male 	= st_matrix("beta_male")

data1	= data	
beta1	= beta_male

end
drop _I*

*** For the prediction of wife's earnings 
xi	i.weduc*agew i.weduc*agew2 i.weduc*agew3 i.wrace*agew i.wrace*agew2 i.wrace*agew3 i.wyb_coh

reg log_yw  agew agew2 agew3 _I* inv_mills
matrix beta_female = e(b) 

* Participation equations for wife using only age and fixed covariates 
probit wife_employed  agew agew2 agew3 _I*
matrix beta_prob = e(b) 

su wyb_coh
local yb_max = r(max)

mata
yb_string = ""
jj = 2

while (jj<=strtoreal(st_local("yb_max"))) {
	yb_string = yb_string + " _Iwyb_coh_" + strofreal(jj)
	jj = jj + 1
}

data 	= st_data(.,("agew","agew2","agew3","_Iweduc_2","_Iweduc_3", ///
					"_IwedXagew_2","_IwedXagew_3","_IwedXagew2_2","_IwedXagew2_3","_IwedXagew3_2", ///
					"_IwedXagew3_3","_Iwrace_2","_Iwrace_3","_IwraXagew_2","_IwraXagew_3","_IwraXagew2_2", ///
					"_IwraXagew2_3","_IwraXagew3_2","_IwraXagew3_3",yb_string))
						
ones 	= J(rows(data),1,1)
data 	= data,ones

beta_female = st_matrix("beta_female")
beta_female	= beta_female[1..(cols(beta_female)-2)],beta_female[cols(beta_female)] // Since we don't want the coefficient on the Mills ratio
beta_prob = st_matrix("beta_prob")

data2	= data	
beta2	= beta_female
beta2p	= beta_prob

end

*===================
* Calculate pi and s
*===================
* The following calcualtes Q1 
xi	i.educ*age i.educ*age2 i.educ*age3 i.race*age i.race*age2 i.race*age3 i.weduc*agew i.weduc*agew2 i.weduc*agew3 i.wrace*agew i.wrace*agew2 i.wrace*agew3 i.yb  i.wyb_coh
gen log_toty_at = (1-tau)*log_toty + log(xsi)
reg log_toty_at age age2 age3 agew agew2 agew3 _I* 

matrix beta_tot = e(b) 

gen Q1 = .

su yb
local yb_min = r(min)+1	/* Since first dummy is omitted */ 
local yb_max = r(max)
local jj_min = `yb_min'+1

su wyb_coh
local wyb_max = r(max)

mata
r		= st_numscalar("r")
tau		= st_numscalar("tau")
T		= st_numscalar("T")

yb_string = " _Iyb_" + st_local("yb_min")
jj = strtoreal(st_local("jj_min"))

while (jj<=strtoreal(st_local("yb_max"))) {
	yb_string = yb_string + " _Iyb_" + strofreal(jj)
	jj = jj + 1
}

wyb_string = ""
jj = 2

while (jj<=strtoreal(st_local("wyb_max"))) {
	wyb_string = wyb_string + " _Iwyb_coh_" + strofreal(jj)
	jj = jj + 1
}


data 	= st_data(.,("age","age2","age3", "agew","agew2","agew3", /// 
					"_Ieduc_2","_Ieduc_3", "_IeduXage_2","_IeduXage_3","_IeduXage2_2","_IeduXage2_3","_IeduXage3_2", ///
					"_IeduXage3_3","_Irace_2","_Irace_3","_IracXage_2","_IracXage_3","_IracXage2_2", ///
					"_IracXage2_3","_IracXage3_2","_IracXage3_3", /// 
					"_Iweduc_2","_Iweduc_3", "_IwedXagew_2","_IwedXagew_3","_IwedXagew2_2","_IwedXagew2_3", ///
					"_IwedXagew3_2", "_IwedXagew3_3","_Iwrace_2","_Iwrace_3","_IwraXagew_2","_IwraXagew_3","_IwraXagew2_2", ///
					"_IwraXagew2_3","_IwraXagew3_2","_IwraXagew3_3",yb_string,wyb_string))
				
ones 	= J(rows(data),1,1)
data 	= data,ones

beta_tot 	= st_matrix("beta_tot")
Q1		= J(rows(data),1,0)

data3	= data
beta3	=beta_tot

i=1
j=0

while (i>0) {
	index	= data[.,1]:<T+1				 
	i		= max(index)
	Q1 		= Q1 + exp((data*beta_tot') - j*log(1+r)*ones):*index
	data[.,1]	= data[.,1] + ones
	data[.,2]	= data[.,1]:^2
	data[.,3]	= data[.,1]:^3
	data[.,4]	= data[.,4] + ones
	data[.,5]	= data[.,4]:^2
	data[.,6]	= data[.,4]:^3
	data[.,9]	= data[.,1]:*data[.,7]
	data[.,10]	= data[.,1]:*data[.,8]
	data[.,11]	= data[.,2]:*data[.,7]
	data[.,12]	= data[.,2]:*data[.,8]
	data[.,13]	= data[.,3]:*data[.,7]
	data[.,14]	= data[.,3]:*data[.,8]
	data[.,17]	= data[.,1]:*data[.,15]
	data[.,18]	= data[.,1]:*data[.,16]
	data[.,19]	= data[.,2]:*data[.,15]
	data[.,20]	= data[.,2]:*data[.,16]
	data[.,21]	= data[.,3]:*data[.,15]
	data[.,22]	= data[.,3]:*data[.,16]
	data[.,25]	= data[.,4]:*data[.,23]
	data[.,26]	= data[.,4]:*data[.,24]
	data[.,27]	= data[.,5]:*data[.,23]
	data[.,28]	= data[.,5]:*data[.,24]
	data[.,29]	= data[.,6]:*data[.,23]
	data[.,30]	= data[.,6]:*data[.,24]
	data[.,33]	= data[.,4]:*data[.,31]
	data[.,34]	= data[.,4]:*data[.,32]
	data[.,35]	= data[.,5]:*data[.,31]
	data[.,36]	= data[.,5]:*data[.,32]
	data[.,37]	= data[.,6]:*data[.,31]
	data[.,38]	= data[.,6]:*data[.,32]
											 
	j = j+1
}

st_store(.,"Q1",Q1)
end
drop _I*

* The following calcualtes s_hat 
gen s_hat = .

mata
s_hat	= J(rows(data1),1,0)
i=1
j=0

/* Note that the timing has to match the timing in the equation - q is at t-1*/
l1_data1=data1
l1_data1[.,1]	= data1[.,1] - ones
l1_data1[.,2]	= l1_data1[.,1]:^2
l1_data1[.,3]	= l1_data1[.,1]:^3
l1_data1[.,6]	= l1_data1[.,1]:*l1_data1[.,4]
l1_data1[.,7]	= l1_data1[.,1]:*l1_data1[.,5]
l1_data1[.,8]	= l1_data1[.,2]:*l1_data1[.,4]
l1_data1[.,9]	= l1_data1[.,2]:*l1_data1[.,5]
l1_data1[.,10]	= l1_data1[.,3]:*l1_data1[.,4]
l1_data1[.,11]	= l1_data1[.,3]:*l1_data1[.,5]
l1_data1[.,14]	= l1_data1[.,1]:*l1_data1[.,12]
l1_data1[.,15]	= l1_data1[.,1]:*l1_data1[.,13]
l1_data1[.,16]	= l1_data1[.,2]:*l1_data1[.,12]
l1_data1[.,17]	= l1_data1[.,2]:*l1_data1[.,13]
l1_data1[.,18]	= l1_data1[.,3]:*l1_data1[.,12]
l1_data1[.,19]	= l1_data1[.,3]:*l1_data1[.,13]

l1_data2=data2
l1_data2[.,1]	= l1_data2[.,1] - ones
l1_data2[.,2]	= l1_data2[.,1]:^2
l1_data2[.,3]	= l1_data2[.,1]:^3
l1_data2[.,6]	= l1_data2[.,1]:*l1_data2[.,4]
l1_data2[.,7]	= l1_data2[.,1]:*l1_data2[.,5]
l1_data2[.,8]	= l1_data2[.,2]:*l1_data2[.,4]
l1_data2[.,9]	= l1_data2[.,2]:*l1_data2[.,5]
l1_data2[.,10]	= l1_data2[.,3]:*l1_data2[.,4]
l1_data2[.,11]	= l1_data2[.,3]:*l1_data2[.,5]
l1_data2[.,14]	= l1_data2[.,1]:*l1_data2[.,12]
l1_data2[.,15]	= l1_data2[.,1]:*l1_data2[.,13]
l1_data2[.,16]	= l1_data2[.,2]:*l1_data2[.,12]
l1_data2[.,17]	= l1_data2[.,2]:*l1_data2[.,13]
l1_data2[.,18]	= l1_data2[.,3]:*l1_data2[.,12]
l1_data2[.,19]	= l1_data2[.,3]:*l1_data2[.,13]

while (i>0) {
	index	= data1[.,1]:<T+1				 
	i		= max(index)
	P		= normal(l1_data2*beta2p')
	s_hat 		= s_hat + exp((data3*beta3') - j*log(1+r)*ones):*(exp(l1_data1*beta1')):/(exp(l1_data1*beta1')+ exp(l1_data2*beta2'):*P):*index
	
	l1_data1=data1
	l1_data2=data2
	
	data1[.,1]	= data1[.,1] + ones
	data1[.,2]	= data1[.,1]:^2
	data1[.,3]	= data1[.,1]:^3
	data1[.,6]	= data1[.,1]:*data1[.,4]
	data1[.,7]	= data1[.,1]:*data1[.,5]
	data1[.,8]	= data1[.,2]:*data1[.,4]
	data1[.,9]	= data1[.,2]:*data1[.,5]
	data1[.,10]	= data1[.,3]:*data1[.,4]
	data1[.,11]	= data1[.,3]:*data1[.,5]
	data1[.,14]	= data1[.,1]:*data1[.,12]
	data1[.,15]	= data1[.,1]:*data1[.,13]
	data1[.,16]	= data1[.,2]:*data1[.,12]
	data1[.,17]	= data1[.,2]:*data1[.,13]
	data1[.,18]	= data1[.,3]:*data1[.,12]
	data1[.,19]	= data1[.,3]:*data1[.,13]

	data2[.,1]	= data2[.,1] + ones
	data2[.,2]	= data2[.,1]:^2
	data2[.,3]	= data2[.,1]:^3
	data2[.,6]	= data2[.,1]:*data2[.,4]
	data2[.,7]	= data2[.,1]:*data2[.,5]
	data2[.,8]	= data2[.,2]:*data2[.,4]
	data2[.,9]	= data2[.,2]:*data2[.,5]
	data2[.,10]	= data2[.,3]:*data2[.,4]
	data2[.,11]	= data2[.,3]:*data2[.,5]
	data2[.,14]	= data2[.,1]:*data2[.,12]
	data2[.,15]	= data2[.,1]:*data2[.,13]
	data2[.,16]	= data2[.,2]:*data2[.,12]
	data2[.,17]	= data2[.,2]:*data2[.,13]
	data2[.,18]	= data2[.,3]:*data2[.,12]
	data2[.,19]	= data2[.,3]:*data2[.,13]

	data3[.,1]	= data3[.,1] + ones
	data3[.,2]	= data3[.,1]:^2
	data3[.,3]	= data3[.,1]:^3
	data3[.,4]	= data3[.,4] + ones
	data3[.,5]	= data3[.,4]:^2
	data3[.,6]	= data3[.,4]:^3
	data3[.,9]	= data3[.,1]:*data3[.,7]
	data3[.,10]	= data3[.,1]:*data3[.,8]
	data3[.,11]	= data3[.,2]:*data3[.,7]
	data3[.,12]	= data3[.,2]:*data3[.,8]
	data3[.,13]	= data3[.,3]:*data3[.,7]
	data3[.,14]	= data3[.,3]:*data3[.,8]
	data3[.,17]	= data3[.,1]:*data3[.,15]
	data3[.,18]	= data3[.,1]:*data3[.,16]
	data3[.,19]	= data3[.,2]:*data3[.,15]
	data3[.,20]	= data3[.,2]:*data3[.,16]
	data3[.,21]	= data3[.,3]:*data3[.,15]
	data3[.,22]	= data3[.,3]:*data3[.,16]
	data3[.,25]	= data3[.,4]:*data3[.,23]
	data3[.,26]	= data3[.,4]:*data3[.,24]
	data3[.,27]	= data3[.,5]:*data3[.,23]
	data3[.,28]	= data3[.,5]:*data3[.,24]
	data3[.,29]	= data3[.,6]:*data3[.,23]
	data3[.,30]	= data3[.,6]:*data3[.,24]
	data3[.,33]	= data3[.,4]:*data3[.,31]
	data3[.,34]	= data3[.,4]:*data3[.,32]
	data3[.,35]	= data3[.,5]:*data3[.,31]
	data3[.,36]	= data3[.,5]:*data3[.,32]
	data3[.,37]	= data3[.,6]:*data3[.,31]
	data3[.,38]	= data3[.,6]:*data3[.,32]


	j = j+1
}

st_store(.,"s_hat",s_hat)
end

replace s_hat=s_hat/Q1
   
*========================================================
* Define pi and q explicitly (s_hat already defined above) 
*========================================================
foreach var of varlist tot_assets* {
	replace `var'=l2_`var'
}

egen med_assets = median(tot_assets3), by(age educ)

gen D3 	= tot_assets3
gen pi_tax  = 1-(D3/(Q1+D3))

*=====================================================================================
* Remove very extreme pi_tax (very few) due to extreme negative debt with low earnings  
*=====================================================================================
replace pi_tax = . if pi_tax<0 | (pi_tax>2 & pi_tax!=.)

gen q=ly/(ly + wly)	
gen l2_q=l2.q		


