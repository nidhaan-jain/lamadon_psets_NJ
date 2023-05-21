/* 	Calcualte residual measures for three specifications: 
	(1) participation=0 and total consumption: Most specifications in the paper.
	(2) participation=1 and total consumption: Robustness with participation correction shut down (col 4 Table 6).
	(2) participation=0 and consumption w.o. housing: Robustness with participation correction shut down (col 5 Table 6). 
*/
clear all
cd "$output"
set matsize 11000

local spec=1
while `spec'<=3	{
qui {
	if `spec'==1 {
		global participation = 0
		global consumption 	=  5
	}
	if `spec'==2 {
		global participation = 1
		global consumption 	=  5
	}
	if `spec'==3 {
		global participation = 0
		global consumption 	=  6
	}

	*===========================
	* Regressions for residuals 
	*===========================
	u data4estimation, replace

	if ${consumption}==6 {
		replace log_c=log_c_nh
	}
	tsset person year
	tab year,gen(yrd)

	/* head */ 
	tab educ, gen(edd)
	tab weduc, gen(wedd)
	gen empl  	= empst==1
	gen w_empl  = wempst==1
	gen unempl	= empst==2
	gen w_unempl= wempst==2
	gen retir 	= empst==3
	gen w_retir = wempst==3
	gen white 	= race==1
	gen w_white	= wrace==1 
	gen black 	= race==2
	gen w_black	= wrace==2 
	gen other 	= race>=3
	gen w_other = wrace>=3
	tab state,gen(stated)
	gen kidsout	= outkid==1
	gen bigcity	= smsa==1|smsa==2
	tab yb,gen(ybd)
	tab wyb,gen(w_ybd)
	tab fsize,gen(fd)
	tab kids,gen(chd)
	replace smsa=6 if smsa>6
	gen extra=(tyoth)>0

	gen dlog_c=log_c-l2.log_c
	gen dlog_w=log_w-l2.log_w
	gen dlog_ww=log_ww-l2.log_ww
	gen dlog_y=log_y-l2.log_y
	gen dlog_yw=log_yw-l2.log_yw
	gen dlog_toty=log_toty-l2.log_toty

	gen log_h = log(hours)		
	gen log_hw = log(hourw)

	foreach var of varlist kidsout bigcity kids fsize empl w_empl unempl w_unempl retir w_retir extra {
		gen d`var'=`var'-l2.`var'
	}

	tab dkids, gen(dchd) 
	tab dfsize, gen(dfd)

	/* consumption regression */ 

	#delimit;
	xi	i.educ*i.year i.weduc*i.year i.white*i.year i.w_white*i.year i.black*i.year i.w_black*i.year 
		i.other*i.year i.w_other*i.year i.bigcity*i.year i.dbigcity*i.year 
		i.empl*i.year i.unempl*i.year i.retir*i.year i.w_empl*i.year 
		i.w_unempl*i.year i.w_retir*i.year i.dempl*i.year i.dunempl*i.year i.dretir*i.year 
		i.dw_empl*i.year i.dw_unempl*i.year i.dw_retir*i.year
		i.state*i.year i.mort1_dum*i.year i.mort2_dum*i.year;
	reg dlog_c   	yrd* ybd* w_ybd* edd* wedd* white w_white black w_black other w_other 
					fd* chd* empl  unempl retir w_empl  w_unempl w_retir kidsout bigcity extra 
					dfd* dchd* dempl dunempl dretir dw_empl dw_unempl dw_retir dextra dkidsout dbigcity dextra 
					stated* 
					_IeduXyea_* _IwedXyea_* _IwhiXyea_* _Iw_wXyea_* _IblaXyea_* _Iw_bXyea_*
					_IothXyea_* _Iw_other_* 
					_IempXyea_* _Iw_empl_* _IuneXyea_* _Iw_unempl_* _IretXyea_* _Iw_retir_*
					_IbigXyea_* _IdbiXyea_*;
	predict uc if e(sample),res;
	#delimit cr

	/* Earnings regression head */ 
	#delimit;
	reg dlog_y   	yrd* ybd* w_ybd* edd* wedd* white w_white black w_black other w_other 
					fd* chd* empl  unempl retir w_empl  w_unempl w_retir kidsout bigcity extra 
					dfd* dchd* dempl dunempl dretir dw_empl dw_unempl dw_retir dextra dkidsout dbigcity dextra 
					stated* 
					_IeduXyea_* _IwedXyea_* _IwhiXyea_* _Iw_wXyea_* _IblaXyea_* _Iw_bXyea_*
					_IothXyea_* _Iw_other_* 
					_IempXyea_* _Iw_empl_* _IuneXyea_* _Iw_unempl_* _IretXyea_* _Iw_retir_*
					_IbigXyea_* _IdbiXyea_*;
	predict uy if e(sample),res;
	#delimit cr

	/* Earnings regression Total household earnings */ 
	#delimit;
	reg dlog_toty   yrd* ybd* w_ybd* edd* wedd* white w_white black w_black other w_other 
					fd* chd* empl  unempl retir w_empl  w_unempl w_retir kidsout bigcity extra 
					dfd* dchd* dempl dunempl dretir dw_empl dw_unempl dw_retir dextra dkidsout dbigcity dextra 
					stated* 
					_IeduXyea_* _IwedXyea_* _IwhiXyea_* _Iw_wXyea_* _IblaXyea_* _Iw_bXyea_*
					_IothXyea_* _Iw_other_* 
					_IempXyea_* _Iw_empl_* _IuneXyea_* _Iw_unempl_* _IretXyea_* _Iw_retir_*
					_IbigXyea_* _IdbiXyea_*;
	predict utoty if e(sample),res;
	#delimit cr

	/* Wage regression head  */ 
	#delimit;
	reg dlog_w	    yrd* ybd* edd*  white black other stated* bigcity dbigcity
					_IeduXyea_* _IwhiXyea_* _IblaXyea_* _IothXyea_* _IbigXyea_* _IdbiXyea_*;
	predict uw if e(sample),res;
	#delimit cr

	/* selection equation for the wife */
	#delimit;
	probit wife_employed   	yrd* ybd* w_ybd* edd* wedd* white w_white black w_black other w_other 
					fd* chd* empl  unempl retir w_empl  w_unempl w_retir kidsout bigcity extra 
					stated* 
					_IeduXyea_* _IwedXyea_* _IwhiXyea_* _Iw_wXyea_* _IblaXyea_* _Iw_bXyea_*
					_IothXyea_* _Iw_other_* 
					_IempXyea_* _Iw_empl_* _IuneXyea_* _Iw_unempl_* _IretXyea_* _Iw_retir_*
					_Imor*;
	#delimit cr
	testparm _Imor*

	predict z_eta if e(sample), xb
	gen pdf = normalden(z_eta)
	gen cdf = normal(z_eta)
	gen inv_mills = pdf/cdf
	drop pdf cdf

	if ${participation}==0 {
		replace inv_mills=0		/* This would imply that it is omitted from the regression in what follows */ 
		replace z_eta = 0
	}

	/* Earnings regression wife */ 
	gen dinv_mills=inv_mills-l2.inv_mills

	#delimit;
	reg dlog_yw   	yrd* ybd* w_ybd* edd* wedd* white w_white black w_black other w_other 
					fd* chd* empl  unempl retir w_empl  w_unempl w_retir kidsout bigcity extra 
					dfd* dchd* dempl dunempl dretir dw_empl dw_unempl dw_retir dextra dkidsout dbigcity dextra 
					stated* 
					_IeduXyea_* _IwedXyea_* _IwhiXyea_* _Iw_wXyea_* _IblaXyea_* _Iw_bXyea_*
					_IothXyea_* _Iw_other_* 
					_IempXyea_* _Iw_empl_* _IuneXyea_* _Iw_unempl_* _IretXyea_* _Iw_retir_*
					_IbigXyea_* _IdbiXyea_*
					dinv_mills;
	local gamma=_b[dinv_mills];
	predict uyw_temp if e(sample),res;
	gen uyw = uyw_temp + `gamma'*dinv_mills;
	drop uyw_temp;
	#delimit cr

	/* Wage regression wife  */ 
	#delimit;
	reg dlog_ww 	yrd* w_ybd* wedd*  w_white w_black w_other stated* bigcity dbigcity
					_IwedXyea_* _Iw_wXyea_* _Iw_bXyea_* _Iw_oXyea_* _IbigXyea_* _IdbiXyea_*   
					dinv_mills;
	local gamma=_b[dinv_mills];
	predict uww_temp if e(sample),res;
	gen uww = uww_temp + `gamma'*dinv_mills;
	drop uww_temp;
	#delimit cr

	drop _I* 
	
	save data3_resid_cons${consumption}_part${participation}, replace 

	*=================================
	* Generate moments for estimation
	*=================================
	gen duc=uc
	gen duy=uy
	gen duyw=uyw
	gen duw=uw
	gen duww=uww
	gen dutoty=utoty

	gen l2_z_eta 	= l2.z_eta
	gen l2_inv_mills= l2.inv_mills
	gen l4_inv_mills= l4.inv_mills

	*** For conditional means also need the difference of mills ratio
	gen d_inv_mills = inv_mills-l2.inv_mills

	tsset person year

	*** Age categories 
	gen age_cat=1
	replace age_cat=2 if age>=38
	replace age_cat=3 if age>=43
	replace age_cat=4 if age>=48
	replace age_cat=5 if age>=53

	tab age_cat, gen(age_cat)

	gen agew_cat=1
	replace agew_cat=2 if agew>=38
	replace agew_cat=3 if agew>=43
	replace agew_cat=4 if agew>=48
	replace agew_cat=5 if agew>=53

	tab agew_cat, gen(agew_cat)

	*** Prepare all the cross residuals for the cross moments 
	* own and cross moments
	gen duc2 		= duc^2
	gen duw2		= duw^2
	gen duww2		= duww^2
	gen duy2 		= duy^2
	gen duyw2 		= duyw^2

	gen duy_lag		= duy*l2.duy
	gen duyw_lag	= duyw*l2.duyw
	gen duw_lag		= duw*l2.duw
	gen duww_lag	= duww*l2.duww

	gen duw_duc		= duw*duc
	gen duw_duy		= duw*duy
	gen duw_duyw	= duw*duyw
	gen duww_duc	= duww*duc
	gen duww_duy	= duww*duy
	gen duww_duyw	= duww*duyw
	gen duc_duy		= duc*duy
	gen duc_duyw	= duc*duyw
	gen duy_duyw	= duy*duyw

	gen duw_duww	= duw*duww		
	gen duw_duww_lag= duw*l2.duww
	gen duww_duw_lag= l2.duw*duww							
		
	gen duc_lag		= duc*l2.duc 	

	gen duw_duc_lag = duw*l2.duc	
	gen duc_duw_lag = duc*l2.duw	
	gen duy_duc_lag = duy*l2.duc
	gen duc_duy_lag = duc*l2.duy
	gen duww_duc_lag = duww*l2.duc	
	gen duc_duww_lag = duc*l2.duww	
	gen duyw_duc_lag = duyw*l2.duc
	gen duc_duyw_lag = duc*l2.duyw

	gen duw_duy_lag = duw*l2.duy	
	gen duy_duw_lag = duy*l2.duw	
	gen duw_duyw_lag = duw*l2.duyw	
	gen duyw_duw_lag = duyw*l2.duw	
	gen duww_duy_lag = duww*l2.duy	
	gen duy_duww_lag = duy*l2.duww	
	gen duww_duyw_lag = duww*l2.duyw	
	gen duyw_duww_lag = duyw*l2.duww	

	gen dutoty2 	= dutoty^2
	sort person year

	save estimation_input_all${consumption}_rhs1_part${participation}, replace 
	local spec=`spec'+1
}
}
cd "$do"
