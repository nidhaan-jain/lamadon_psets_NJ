/* 	This file generates Figure 8 that compares moments in the data and in the model. 
	It also generates the appendix table that has these moments as well as the moments
	for hours. Note that moments for hours are not targeted, and that we need to first 
	calculate residuals for these moments. 
	*/ 

clear all
cd "$output"

*==============
* Prepare data 
*==============
u "$directory/data_het12pi1fv10cons5_rep0.dta", clear
foreach var in duw2 duw_lag duww2 duww_lag duw_duww duw_duww_lag duww_duw_lag ///
				duc2 duy2 duyw2 duy_lag duyw_lag duw_duc duw_duy duw_duyw duww_duc ///
				duww_duy duww_duyw duc_duy duc_duyw duy_duyw duc_lag duw_duc_lag duww_duc_lag /// 
				duc_duw_lag duc_duww_lag duy_duc_lag duyw_duc_lag duc_duy_lag duc_duyw_lag ///
				duw_duy_lag duww_duyw_lag duy_duw_lag duyw_duww_lag duw_duyw_lag duww_duy_lag /// 
				duyw_duw_lag duy_duww_lag {
				
				reg `var'
				gen `var'_SE = _se[_cons]	/*(note: previous versions had bootstrapped s.e., which deliver very similar results)*/
		}

keep ss* ts* du*

gen duw2_model   		= 	  	duw2 			- ss_resid1
gen duw_lag_model    	=    	duw_lag			- ss_resid2
gen duww2_model			= 		duww2		 	- ss_resid3
gen duww_lag_model	 	=		duww_lag	 	- ss_resid4
gen duw_duww_model	 	=		duw_duww		- ss_resid5
gen duw_duww_lag_model 	=		duw_duww_lag	- ss_resid6
gen duww_duw_lag_model 	=		duww_duw_lag	- ss_resid6

gen duc2_model = 		  duc2 			- ts_resid1
gen duy2_model = 	      duy2 	        - ts_resid2 
gen duyw2_model = 	      duyw2 	        - ts_resid3 
gen duy_lag_model =       duy_lag         - ts_resid4 
gen duyw_lag_model =      duyw_lag        - ts_resid5 
gen duw_duc_model =       duw_duc         - ts_resid6 
gen duw_duy_model =       duw_duy         - ts_resid7 
gen duw_duyw_model =      duw_duyw        - ts_resid8 
gen duww_duc_model =      duww_duc        - ts_resid9 
gen  duww_duy_model =      duww_duy       - ts_resid10
gen  duww_duyw_model = 	   duww_duyw 	    - ts_resid11
gen  duc_duy_model = 	   duc_duy 	    - ts_resid12
gen  duc_duyw_model = 	   duc_duyw 	    - ts_resid13
gen  duy_duyw_model = 	   duy_duyw 	    - ts_resid14
gen  duc_lag_model = 	   duc_lag 	    - ts_resid15
gen  duw_duc_lag_model =   duw_duc_lag    - ts_resid16
gen  duww_duc_lag_model =  duww_duc_lag   - ts_resid17
gen  duc_duw_lag_model =   duc_duw_lag    - ts_resid18
gen  duc_duww_lag_model =  duc_duww_lag   - ts_resid19
gen  duy_duc_lag_model =   duy_duc_lag    - ts_resid20
gen  duyw_duc_lag_model =  duyw_duc_lag   - ts_resid21
gen  duc_duy_lag_model =   duc_duy_lag    - ts_resid22
gen  duc_duyw_lag_model =  duc_duyw_lag   - ts_resid23
gen  duw_duy_lag_model =   duw_duy_lag    - ts_resid24
gen  duww_duyw_lag_model = duww_duyw_lag  - ts_resid25
gen  duy_duw_lag_model =   duy_duw_lag    - ts_resid26
gen  duyw_duww_lag_model = duyw_duww_lag  - ts_resid27
gen  duw_duyw_lag_model =  duw_duyw_lag   - ts_resid28
gen  duww_duy_lag_model =  duww_duy_lag   - ts_resid29
gen  duyw_duw_lag_model =  duyw_duw_lag   - ts_resid30
gen  duy_duww_lag_model =  duy_duww_lag   - ts_resid31

keep 			duw2 duw_lag duww2 duww_lag duw_duww duw_duww_lag duww_duw_lag ///
				duc2 duy2 duyw2 duy_lag duyw_lag duw_duc duw_duy duw_duyw duww_duc ///
				duww_duy duww_duyw duc_duy duc_duyw duy_duyw duc_lag duw_duc_lag duww_duc_lag /// 
				duc_duw_lag duc_duww_lag duy_duc_lag duyw_duc_lag duc_duy_lag duc_duyw_lag ///
				duw_duy_lag duww_duyw_lag duy_duw_lag duyw_duww_lag duw_duyw_lag duww_duy_lag /// 
				duyw_duw_lag duy_duww_lag *_SE *_model
collapse *

preserve 
	keep *_model
	xpose, clear varname
	ren v1 model
	ren _ varname
	sort varname
	save model, replace
restore 

preserve 
	keep *_SE
	xpose, clear varname
	ren v1 data_se
	ren _ varname
	replace varname = reverse(substr(reverse(varname),4,.))
	sort varname
	save SE, replace
restore 

drop *_model *_SE
foreach var of varlist * {
	ren `var' `var'_model
}
xpose, clear varname
ren v1 data
ren _ varname
sort varname

merge 1:1 varname using model
drop _m
replace varname = reverse(substr(reverse(varname),7,.))
merge 1:1 varname using SE
drop _m

*==========
* Figure 8
*==========
gen group = 1
replace group = 2 if inlist(varname,"duw_duc","duww_duc","duc_duy","duc_duyw")
replace group = 3 if inlist(varname,"duw_duy","duww_duyw","duw_duyw","duww_duy","duy_duyw")
replace group = 4 if inlist(varname,"duc_lag","duc_duw_lag","duc_duww_lag","duc_duy_lag","duc_duyw_lag","duw_duc_lag","duww_duc_lag","duy_duc_lag","duyw_duc_lag")
replace group = 5 if inlist(varname,"duy_lag","duyw_lag","duw_duy_lag","duww_duyw_lag","duy_duw_lag","duyw_duww_lag","duw_duyw_lag")
replace group = 5 if inlist(varname,"duww_duy_lag","duyw_duw_lag","duy_duww_lag")
replace group = 6 if inlist(varname,"duw2", "duww2")
replace group = 7 if inlist(varname,"duw_lag", "duww_lag", "duw_duww", "duw_duww_lag", "duww_duw_lag")

gen order=.
replace order = 1  	 if varname ==  "duw2" 	        
replace order = 2 	 if varname ==  "duww2" 	        
replace order = 4    if varname ==  "duw_lag"       
replace order = 5    if varname ==  "duww_lag"      
replace order = 3     if varname ==  "duw_duww"      
replace order = 7    if varname ==  "duww_duw_lag" 
replace order = 6    if varname ==  "duw_duww_lag" 
replace order = 8	 if varname ==  "duc2" 	        
replace order = 9		if varname ==  "duy2" 	        
replace order = 10	if varname ==  "duyw2" 	    
replace order = 29         if varname ==  "duy_lag"       
replace order = 30        if varname ==  "duyw_lag"      
replace order = 11      if varname ==  "duw_duc"       
replace order = 15      if varname ==  "duw_duy"       
replace order = 16     if varname ==  "duw_duyw"      
replace order = 12    if varname ==  "duww_duc"      
replace order = 17     if varname ==  "duww_duy"     
replace order = 18	   if varname ==  "duww_duyw" 	
replace order = 13     if varname ==  "duc_duy" 	    
replace order = 14 if varname ==  "duc_duyw" 	
replace order = 19	   if varname ==  "duy_duyw" 	
replace order = 20       if varname ==  "duc_lag" 	    
replace order = 21  if varname ==  "duw_duc_lag"  
replace order = 22 if varname ==  "duww_duc_lag" 
replace order = 23  if varname ==  "duc_duw_lag"  
replace order = 24 if varname ==  "duc_duww_lag" 
replace order = 25  if varname ==  "duy_duc_lag"  
replace order = 26 if varname ==  "duyw_duc_lag" 
replace order = 27  if varname ==  "duc_duy_lag"  
replace order = 28  if varname ==  "duc_duyw_lag" 
replace order = 31    if varname ==  "duw_duy_lag"  
replace order = 32  if varname ==  "duww_duyw_lag"
replace order = 33    if varname ==  "duy_duw_lag"  
replace order = 34  if varname ==  "duyw_duww_lag"
replace order = 35   if varname ==  "duw_duyw_lag" 
replace order = 36   if varname ==  "duww_duy_lag" 
replace order = 37   if varname ==  "duyw_duw_lag" 
replace order = 38   if varname ==  "duy_duww_lag" 

gen varname0=varname

replace varname =  "Var({&Delta}w{subscript:1,t})" 	   if varname ==  "duw2" 	        
replace varname =  "Var({&Delta}w{subscript:2,t})" 	   if varname ==  "duww2" 	        
replace varname =  "Cov({&Delta}w{subscript:1,t},{&Delta}w{subscript:1,t-2})"         if varname ==  "duw_lag"       
replace varname =  "Cov({&Delta}w{subscript:2,t},{&Delta}w{subscript:2,t-2})"        if varname ==  "duww_lag"      
replace varname =  "Cov({&Delta}w{subscript:1,t},{&Delta}w{subscript:2,t})"        if varname ==  "duw_duww"      
replace varname =  "Cov({&Delta}w{subscript:2,t},{&Delta}w{subscript:1,t-2})"    if varname ==  "duww_duw_lag" 
replace varname =  "Cov({&Delta}w{subscript:1,t},{&Delta}w{subscript:2,t-2})"    if varname ==  "duw_duww_lag" 

replace varname =  "Var({&Delta}c{subscript:t})" 	   if varname ==  "duc2" 	        
replace varname =  "Var({&Delta}y{subscript:1,t})" 	       if varname ==  "duy2" 	        
replace varname =  "Var({&Delta}y{subscript:2,t})" 	       if varname ==  "duyw2" 	    
replace varname =  "Cov({&Delta}y{subscript:1,t},{&Delta}y{subscript:1,t-2})"         if varname ==  "duy_lag"       
replace varname =  "Cov({&Delta}y{subscript:2,t},{&Delta}y{subscript:2,t-2})"        if varname ==  "duyw_lag"      
replace varname =  "Cov({&Delta}w{subscript:1,t},{&Delta}c{subscript:t})"         if varname ==  "duw_duc"       
replace varname =  "Cov({&Delta}w{subscript:1,t},{&Delta}y{subscript:1,t})"         if varname ==  "duw_duy"       
replace varname =  "Cov({&Delta}w{subscript:1,t},{&Delta}y{subscript:2,t})"        if varname ==  "duw_duyw"      
replace varname =  "Cov({&Delta}w{subscript:2,t},{&Delta}c{subscript:t})"        if varname ==  "duww_duc"      
replace varname =  "Cov({&Delta}w{subscript:2,t},{&Delta}y{subscript:1,t})"        if varname ==  "duww_duy"     
replace varname =  "Cov({&Delta}w{subscript:2,t},{&Delta}y{subscript:2,t})" 	   if varname ==  "duww_duyw" 	
replace varname =  "Cov({&Delta}y{subscript:1,t},{&Delta}c{subscript:t})" 	       if varname ==  "duc_duy" 	    
replace varname =  "Cov({&Delta}y{subscript:2,t},{&Delta}c{subscript:t})" 	   if varname ==  "duc_duyw" 	
replace varname =  "Cov({&Delta}y{subscript:1,t},{&Delta}y{subscript:2,t})" 	   if varname ==  "duy_duyw" 	
replace varname =  "Cov({&Delta}c{subscript:t},{&Delta}c{subscript:t-2})" 	       if varname ==  "duc_lag" 	    
replace varname =  "Cov({&Delta}w{subscript:1,t},{&Delta}c{subscript:t-2})"     if varname ==  "duw_duc_lag"  
replace varname =  "Cov({&Delta}w{subscript:2,t},{&Delta}c{subscript:t-2})"    if varname ==  "duww_duc_lag" 
replace varname =  "Cov({&Delta}c{subscript:t},{&Delta}w{subscript:1,t-2})"     if varname ==  "duc_duw_lag"  
replace varname =  "Cov({&Delta}c{subscript:t},{&Delta}w{subscript:2,t-2})"    if varname ==  "duc_duww_lag" 
replace varname =  "Cov({&Delta}y{subscript:1,t},{&Delta}c{subscript:t-2})"     if varname ==  "duy_duc_lag"  
replace varname =  "Cov({&Delta}y{subscript:2,t},{&Delta}c{subscript:t-2})"    if varname ==  "duyw_duc_lag" 
replace varname =  "Cov({&Delta}c{subscript:t},{&Delta}y{subscript:1,t-2})"   if varname ==  "duc_duy_lag"  
replace varname =  "Cov({&Delta}c{subscript:t},{&Delta}y{subscript:2,t-2})"   if varname ==  "duc_duyw_lag" 
replace varname =  "Cov({&Delta}w{subscript:1,t},{&Delta}y{subscript:1,t-2})"     if varname ==  "duw_duy_lag"  
replace varname =  "Cov({&Delta}w{subscript:2,t},{&Delta}y{subscript:2,t-2})"   if varname ==  "duww_duyw_lag"
replace varname =  "Cov({&Delta}y{subscript:1,t},{&Delta}w{subscript:1,t-2})"     if varname ==  "duy_duw_lag"  
replace varname =  "Cov({&Delta}y{subscript:2,t},{&Delta}w{subscript:2,t-2})"   if varname ==  "duyw_duww_lag"
replace varname =  "Cov({&Delta}w{subscript:1,t},{&Delta}y{subscript:2,t-2})"    if varname ==  "duw_duyw_lag" 
replace varname =  "Cov({&Delta}w{subscript:2,t},{&Delta}y{subscript:1,t-2})"    if varname ==  "duww_duy_lag" 
replace varname =  "Cov({&Delta}y{subscript:2,t},{&Delta}w{subscript:1,t-2})"    if varname ==  "duyw_duw_lag" 
replace varname =  "Cov({&Delta}y{subscript:1,t},{&Delta}w{subscript:2,t-2})"    if varname ==  "duy_duww_lag" 

save model_vs_data_SE, replace 

gen data_05 = data-1.96*data_se 
gen data_95 = data+1.96*data_se 
ren data data_est
ren model model_est

#delimit;
graph hbar data_est model_est if group==6 | group==7, over(varname, sort(order)) bar(1, color(black)) bar(2, color(black) fintensity( inten20))
	  legend(label(1 "Data") label(2 "Model"))
	  graphregion(color(white))
	  name(g6, replace);
#delimit cr

#delimit;
graph hbar data_est model_est if group==1 , over(varname, sort(order)) bar(1, color(black)) bar(2, color(black) fintensity( inten20))
	  legend(label(1 "Data") label(2 "Model"))
	  graphregion(color(white))
	  name(g1, replace);
#delimit cr

#delimit;
graph hbar data_est model_est if group==2 | group==3 , over(varname, sort(order)) bar(1, color(black)) bar(2, color(black) fintensity( inten20))
	  legend(label(1 "Data") label(2 "Model"))
	  graphregion(color(white))
	  name(g2, replace);
#delimit cr

#delimit;
graph hbar data_est model_est if group==4, over(varname, sort(order)) bar(1, color(black)) bar(2, color(black) fintensity( inten20))
	  legend(label(1 "Data") label(2 "Model"))
	  graphregion(color(white))
	  name(g4, replace);
#delimit cr

#delimit;
graph hbar data_est model_est if group==5, over(varname, sort(order)) ylabel(-0.12(.04)0) bar(1, color(black)) bar(2, color(black) fintensity( inten20))
	  legend(label(1 "Data") label(2 "Model"))
	  graphregion(color(white))
	  name(g5, replace);
#delimit cr
window manage close graph _all

grc1leg g6 g1 g2 g4 g5, graphregion(color(white)) rows(2) holes(4) 

graph export  "$graphs/Figure8_fit_internal.png", replace as(png) width(1600)
graph export  "$graphs/Figure8_fit_internal.eps", replace as(eps) 
graph save "$graphs/Figure8_fit_internal.gph", replace  
window manage close graph _all


*================================================
* Appendix moments table: includes hours and s.e.
*================================================
*** Prepare data for hours (need to calculate residuals)
u "$directory/data_het12pi1fv10cons5_rep0", clear
cap drop __*
gen dlog_h  = log_h-l2.log_h
gen dlog_hw = log_hw-l2.log_hw

* Prepare residuals in a similar way to earnings (no participation correction option)
#delimit;
xi	i.educ*i.year i.weduc*i.year i.white*i.year i.w_white*i.year i.black*i.year i.w_black*i.year 
	i.other*i.year i.w_other*i.year i.bigcity*i.year i.dbigcity*i.year 
	i.empl*i.year i.unempl*i.year i.retir*i.year i.w_empl*i.year 
	i.w_unempl*i.year i.w_retir*i.year i.dempl*i.year i.dunempl*i.year i.dretir*i.year 
	i.dw_empl*i.year i.dw_unempl*i.year i.dw_retir*i.year
	i.state*i.year i.mort1_dum*i.year i.mort2_dum*i.year;
#delimit cr

#delimit;
reg dlog_h   	yrd* ybd* w_ybd* edd* wedd* white w_white black w_black other w_other 
				fd* chd* empl  unempl retir w_empl  w_unempl w_retir kidsout bigcity extra 
				dfd* dchd* dempl dunempl dretir dw_empl dw_unempl dw_retir dextra dkidsout dbigcity dextra 
				stated* 
				_IeduXyea_* _IwedXyea_* _IwhiXyea_* _Iw_wXyea_* _IblaXyea_* _Iw_bXyea_*
				_IothXyea_* _Iw_other_* 
				_IempXyea_* _Iw_empl_* _IuneXyea_* _Iw_unempl_* _IretXyea_* _Iw_retir_*
				_IbigXyea_* _IdbiXyea_*;
predict uh if e(sample),res;
#delimit cr

#delimit;
reg dlog_hw   	yrd* ybd* w_ybd* edd* wedd* white w_white black w_black other w_other 
				fd* chd* empl  unempl retir w_empl  w_unempl w_retir kidsout bigcity extra 
				dfd* dchd* dempl dunempl dretir dw_empl dw_unempl dw_retir dextra dkidsout dbigcity dextra 
				stated* 
				_IeduXyea_* _IwedXyea_* _IwhiXyea_* _Iw_wXyea_* _IblaXyea_* _Iw_bXyea_*
				_IothXyea_* _Iw_other_* 
				_IempXyea_* _Iw_empl_* _IuneXyea_* _Iw_unempl_* _IretXyea_* _Iw_retir_*
				_IbigXyea_* _IdbiXyea_*;
predict uhw if e(sample),res;
#delimit cr

replace dlog_h=uh
replace dlog_hw=uhw

gen duh2 = dlog_h*dlog_h
gen duh_lag = dlog_h*l2.dlog_h
gen duhw2 = dlog_hw*dlog_hw
gen duhw_lag = dlog_hw*l2.dlog_hw
gen duh_duhw = dlog_h*dlog_hw

foreach var in duh2 duh_lag duhw2 duhw_lag duh_duhw {
				reg `var'
				gen `var'_SE = _se[_cons]
}

foreach var in k5 k7 k10 k12 {
	replace `var'=`var'-1
}

#delimit;
gen duh2_model  = 2*((k5)^2*m_s_u1 + (k6)^2*m_s_u2
				+ (k7)^2*m_s_v1 + (k8)^2*m_s_v2
				+ k5*k6*(2*m_r_u1u2)
				+ k7*k8*(2*m_r_v1v2)
				+ s_me_h);
		
gen duhw2_model    = 2*((k9)^2*m_s_u1 + (k10)^2*m_s_u2
						+ (k11)^2*m_s_v1 + (k12)^2*m_s_v2
						+ 2*(k9)*(k10)*m_r_u1u2
						+ 2*(k11)*(k12)*m_r_v1v2
						+ s_me_hw_cons);

gen duh_lag_model  = (-(k5)^2*(l2_s_u1) -(k6)^2*(l2_s_u2)
						- 2*(k5)*(k6)*l2_r_u1u2
						- s_me_h);
						
gen duhw_lag_model = (-(k9)^2*(l2_s_u1) -(k10)^2*(l2_s_u2)
						-2*(k9)*(k10)*(l2_r_u1u2)
						- s_me_hw_cons);
gen duh_duhw_model = (k5*k9*(2*(m_s_u1)) + k6*k10*(2*(m_s_u2))
						+ (k5*k10 + k6*k9)*(2*m_r_u1u2)
						+ k7*k11*(2*m_s_v1) + k8*k12*(2*m_s_v2)
						+ (k7*k12 + k8*k11)*(2*m_r_v1v2));

#delimit cr

foreach var in duh2 duhw2 duh_lag duhw_lag duh_duhw {
	replace `var'_model = . if `var'==.
}
 
 
keep duh* 
collapse *

preserve 
	keep *_model
	xpose, clear varname
	ren v1 model
	ren _ varname
	sort varname
	save model, replace
restore 

preserve 
	keep *_SE
	xpose, clear varname
	ren v1 data_se
	ren _ varname
	replace varname = reverse(substr(reverse(varname),4,.))
	sort varname
	save SE, replace
restore 

drop *_model *_SE
foreach var of varlist * {
	ren `var' `var'_model
}
xpose, clear varname
ren v1 data
ren _ varname
sort varname

merge 1:1 varname using model
drop _m
replace varname = reverse(substr(reverse(varname),7,.))
merge 1:1 varname using SE
drop _m

ren varname varname0
append using model_vs_data_SE
replace group=8 if group==.

replace varname =  "Var({&Delta}h{subscript:1,t})" 	       if varname0 ==  "duh2" 	        
replace varname =  "Var({&Delta}h{subscript:2,t})" 	       if varname0 ==  "duhw2" 	    
replace varname =  "Cov({&Delta}h{subscript:1,t},{&Delta}h{subscript:1,t-2})"         if varname0 ==  "duh_lag"       
replace varname =  "Cov({&Delta}h{subscript:2,t},{&Delta}h{subscript:2,t-2})"        if varname0 ==  "duhw_lag"      
replace varname =  "Cov({&Delta}h{subscript:1,t},{&Delta}h{subscript:2,t})" 	   if varname0 ==  "duh_duhw" 	

replace order=39 if varname0 ==  "duh2"
replace order=40 if varname0 ==  "duhw2"
replace order=41 if varname0 ==  "duh_duhw"
replace order=42 if varname0 ==  "duh_lag"
replace order=43 if varname0 ==  "duhw_lag"


sort order
drop varname group
order varname0 model data data_se
outsheet using "$graphs/Table_Appendix_fit.csv", comma replace

cd "$do"
