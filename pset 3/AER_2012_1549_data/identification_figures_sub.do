/* Called by identification_figures */ 

		tempvar psi_c_lam psi_c_w1 psi_c_w2 ///
				psi_h1_lam psi_h1_w1 psi_h1_w2 ///
				psi_h2_lam psi_h2_w1 psi_h2_w2 /// 
				psi_h_lam psi_h_w1 psi_h_w2 ///
				k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 ///
				pi denom 
	
local eta_c_p	= $eta_c_p
local eta_c_w1	= $eta_c_w1
local eta_h1_w1	= $eta_h1_w1	 
local eta_c_w2	= $eta_c_w2
local eta_h1_w2	= $eta_h1_w2
local eta_h2_w2	= $eta_h2_w2
local a0		= $a0
local eta_h1_p	= $eta_h1_p
local eta_h2_p	= $eta_h2_p
local eta_h2_w1 = $eta_h2_w1

gen `pi'	= `a0'*pi_hat `if' 

gen `psi_c_lam'=.
gen `psi_c_w1'=.
gen `psi_c_w2'=.
gen `psi_h1_lam'=.
gen `psi_h1_w1'=.
gen `psi_h1_w2'=.
gen `psi_h2_lam'=.
gen `psi_h2_w1'=.
gen `psi_h2_w2'=.

noisily	mata: calc_invQR()

gen `psi_h_lam' = s_hat*`psi_h1_lam' + (1-s_hat)*`psi_h2_lam'
gen `psi_h_w1' 	= s_hat*`psi_h1_w1' + (1-s_hat)*`psi_h2_w1'
gen `psi_h_w2' 	= s_hat*`psi_h1_w2' + (1-s_hat)*`psi_h2_w2'
gen `denom'  	= `psi_c_lam' - (1-tau)*(`pi')*`psi_h_lam'

/* consumption Equation */
gen `k1' = `psi_c_w1'
gen `k2' = `psi_c_w2'
gen `k3' = `psi_c_w1' + `psi_c_lam'*((1-tau)*(`pi')*(s_hat+`psi_h_w1')-`psi_c_w1')/`denom'
gen `k4' = `psi_c_w2' + `psi_c_lam'*((1-tau)*(`pi')*((1-s_hat)+`psi_h_w2')-`psi_c_w2')/`denom'

/* Head's earnings equation */
gen `k5' = 1 + `psi_h1_w1'
gen `k6' = `psi_h1_w2'
gen `k7' = 1 + `psi_h1_w1' + `psi_h1_lam'*((1-tau)*(`pi')*(s_hat+`psi_h_w1') - `psi_c_w1')/`denom'
gen `k8' = `psi_h1_w2' + `psi_h1_lam'*((1-tau)*(`pi')*((1-s_hat)+`psi_h_w2') - `psi_c_w2')/`denom'

/* Wife's earnings equation */
gen `k9' = `psi_h2_w1'
gen `k10' = 1 + `psi_h2_w2'
gen `k11' = `psi_h2_w1' + `psi_h2_lam'*((1-tau)*(`pi')*(s_hat+`psi_h_w1') - `psi_c_w1')/`denom'
gen `k12' = 1 + `psi_h2_w2' + `psi_h2_lam'*((1-tau)*(`pi')*((1-s_hat)+`psi_h_w2') - `psi_c_w2')/`denom'


foreach var in  k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 {
		gen `var'_temp = ``var''
}

collapse k*temp* eta* a0

