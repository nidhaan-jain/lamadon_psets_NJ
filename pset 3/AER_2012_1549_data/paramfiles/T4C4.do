/* Bootstrap parameters
   This file includes the parameters for the estimation procedure */
   
global directory "T4C4_bs"

global bn = 220 	/* number of bootstrap samples. Zero will be no bootstrap */ 

global savereps = 0 /* Choose 1 to save the data generated in each rep - very consuming on memory! */ 

global bskeep = 200	/* Keep the first bskeep replications. Can be used if convergence is not obtained for some reps (then assign bs to be larger than bskeep which would be the desired number of replications */

global samplebs ""			/* 	List here the sampling condition. If left empty, the entire sample is used. 
								Note that this is only applied for the Frisch parameters estimation */ 

global consumption 	=  5	/*  5=totcons_psid_zero							*** The default
								6=same as 5 without housing */

global het_elasticities	= 12			/* 	1-  seperable case (Table 4, col 3)
										10- non-separable case without taxes (Table 4, col 2)
										12- Same as 10 but with non-linear taxes *** The default */

global use_pi			= 1 		/*  1- beta set to zero *** The default
										7- beta multiplies pi */ 


global fix_var			= 9				/* 	4- Variances changing by age and participation correction
											9- Variances constant by age and NO participation correction 
											10- Variances changing by age and NO participation correction *** The default */

global medass			= 0			/*  1- pi replaced by a measure which uses median assets by age and education (Table 6 Col 6) */ 
									
global maxiter 			= 25	/* max number of GMM iteration in each step */ 

global tax_graph		= 0		/* 1 for tax graph (figure 1 with the accompanied numbers) */

*** Calibrated M.E parameters									
scalar sh_me_y			= 0.04		/* 	Share of me in earnings out of the total variance of log(earnings) */ 
scalar sh_me_h			= 0.23		/* 	Share of me in hours out of the total variance of log(hours) */
scalar sh_me_w			= 0.13		/* 	Share of me in hours out of the total variance of log(earnings/hours) - note that this 
										is assuming hours and earnings' me are correlated */ 

*** Initial values   
* wage parameters
global s_u1_0	= 0.2		/* variance of u1 */ 
global s_u2_0	= 0.2		/* variance of u2 */
global s_v1_0	= 0.2		/* variance of v1 */ 
global s_v2_0	= 0.2		/* variance of v2 */
global r_u1_0	= 0			/* correlation of u1 and selection shock */ 
global r_u2_0	= 0			/* correlation of u2 and selection shock */ 
global r_v1_0	= 0			/* correlation of v1 and selection shock */ 
global r_v2_0	= 0			/* correlation of v2 and selection shock */ 
global r_u1u2_0 = 0			/* correlation of u1 and u2 */ 
global r_v1v2_0 = 0			/* correlation of v1 and v2 */ 

* Preference parameters - nonseparability of both
global eta_c_p_0 	= 0.5 		
global eta_h1_w1_0	= 0.5
global eta_h2_w2_0	= 0.5
global eta_h1_p_0 	= 0.3
global eta_h2_p_0 	= 0.3
global eta_c_w1_0 	= 0.1
global eta_c_w2_0	= 0.1
global eta_h1_w2_0	= 0.1
global eta_h2_w1_0	= 0.1

* Other parameters
global a0_0		= 1			/* Note: 1-beta rather than beta (the results will report beta) */ 
										
*** Internal assignment of parameters (not for use!)
global participation = 1
if ${fix_var}==9 | ${fix_var}==10 {
	global participation = 0
}

global poststring = "het${het_elasticities}pi${use_pi}fv${fix_var}cons${consumption}"
if ${medass}==1 {
	global poststring = "${poststring}_medass"
}

global nocommonesample "nocommonesample"
global gmm_config "winitial(identity) onestep"

global sep	= 0		
if ${het_elasticities}==1 {
	global sep	= 1		
}
									