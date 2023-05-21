/* Global Parameters */ 

clear all
macro drop _all

*** Store all do and provided data files in a single directory and write the path below
global do = "C:\Users\itaysap\Dropbox\family_labor_supply\bps_aer_replication_package"

*** Give a path to a directory where you wish the output files to be stored. 
*** These outputs include temporary dta files, but also the results for Tables 3, 4, 5 and 6
global output = "$do\output"

*** Give a path to a directory where you wish (most) Tables and Figures to be stored
global graphs = "$do\results"

*** Store all the PSID files in dta format in a single directory and write the path below
global PSID = "C:\Users\itaysap\Documents\PSID"

*** ado path for the GMM moment evaluation programs. Store that provided ado files in this directory 
adopath + "$do\gmm_ado

cd "$do"

cap mkdir "$output"
cap mkdir "$graphs"

************************ List of files that should be placed in the "do" directory:
/* (other than the do files listed below for direct run). Note that all files are provided
   in the rep.zip replication package. 

data files:
(1) min_wage.dta 			/* Minimum wage by state-year */ 
(2) psid_states.dta			/* State codes in PSID */
(3) price_indices.dta		/* Price index aggregate */
(4) poverty_line2.dta		/* Poverty line */
(5) cpi_detailed.dta		/* Price index detailed goods */
(6) init_cond.dta			/* Initial conditions for Figure 9, calculated using past years of PSID and CEX */
(7) table_2_3_5_annual.csv	/* NIPA table 2.3.5 as of 2011 (note that these are updated backwards by NIPA) */

do files called by other do files in the shell:
(1) variance_estimation.do
(2) tax_rate_estimation_snap.do
(3) pi_s_estimation.do
(4) frisch_estimation
(5) Parameter files for bootstrap stored in a sub-directory called "paramfiles": T4C1.do, T4C11.do, T4C2.do, T4C3.do, T4C4.do, T4C41.do, T4C5.do, T6C2.do, T6C3.do, T6C4.do, T6C5.do, T6C6.do
(6) gmm ado files stored in a sub-directory called "gmm_ado" (unless changed by the user above): 
	(a) gmm_age_fv4_pi5_ns_nltax_add
	(b) gmm_age_fv4_pi5_ns_nltax_b0
	(c) gmm_age_fv4_pi5_sep_nltax_b0
	
(7) identification_figures_sub

Matlab file for the concavity test:
(1) hessian
*/

************************  CREATION OF PSID FILE *******************************************************
do psid_interview			/* Extracts variables from the family files PSID */
do psid_individual			/* Constructs the panel of heads using individual data file (also drops Latino sample). 
							   Adds education for wife using the individual file (sometime missing in the family file) 
                               Adds age of youngest kid residing in household using the individual file */
do psid_consumption			/* Generates total consumption and asset measures at the household level */ 
do psid_sample				/* Selects PSID sample for the main analysis. More sample restrictions are in prepare_bootstrap.do */
do prepare_bootstrap		/* Prepares the final sample for GMM estimation (not using variables with unlikely jumps for particular obs and so on).*/ 

************************  PREPARATION OF FINAL DATA FOR ESTIMATION AND RUNS ESTIMATION ********
/*
Will generate the outputs required for:
Table 3
Table 4
Table 5 (additional calculations required - see below)
Table 6
Figure 1

*/

do residual_measures		 /* Calculates residual measures. This file runs more than once for different parameters for different specifications (there is a loop that does that) */ 
do advanced_information_test /* Runs the test for advanced information */ 
do bootstrap			 	 /* This file will call the parameters and then accordingly to:
								- variance_estimation 
								- tax_rate_estimation_snap
								- pi_s_estimation		
								- frisch_estimation 
								IMPORTANT NOTE 1: Need to run this file once for each set of parameters as explained in the top of the file. 
								IMPORTANT NOTE 2: Results from this file are stored in sub-directory output/TiCj_bs and TiCj_gmm
								Figure 1 will be stored in the directory defined by "graphs" global above. */ 

************************  ADDITIONAL TABLES and FIGURES *******************************************************
/*
Will generate the outputs required for:
Table 1
Table 2
Table 7 
Table 8
Table 9
*/

do aggregate_consumption			/* Generates the comparison for NIPA (Table 2) */
do descriptive_stats				/* Descriptive statistics by year.  (Table 1) */
do tables_7_8_9						/* Tables 7, 8 and 9 */ 


************************  ADDITIONAL RESULTS *******************************************************
/* 	For the following files to run, need to specify first the input directory in the global directory below. 
	Keeping the directory $output\T4C1_bs will allow to replicate the results from the paper. */
global directory "$output\T4C1_bs"


/*
Will generate the outputs required for:
Table 5 - AT Marshallian elasticities
Figure 3
Figure 4
Figure 5
Figure 6
Figure 7
Figure 8
Figure 9
Appendix Table 1
Appendix Figure 1
*/

do pi_s_by_age_figure			/* Figures for PI and S by age. (Figures 3,4) */ 
do calculate_AT_elasticities	/* Calculates after tax Marshallian elasticities (Extra numbers for Table 5) */
do marshallian_figure			/* Graphs of Marshallian elasticities and consumption response decomposition by age + kappas for insurance parameters for FS recipients */ 
do moment_fit_table				/* Figure 8 and appendix Table 1 */
do external_fit					/* Figure 9 */ 
do identification_figures 		/* Figure 1 (eta_cw) and appendix Figure 7. Calls the do "identification_figures_sub" */ 
do insurance_extensive_margin	/* Insurance calculation on the extensive margin (appears in the text) */


