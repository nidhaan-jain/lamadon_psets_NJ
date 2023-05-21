/* Evaluate insurance on extensive margin */ 

cd "$output"
u data4estimation, clear

cap log close
log using "$graphs/insurance_extensive_margin", replace t

*** Earnings of wife that just got employed at t
su wly if l2.wife_employed==0 & wife_employed==1

*** Earnings of husbands with non-participating wives
su ly if wife_employed==0 

*** Average Earnings of wife 
su wly if wife_employed==1

log close

