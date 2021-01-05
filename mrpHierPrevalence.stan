data {
  int<lower = 0> N;  // number of tests in the sample in Ethiopia
  int<lower = 0, upper = 1> y[N];  // 1 if positive, 0 if negative

  int y_sens;
  int n_sens;

  int<lower = 0> J_spec;
  int<lower = 0> y_spec [J_spec];
  int<lower = 0> n_spec [J_spec];
  real<lower = 0> logit_spec_prior_scale;

  // 0 if female, 1 if male
  vector<lower = 0, upper = 1>[N] male;
  // age codes: 1 - [18,35) 2 - [35, 50) 3 - [50, 65) 4 - [65, 105)
  int<lower = 1, upper = 4> age[N];
  //number of zone codes (10 at the subcity/zone level in addis, 3 in jimma)
  int<lower = 0> N_zone;
  // zone codes 1 through 10 Addis, 1..3 Jimma
  int<lower = 1, upper = N_zone> zone[N];
  vector[N_zone] x_zone;  // predictors at the zone code level (% over 60 yo)
  int<lower = 0> J;  // number of population cells, J = 2*4*13
  vector<lower = 0>[J] N_pop;  // population sizes for poststratification
  real intercept_prior_mean;
  real<lower = 0> intercept_prior_scale;
  real<lower = 0> coef_prior_scale;

  int<lower=0, upper=1> sample_prior;
}

parameters {
  real mu_logit_spec;
  real<lower = 0> sigma_logit_spec;
  vector<offset = mu_logit_spec, multiplier = sigma_logit_spec>[J_spec] logit_spec;

  real<lower = 0, upper = 1> sens;

  vector[3] b;  // intercept, coef for male, and coef for x_zone
  real<lower = 0> sigma_age;
  real<lower = 0> sigma_zone;
  vector<multiplier = sigma_age>[4] a_age;  // varying intercepts for age category
  vector<multiplier = sigma_zone>[N_zone] a_zone;  // varying intercepts for zone code
}

transformed parameters{
  vector[J_spec] spec = inv_logit(logit_spec);
  vector[N] p = inv_logit(b[1] + b[2] * male + b[3] * x_zone[zone] + a_age[age] + a_zone[zone]);
  vector[N] p_sample = p * sens + (1 - p) * (1 - spec[1]);
}

model {
  //y_sample
 if(sample_prior==0) y ~ bernoulli(p_sample);
  y_spec ~ binomial(n_spec, spec);
  y_sens ~ binomial(n_sens, sens);

  logit_spec ~ normal(mu_logit_spec, sigma_logit_spec);
  sigma_logit_spec ~ normal(0, logit_spec_prior_scale);
  // mu_logit_spec ~ normal(4, 2); // weak prior on mean of distribution of spec

  a_age ~ normal(0, sigma_age);
  a_zone ~ normal(0, sigma_zone);
  // prior on centered intercept
  b[1] + b[2] * mean(male) + b[3] * mean(x_zone[zone]) ~ normal(intercept_prior_mean, intercept_prior_scale);
  b[2] ~ normal(0, coef_prior_scale);
  sigma_age ~ normal(0, coef_prior_scale);
  sigma_zone ~ normal(0, coef_prior_scale);
  b[3] ~ normal(0, coef_prior_scale / sd(x_zone[zone]));  // prior on scaled coefficient
}

generated quantities {
  // Estimated prevalence at the population level
  real p_avg;

  vector[J] p_pop;  // population prevalence in the J poststratification cells
  vector[N_zone] p_zone; // population prevalence in the N_zone zones

  // estimated prevalences at different demographic aggregation levels
  real p_female;
  real p_male;
  real p_age18_34;
  real p_age35_49;
  real p_age50_64;
  real p_age65_over;

  // indexes in p_pop corresponding to specific demographic subgroups
  int f_idx[J/2];
  int m_idx[J/2];
  int age1_idx[J/4];
  int age2_idx[J/4];
  int age3_idx[J/4];
  int age4_idx[J/4];

  // counters, auxiliary
  int count;
  int fcount;
  int mcount;
  int a1count;
  int a2count;
  int a3count;
  int a4count;
  int cells_per_zone;

  int<lower=0, upper=1> y_rep[N];

  count = 1;
  cells_per_zone = J/N_zone;
  fcount = 1;
  mcount = 1;
  a1count = 1;
  a2count = 1;
  a3count = 1;
  a4count = 1;

  for (i_zone in 1:N_zone) {
    for (i_age in 1:4) {
      for (i_male in 0:1) {
	p_pop[count] = inv_logit(b[1] + b[2] * i_male + b[3] * x_zone[i_zone] + a_age[i_age]+ a_zone[i_zone]);

	if(i_male == 0) {
	  f_idx[fcount] = count;
	  fcount += 1;
	}
	else {
	  m_idx[mcount] = count;
	  mcount += 1;
	}

	if (i_age == 1){
	  age1_idx[a1count] = count;
	  a1count += 1;
	}
	else if (i_age == 2){
	  age2_idx[a2count] = count;
	  a2count += 1;
	}
	else if (i_age == 3){
	  age3_idx[a3count] = count;
	  a3count += 1;
	}
	else if (i_age == 4){
	  age4_idx[a4count] = count;
	  a4count += 1;
	}

	count += 1;
      }
    }
    p_zone[i_zone] = dot_product(N_pop[count - cells_per_zone:count-1], p_pop[count - cells_per_zone:count-1]) /
      sum(N_pop[count - cells_per_zone:count-1]);
  }

  p_avg = dot_product(N_pop, p_pop) / sum(N_pop);

  p_age18_34 = dot_product(N_pop[age1_idx], p_pop[age1_idx]) / sum(N_pop[age1_idx]);
  p_age35_49 = dot_product(N_pop[age2_idx], p_pop[age2_idx]) / sum(N_pop[age2_idx]);
  p_age50_64 = dot_product(N_pop[age3_idx], p_pop[age3_idx]) / sum(N_pop[age3_idx]);
  p_age65_over = dot_product(N_pop[age4_idx], p_pop[age4_idx]) / sum(N_pop[age4_idx]);

  p_female = dot_product(N_pop[f_idx], p_pop[f_idx]) / sum(N_pop[f_idx]);
  p_male = dot_product(N_pop[m_idx], p_pop[m_idx]) / sum(N_pop[m_idx]);

  // PPC
  for(i in 1:N) y_rep[i] = bernoulli_rng(p_sample[i]);
}
