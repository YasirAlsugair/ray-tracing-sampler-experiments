/* Autocorrelation Time Estimator
   Copyright (C) 2025  Peter Behroozi

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
   If so, it should be in a file called "LICENSE".
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include "check_syscalls.h"

double *params = NULL;
int64_t total_steps = 0;


double autocorrelation(double *params, int64_t nwalkers, int64_t total_steps, int64_t ss, int64_t absolute) {
  int64_t max_step_size = total_steps/nwalkers;
  if (ss < 1) ss=1;
  if (ss > max_step_size) ss = max_step_size;
  int64_t i, max_i = total_steps - (ss*nwalkers);
  double a1=0, a2=0, s1=0, s2=0, s12=0;
  for (i=0; i<max_i; i++) {
    a1 += params[i];
    a2 += params[i+ss*nwalkers];
  }
  a1 /= (double)max_i;
  a2 /= (double)max_i;

  double aa1 = 0, aa2 = 0;
  if (absolute) {
    for (i=0; i<max_i; i++) {
      aa1 += fabs(params[i]-a1);
      aa2 += fabs(params[i+ss*nwalkers]-a2);
    }
    aa1 /= (double)max_i;
    aa2 /= (double)max_i;
  }
  
  int64_t duplicates = 0;
  if (absolute) {
    for (i=0; i<max_i; i++) {
      /*if ((params[i])==(params[i+ss*nwalkers])) {
	duplicates++;
	continue;
	}*/
      double d1 = fabs(params[i]-a1)-aa1;
      s1 += d1*d1;
      double d2 = fabs(params[i+ss*nwalkers] - a2)-aa2;
      s2 += d2*d2;
      s12 += d1*d2;
    }
  } else {
    for (i=0; i<max_i; i++) {
      /*if ((params[i])==(params[i+ss*nwalkers])) {
	duplicates++;
	continue;
	}*/
      double d1 = params[i]-a1;
      s1 += d1*d1;
      double d2 = params[i+ss*nwalkers]-a2;
      s2 += d2*d2;
      s12 += d1*d2;
    }
  }
  s1 /= (double)(max_i-duplicates);
  s2 /= (double)(max_i-duplicates);
  s12 /= (double)(max_i-duplicates);
  double fdup = (double)duplicates / (double) max_i;
  return fdup + (1.0-fdup)*fabs(((s1 > 0 && s2 > 0) ? s12/sqrt(s1*s2) : 0));
}

//Autocorrelation time estimator, from Sokal
double autocorrelation_time_estimate(double *params, int64_t nwalkers, int64_t total_steps, int64_t absolute) {
  double tau = 1;
  int64_t i, max_step_size = total_steps/nwalkers;
  double skip_size = 1.0;
  for (i=1; i<max_step_size; i+=skip_size) {
    tau += 2.0*skip_size*fabs(autocorrelation(params, nwalkers, total_steps, i, absolute));
    if (i>15*skip_size) skip_size *= 2.0;
    if (i>5.0*tau) break;
  }
  return tau;
}

//Helper function to perform autocorrelation and print results
void autocorrelation_print(double *params, int64_t nwalkers, int64_t total_steps, int64_t absolute, FILE *out) {
  int64_t i;
  int64_t max_step_size = total_steps/nwalkers;
  for (i=1; i<max_step_size; i++) {
    fprintf(out, "%"PRId64" %f\n", i, autocorrelation(params, nwalkers, total_steps, i, absolute));
  }
  fprintf(out, "\n");
}


int64_t read_params(char *buffer, double *params, int max_n) {
  int num_entries = 0;
  char *cur_pos = buffer, *end_pos;
  double val = strtod(cur_pos, &end_pos);
  while (cur_pos != end_pos && num_entries < max_n) {
    params[num_entries] = val;
    num_entries++;
    cur_pos=end_pos;
    while (*cur_pos==' ' || *cur_pos=='\t' || *cur_pos=='\n') cur_pos++;
    val = strtod(cur_pos, &end_pos);
  }
  return num_entries;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "Usage: %s <mcmc_output> <ndims> <nwalkers>\n", argv[0]);
    fprintf(stderr, "Assumes one space-separated column per dimension, followed by chi^2 or ln(P).\n");
    exit(EXIT_FAILURE);
  }

  char *buffer = NULL;
  int64_t dims = atol(argv[2]);
  int64_t nwalkers = atol(argv[3]);
  FILE *in = check_fopen(argv[1], "r");

  int64_t max_size = (1 + /*sign*/
		      (1+1+20) + /*number+decimal places*/
		      (2) + /* e+/- */
		      (3) + /*exponent*/
		      1)    /*space*/
    * (dims+3) + 2; //Final "\n" and "\0";
  check_realloc_s(buffer, sizeof(char), max_size);
  while (fgets(buffer, max_size, in)) {
    if (buffer[0]=='#') continue;
    check_realloc_every(params, sizeof(double)*(dims+1), total_steps, nwalkers);
    if (read_params(buffer, params+((dims+1)*total_steps), dims+1) < dims+1) continue;
    total_steps++;
  }

  int64_t i, j;
  double avg = 0, avg2 = 0;
  double chi2_corr = 0;
  int64_t static_dims = 0;
  free(buffer);
  double *dim_params = NULL;
  check_realloc_s(dim_params, sizeof(double), total_steps);
  int64_t converged = 1;

  for (i=0; i<dims+1; i++) {
    double avg_dim = 0;
    double var_dim = 0;
    for (j=0; j<total_steps; j++) {
      dim_params[j] = params[j*(dims+1)+i];
      avg_dim += dim_params[j];
      var_dim += dim_params[j]*dim_params[j];
    }
    double acor = autocorrelation_time_estimate(dim_params, nwalkers, total_steps,0);
    double acor2 = autocorrelation_time_estimate(dim_params, nwalkers, total_steps,1);
    double sd = var_dim/(double)total_steps - pow(avg_dim/(double)total_steps,2);
    if (sd>0) sd = sqrt(sd);
    printf("Dimension %"PRId64": %lf %lf (avg: %e; sd: %e)\n", i, acor, acor2, avg_dim/(double)total_steps, sd);
    if (acor*5 > (total_steps/nwalkers) || acor2*5 > (total_steps/nwalkers))
      converged = 0;
    if (i<dims) {
      if (acor > 1 && acor2>1) {
	avg += acor;
	avg2 += acor2;
      } else {
	static_dims++;
      }
    } else {
      chi2_corr = acor;
    }
  }
  avg /= (double)(dims-static_dims);
  avg2 /= (double)(dims-static_dims);
  printf("Likelihood: %lf\n", chi2_corr);
  printf("Avg: %lf\n", avg);
  printf("Avg Abs: %lf\n", avg2);
  if (avg < chi2_corr) avg = chi2_corr;
  if (avg < avg2) avg = avg2;
  printf("Max of Likelihood, Average, Average Abs.: %lf\n", avg);
  if (!converged)
    printf("***Warning: Autocorrelation times are large relative to chain lengths, and are likely underestimates.\n");
  //autocorrelation_print(dim_params, nwalkers, total_steps, 0, stdout);
  return 0;
}
