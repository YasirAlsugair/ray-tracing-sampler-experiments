#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include "../raytrace.h"

#define DIMENSIONS 10000

double gaussian_ln_likelihood(double *params, int64_t DIMS) {
  double s = 0;
  for (int64_t i=0; i<DIMS; i++) s -= 0.5*params[i]*params[i];
  return s;
}

void gaussian_gradient(double *params, double *grad, int64_t DIMS) {
  //ln L = (-1/2 x^2) -> grad ln L = -X
  for (int64_t i=0; i<DIMS; i++) grad[i] = -(params[i]);
}


int main(int argc, char **argv) {
  double x0[DIMENSIONS] = {0}, likelihood = 0;
  int64_t steps = 30, chain_points = 1000, accepted = 0;
  
  struct raytracer *r = Init_Raytracer(x0, DIMENSIONS, steps, M_PI/(2.0*steps),
				       0, &gaussian_ln_likelihood,
				       &gaussian_gradient, 1, RDKD, Raytracing);

  /* Optionally set random state */
  //uint64_t s[4] = {my random numbers};
  //Raytrace_Init_Random_State(r, s, 4);

  /* Perform burn-in */
  r->metropolis_check = 0;
  for (int64_t i=0; i<chain_points/10; i++) Raytrace(r, NULL);

  /* Perform sampling */
  r->metropolis_check = 1;
  printf("#X0 X1 X2 Likelihood\n");
  for (int64_t i=0; i<chain_points; i++) {
    accepted += Raytrace(r, &likelihood);
    printf("%f %f %f %f\n", r->x[0], r->x[1], r->x[2], likelihood);
  }
  printf("#Accepted fraction: %f\n", (double)accepted/(double)chain_points);
  return 0;
}
