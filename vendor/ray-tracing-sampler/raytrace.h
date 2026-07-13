#ifndef __RAYTRACE_H__
#define __RAYTRACE_H__
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <assert.h>

/* Ray tracing velocity update.  Returns change in ln luminosity. */
double UpdateV_Raytrace(double *v, double *grad, double dt, int64_t DIMS);

/* HMC velocity update.  Returns the kinetic energy difference (equivalent
   to delta_ln_luminosity for ray tracing). */
double UpdateV_HMC(double *v, double *grad, double dt, int64_t DIMS);

/* Helper function to update position */
void UpdateX(double *x, double *v, double dt, int64_t DIMS);

enum raytrace_integration_type {
  KDK,
  DKD,
  RDKD,
  Omelyan,
  Yoshida,
  Omelyan4th
};

enum raytrace_mcmc_type {
  Raytracing,
  HMC
};

struct xoshiro256ss_state { uint64_t s[4]; };

struct raytracer {
  double *x, *v, dt, refresh_rate;
  double *x0, *v0, *grad;
  int64_t DIMS, steps, metropolis_check;
  double (*likelihood)(double *, int64_t);
  void (*gradient)(double *, double *, int64_t);
  enum raytrace_integration_type itype;
  enum raytrace_mcmc_type mcmc_type;
  struct xoshiro256ss_state random_state;
};


struct raytracer *Init_Raytracer(double *x, int64_t DIMS, int64_t trajectory_steps, double dt, double refresh_rate,
				 double (*likelihood)(double *, int64_t), void (*gradient)(double *, double *, int64_t),
				 int64_t metropolis_check, enum raytrace_integration_type itype,
				 enum raytrace_mcmc_type mcmc_type);

/* Sample implementation of Raytracing and HMC. */
//Returns 1 if sampling succeeded, 0 if it failed
int64_t Raytrace(struct raytracer *r, double *likelihood);

/* Releases memory of sampler structure and nulls out the pointer */
void Free_Raytracer(struct raytracer **r);

/* Initializes the random state of the Raytracer */
void Raytrace_Init_Random_State(struct raytracer *r, uint64_t *s, int64_t vals);

#endif /* ndef __RAYTRACE_H__ */
