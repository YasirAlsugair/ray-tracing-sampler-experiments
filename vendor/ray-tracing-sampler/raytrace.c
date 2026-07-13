/**************************************
Ray Tracing Sampler
Copyright (C) 2025, Peter Behroozi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
***************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <inttypes.h>
#include <assert.h>
#include "raytrace.h"

/* Helper functions for dot products, norms, etc. */
double raytrace_dot(double *a, double *b, int64_t DIMS);
double raytrace_norm(double *a, int64_t DIMS) { return sqrt(raytrace_dot(a,a, DIMS)); }
double raytrace_norm_mul(double *a, double amul, double *b, double bmul, int64_t DIMS);

/* Helper functions/definitions for random numbers */
double xoshiro256ss_drand(struct xoshiro256ss_state *state); //Uniform random in [0,1)
double xoshiro256ss_normal_random(double mean, double stddev, struct xoshiro256ss_state *state); //Normally-distributed random
void raytrace_rand_gaussian(double *params, double stddev, int64_t DIMS, struct xoshiro256ss_state *state);

/* Helper functions/definitions for memory allocation */
#define stringify(x) #x
void *raytrace_check_realloc(void *ptr, size_t size, char *reason);
#define check_realloc_s(x,y,z) { (x) = raytrace_check_realloc((x),((int64_t)(y))*((int64_t)(z)), "Reallocating " #x " at " __FILE__ ":" stringify(__LINE__)); }
#define check_calloc_s(x,y,z) { check_realloc_s(x,y,z); memset(x, 0, ((int64_t)(y))*((int64_t)(z))); }



/* Ray tracing velocity update */
double UpdateV_Raytrace(double *v, double *grad, double dt, int64_t DIMS) {
  double mnorm = raytrace_norm(v, DIMS);
  double gnorm = raytrace_norm(grad, DIMS);
  if (!mnorm || !gnorm) return 0;
  double x = raytrace_norm_mul(v, gnorm, grad, mnorm, DIMS);
  double y = raytrace_norm_mul(v, gnorm, grad, -mnorm, DIMS);
  double angle = 2.0*atan2(y, x);
  double cos_angle = cos(angle);
  double sin_angle = sin(angle);
  if (sin_angle==0) return cos_angle*dt*mnorm*gnorm;
  double new_angle = 2.0*atan2(y, x*exp(dt*mnorm*gnorm/((double)DIMS-1.0)));
  double new_cos = cos(new_angle);
  double new_sin = sin(new_angle);
  double m_frac = new_sin / sin_angle;
  double g_frac = (new_cos - m_frac*cos_angle)*mnorm/gnorm; //we will add -grad*g_frac
  double delta_ln_luminosity = ((double)(DIMS-1))*log(fabs(sin_angle / new_sin));
  for (int64_t i=0; i<DIMS; i++) v[i] = v[i]*m_frac + grad[i]*g_frac;
  return delta_ln_luminosity;
}

/* HMC velocity update.  Returns the kinetic energy difference (equivalent
   to delta_ln_luminosity for ray tracing). */
double UpdateV_HMC(double *v, double *grad, double dt, int64_t DIMS) {
  double old_norm = raytrace_norm(v, DIMS);
  for (int64_t i = 0; i<DIMS; i++) v[i] +=  grad[i]*dt;
  double new_norm = raytrace_norm(v, DIMS);
  return 0.5*(new_norm*new_norm-old_norm*old_norm);
}

/* Partial velocity refresh */
void ScatterV(double *v, double refresh_rate, double dt, struct xoshiro256ss_state *state, int64_t DIMS) {
  double f = exp(-fabs(refresh_rate*dt));
  double alpha = sqrt(1.0-f*f);
  for (int64_t i=0; i<DIMS; i++)
    v[i] = f*v[i] + alpha*xoshiro256ss_normal_random(0, 1, state);
}

/* Position update */
void UpdateX(double *x, double *v, double dt, int64_t DIMS) {
  for (int64_t i=0; i<DIMS; i++) x[i] += v[i]*dt;
}

struct raytracer *Init_Raytracer(double *x, int64_t DIMS, int64_t trajectory_steps, double dt, double refresh_rate, double (*likelihood)(double *, int64_t),
				 void (*gradient)(double *, double *, int64_t), int64_t metropolis_check, enum raytrace_integration_type itype,
				 enum raytrace_mcmc_type mcmc_type) {
  struct raytracer *r = NULL;
  if ((DIMS < 2) && (mcmc_type==Raytracing)) {
    fprintf(stderr, "[Warning] DIMS must be > 1 to use ray tracing; defaulting to using HMC instead.\n");
    mcmc_type = HMC;
  }
  uint64_t default_state[4] = {8248649997206829523ull, 9897252137414991509ull, 4228299438205458557ull, 2836431340494989057ull};
  check_calloc_s(r, sizeof(struct raytracer), 1);
  check_calloc_s(r->x, sizeof(double), DIMS);
  memcpy(r->x, x, sizeof(double)*DIMS);
  r->DIMS = DIMS;
  r->steps = trajectory_steps;
  r->dt = dt;
  r->refresh_rate = refresh_rate;
  r->likelihood = likelihood;
  r->gradient = gradient;
  r->metropolis_check = metropolis_check;
  r->itype = itype;
  r->mcmc_type = mcmc_type;
  Raytrace_Init_Random_State(r, default_state, 4);
  return r;
}

void Free_Raytracer(struct raytracer **r) {
  if (r[0]==NULL) return;
  if (r[0]->x) free(r[0]->x);
  if (r[0]->v) free(r[0]->v);
  if (r[0]->v0) free(r[0]->v0);
  if (r[0]->x0) free(r[0]->x0);
  if (r[0]->grad) free(r[0]->grad);
  free(r[0]);
  r[0] = NULL;
}

/* Ray tracing algorithm.  With minimal changes, the same code works for Hamiltonian Monte Carlo. */
int64_t Raytrace(struct raytracer *r, double *likelihood) {
  //The only difference between HMC and Raytracing:
  double (*UpdateV)(double *, double *, double, int64_t) = (r->mcmc_type == HMC) ? &UpdateV_HMC : &UpdateV_Raytrace;

  struct xoshiro256ss_state *state = &(r->random_state);

  int64_t last_kick=0, passed_metro_check=1, need_v0 = 0;

#define DRIFT(t) { UpdateX(r->x, r->v, t, r->DIMS); last_kick = 0; }
#define KICK(t) { if (!last_kick) r->gradient(r->x, r->grad, r->DIMS); ln_luminosity += UpdateV(r->v, r->grad, t, r->DIMS); last_kick=1; }
  
  double initial_ln_likelihood = r->likelihood(r->x,r->DIMS);
  double ln_luminosity = 0;

  if (r->metropolis_check) {
    check_realloc_s(r->x0, sizeof(double), r->DIMS);
    memcpy(r->x0, r->x, sizeof(double)*r->DIMS);
    if (r->refresh_rate>0) need_v0=1;
  }
  
  if (!r->v) { check_calloc_s(r->v, sizeof(double), r->DIMS); need_v0=0; }
  if (!(r->refresh_rate>0) || raytrace_norm(r->v,r->DIMS)==0) raytrace_rand_gaussian(r->v, 1, r->DIMS, state);
  if (need_v0) { check_realloc_s(r->v0, sizeof(double), r->DIMS); memcpy(r->v0, r->v, sizeof(double)*r->DIMS); }

  check_calloc_s(r->grad, sizeof(double), r->DIMS);
  
  for (int64_t j=0; j<r->steps; j++) {
    enum raytrace_integration_type this_itype = r->itype;
    if (r->itype == RDKD) this_itype = (xoshiro256ss_drand(state) < 0.5) ? DKD : KDK;

    if (r->refresh_rate>0) ScatterV(r->v, r->refresh_rate, (j==0) ? r->dt/2.0 : r->dt, state, r->DIMS); //Partial velocity refreshment
    
    if (this_itype == KDK) {
      KICK(r->dt/2.0);
      DRIFT(r->dt);
      KICK(r->dt/2.0);
    }
    else if (this_itype == DKD) {
      DRIFT(r->dt/2.0);
      KICK(r->dt);
      DRIFT(r->dt/2.0);
    }
    else if (this_itype == Omelyan) {
      //Omelyan second order integrator
      double lambda = 0.1931833275037836;
      KICK(r->dt*lambda);
      DRIFT(r->dt/2.0);
      KICK(r->dt*(1.0-2.0*lambda));
      DRIFT(r->dt/2.0);
      KICK(r->dt*lambda);
    }
    else if (this_itype == Yoshida) {
      //Yoshida 4th order
      double w0 = -cbrt(2)/(2.0-cbrt(2));
      double w1 = 1.0/(2.0-cbrt(2));
      double c1, c2, c3, c4, d1, d2, d3;
      c1 = c4 = w1/2.0;
      c2 = c3 = (w0+w1)/2.0;
      d1 = d3 = w1;
      d2 = w0;
	
      DRIFT(c1*r->dt);
      KICK(r->dt*d1);
      DRIFT(c2*r->dt);
      KICK(r->dt*d2);
      DRIFT(c3*r->dt);
      KICK(r->dt*d3);
      DRIFT(c4*r->dt);	
    }

    else if (this_itype == Omelyan4th) { //Omelyan 4th order
      double rho = 0.2539785108410595;
      double theta = -0.3230286765269967e-01;
      double vartheta = 0.8398315262876693e-01;
      double lambda = 0.6822365335719091;
      KICK(r->dt*vartheta);
      DRIFT(rho*r->dt);
      KICK(r->dt*lambda);
      DRIFT(theta*r->dt);
      KICK(0.5*r->dt*(1.0-2.0*(lambda+vartheta)));
      DRIFT((1.0-2.0*(theta+rho))*r->dt);
      KICK(0.5*r->dt*(1.0-2.0*(lambda+vartheta)));
      DRIFT(theta*r->dt);
      KICK(r->dt*lambda);
      DRIFT(rho*r->dt);
      KICK(r->dt*vartheta);
    }
  }

  if (r->refresh_rate>0) ScatterV(r->v, r->refresh_rate, r->dt/2.0, state, r->DIMS); //Partial velocity refreshment

  double ln_l = r->likelihood(r->x,r->DIMS);

  if (r->metropolis_check) {
    if (log(drand48()) > ((ln_l-initial_ln_likelihood)-ln_luminosity)) {
      //Metropolis test failed
      memcpy(r->x, r->x0, sizeof(double)*r->DIMS);
      if (need_v0) for (int64_t i=0; i<r->DIMS; i++) r->v[i] = -r->v0[i]; //Restoration with velocity flip
      passed_metro_check = 0;
    }
    if (likelihood) *likelihood = initial_ln_likelihood;
  }
  else if (likelihood) *likelihood = ln_l;

  return passed_metro_check;
}


/* Helper functions for dot products, norms, etc. */
double raytrace_dot(double *a, double *b, int64_t DIMS) {
  double s = 0;
  for (int64_t i=0; i<DIMS; i++) s += a[i]*b[i];
  return s;
}

double raytrace_norm_mul(double *a, double amul, double *b, double bmul, int64_t DIMS) {
  double s = 0;
  for (int64_t i=0; i<DIMS; i++) {
    double x = amul*a[i]+bmul*b[i];
    s += x*x;
  }
  return sqrt(s);
}




/* Xoshiro256** implementation.  From https://en.wikipedia.org/wiki/Xorshift#xoshiro256** ,
 itself adapted from https://prng.di.unimi.it/xoshiro256starstar.c by David Blackman and Sebastiano Vigna. */
uint64_t xoshiro256ss_rol64(uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}


uint64_t xoshiro256ss(struct xoshiro256ss_state *state) {
	uint64_t *s = state->s;
	uint64_t const result = xoshiro256ss_rol64(s[1] * 5, 7) * 9;
	uint64_t const t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;
	s[3] = xoshiro256ss_rol64(s[3], 45);

	return result;
}

double xoshiro256ss_drand(struct xoshiro256ss_state *state)
{
    return (xoshiro256ss(state) >> 11) * (1.0/9007199254740992.0);
}


void Raytrace_Init_Random_State(struct raytracer *r, uint64_t *s, int64_t vals) {
  for (int64_t i=0; i<4; i++) {
    r->random_state.s[i] = s[i%vals]+(i/vals);
  }
}


//Use Box-Muller Transform to return random number.  y2 could be retained, but we discard to simplify state vector.
double xoshiro256ss_normal_random(double mean, double stddev, struct xoshiro256ss_state *state)
{
  double x1, x2, w;
  double y1;
  do {
    x1 = 2.0 * xoshiro256ss_drand(state) - 1.0;
    x2 = 2.0 * xoshiro256ss_drand(state) - 1.0;
    w = x1 * x1 + x2 * x2;
  } while ( w >= 1.0 );
  w = sqrt( (-2.0 * log( w ) ) / w );
  y1 = x1 * w;
  //  y2 = x2 * w;
  return (y1*stddev + mean);
}

void raytrace_rand_gaussian(double *params, double stddev, int64_t DIMS, struct xoshiro256ss_state *state) {
  for (int64_t i=0; i<DIMS; i++)
    params[i] = xoshiro256ss_normal_random(0,stddev, state);
}


void *raytrace_check_realloc(void *ptr, size_t size, char *reason) {
  if (size > 0) {
    void *res = realloc(ptr, size);
    if (res == NULL) {
      fprintf(stderr, "[Error] Failed to allocate memory (%s)!\n", reason);
      exit(EXIT_FAILURE);
    }
    return res;
  }
  if (ptr != NULL) free(ptr);
  return(NULL);
}
