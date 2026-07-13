###**************************************
### Ray Tracing Sampler
### Original Adam Implementation: Copyright (C) PyTorch Team 2024
### See original Adam source at https://github.com/pytorch/pytorch/blob/v2.5.0/torch/optim/adam.py
### Additional Changes to all functions: Copyright (C) 2025, Peter Behroozi
###
### Licensed under the PyTorch license, included with this repository
### in the file LICENSE.torch.
###
### Unless required by applicable law or agreed to in writing, software
### distributed under the License is distributed on an "AS IS" BASIS,
### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
### See the License for the specific language governing permissions and
### limitations under the License.
###***************************************/

import numpy as np
import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, 
                        _stack_if_compiling, _get_scalar_dtype, _capturable_doc, _differentiable_doc,
                        _foreach_doc, _fused_doc, _maximize_doc, _default_to_fused_or_foreach,
                        ParamsT, _view_as_real)
from typing import List, Optional, Tuple, Union
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices

__all__ = ["Raytracer", "raytracer", "sample_raytrace", "sample_hamiltonian"]


#[docs]
class Raytracer(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        *,
        weight_decay: float = 0,
        scale_likelihood: float = 0,
        stochastic_hmc = False,
        mom_norm: float = 1,
        dt: float = 0,
        mom_decay: float = 0,
        refresh_rate: float = 1,
        loss_tolerance: float = 0,
        verbose = False,
        norm_limit = -1,
    ):
        if not 0.0 <= dt:
            raise ValueError(f"Invalid timestep: {dt}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= mom_decay:
            raise ValueError(f"Invalid momentum_decay value: {mom_decay}")
        if not 0.0 <= scale_likelihood:
            raise ValueError(f"Invalid scale_likelihood value: {scale_likelihood}")
        if not 0.0 <= mom_norm:
            raise ValueError(f"Invalid mom_norm value: {mom_norm}")
        if not 0.0 <= mom_norm:
            raise ValueError(f"Invalid refresh_rate value: {refresh_rate}")
        params = list(params)
        total_params = sum(p.numel() for p in params if p.requires_grad)
        if (total_params < 2 and stochastic_hmc==False):
            print('[Warning] Ray tracing requires 2 or more dimensions.  Defaulting to HMC.')
            stochastic_hmc=True
        if (dt==0):
            dt = 0.001*torch.sqrt(1e3/total_params)
        if (scale_likelihood==0 and loss_tolerance==0):
            scale_likelihood = total_params
        elif (scale_likelihood==0):
            scale_likelihood = total_params/(2.0*loss_tolerance)
        defaults = dict(
            weight_decay=weight_decay,
            scale_likelihood=scale_likelihood,
            stochastic_hmc = stochastic_hmc,
            mom_norm = mom_norm,
            mom_decay = mom_decay,
            norm_limit = norm_limit,
            dt = dt,
            refresh_rate = refresh_rate,
            verbose=verbose,
            volume = 0.0,
            t = 0.0,
            ln_luminosity = 0,
            capturable = False,
            differentiable = False,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state['step']):
                    step_val = float(p_state["step"])
                    p_state["step"] = (torch.tensor(step_val, dtype=_get_scalar_dtype()))

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        momenta,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("Raytracer does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                # Momentum
                state["momenta"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format)
                state["momenta"].normal_(mean=0.0, std=group["mom_norm"])

            momenta.append(state["momenta"])
            state_steps.append(state["step"])
        return has_complex

#[docs]   
    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        net_volume = 0
        net_mom_loss = 0
        verbose = False
        for group in self.param_groups:
            #print("Group")
            params_with_grad = []
            grads = []
            state_steps = []
            momenta = []

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                momenta,
                state_steps,
            )

            (delta_volume, mom_loss, delta_ln_luminosity) = raytracer(
                params_with_grad,
                grads,
                momenta,
                state_steps,
                weight_decay=group["weight_decay"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                has_complex=has_complex,
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                scale_likelihood=group["scale_likelihood"],
                stochastic_hmc = group["stochastic_hmc"],
                mom_norm = group["mom_norm"],
                norm_limit = group["norm_limit"],
                refresh_rate = group["refresh_rate"],
                dt = group["dt"],
                verbose = group["verbose"],
                t = group["t"],
                ln_luminosity = group["ln_luminosity"],
                mom_decay = group["mom_decay"],
            )

            group["volume"] += delta_volume
            net_volume += group["volume"]
            net_mom_loss += mom_loss
            group["t"] += group["dt"]
        for group in self.param_groups:
            group["mom_loss"] = net_mom_loss
            group["ln_luminosity"] += delta_ln_luminosity
            if (group["verbose"]==True):
                verbose=True
        
        if (verbose):
            print ("Volume: ",net_volume, " T:", group["t"])
        return loss
    
    def first_step(self):
        for group in self.param_groups:
            group["t"] = float(0)
            group["ln_luminosity"] = float(0)
    
    def final_step(self):
        for group in self.param_groups:
            group["t"] = float(-1)

    def reverse_momentum(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if ("momenta" in state):
                    state["momenta"].mul_(-1)
    
    def set_scale_likelihood(self, scale):
        for group in self.param_groups:
            group["scale_likelihood"] = scale
        #print("Set scale_likelihood to ",scale)

    def set_dt(self, dt):
        for group in self.param_groups:
            group["dt"] = dt
        #print("Set dt to ",dt)
            
    def set_refresh_rate(self, rr):
        for group in self.param_groups:
            group["refresh_rate"] = rr   
            
    def scale_likelihood(self):
        return self.param_groups[0]['scale_likelihood']
    
    def ln_luminosity(self):
        return self.param_groups[0]['ln_luminosity']
    

Raytracer.__doc__ = r"""Implements Ray tracing sampler algorithm.

    """ + fr"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        dt (float, optional): timestep size, equivalent to learning rate.  Default value
           is 1e-3*sqrt(1e3/num_parameters).
        weight_decay (float, optional): weight decay coefficient (default: 0, as other choices cause non-reversible paths)
        mom_decay (float, optional): momentum decay coefficient (default: 0, as other choices cause non-reversible paths)
        scale_likelihood (float, optional): factor by which to scale loss function.  
          Recommended to set to approximately eff_num_parameters / (2*loss_delta),
          where loss_delta is the desired tolerance compared to the best fitting location, and
          eff_num_parameters is the effective number of parameters constrained by the training set.
          You may need to experiment with scale_likelihood to get the desired loss_delta, since
          the number of effective parameters is not always clear a priori.  Default value is
          num_parameters.
        stochastic_hmc (Bool, optional): mimic the behavior of stochastic Hamiltonian Monte
          Carlo (default: False).
        mom_norm (float, optional): optionally renormalize the momentum vector (default: 1).
          Changing this is strongly not recommended--you should change dt instead.
        refresh_rate (float, optional): rate per unit time for continuous momentum refreshment (default: 1).
        verbose (Bool, optional): print out additional information about parameter and momentum
          norms at every step.
    """



def raytracer(
    params: List[Tensor],
    grads: List[Tensor],
    momenta: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    capturable: bool = False,
    differentiable: bool = False,
    verbose: bool,
    weight_decay: float,
    mom_norm: float,
    norm_limit: float,
    refresh_rate: float,
    dt: float,
    t:float,
    ln_luminosity:float,
    scale_likelihood: float,
    stochastic_hmc: bool,
    mom_decay: float,
):
    r"""Functional API that performs ray tracing sampling.

    See :class:`~torch.optim.Raytracer` for details.
    """
    func = _single_tensor_raytracer
    
    return func(
        params,
        grads,
        momenta,
        state_steps,
        weight_decay=weight_decay,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
        mom_norm=mom_norm,
        norm_limit=norm_limit,
        refresh_rate=refresh_rate,
        dt=dt,
        verbose=verbose,
        t=t,
        ln_luminosity = ln_luminosity,
        scale_likelihood=scale_likelihood,
        stochastic_hmc=stochastic_hmc,
        mom_decay = mom_decay,
        capturable=capturable,
        differentiable=differentiable,
    )


    
def UpdateV(params, momenta, grads, scale_likelihood, total_params, dt):
    cos_angle = 0
    mnorm = 0
    gnorm = 0.0
    x=0
    y=0
    for i, param in enumerate(params):
        mnorm += torch.sum(torch.square(momenta[i]))
        gnorm += torch.sum(torch.square(grads[i]))
    mnorm = torch.sqrt(mnorm)
    gnorm = torch.sqrt(gnorm)
    if (torch.isnan(gnorm) or gnorm <= 0):
        print("Warning: zero gradient.  Check network structure and/or increase refresh_rate.")
        return 0
    for i, param in enumerate(params):
        y += torch.sum(torch.square(momenta[i]*gnorm + mnorm*grads[i]))
        x += torch.sum(torch.square(momenta[i]*gnorm - mnorm*grads[i]))
    angle = 2 * torch.atan2(torch.sqrt(y), torch.sqrt(x)) #numerically stable angle between -grad and momenta
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    if (sin_angle==0):
        return dt*mnorm*gnorm*scale_likelihood*cos_angle #gradient is parallel to velocity

    new_angle = 2 * torch.atan2(torch.sqrt(y), torch.sqrt(x)*torch.exp(dt*mnorm*gnorm*scale_likelihood/(total_params-1))) #numerically stable angle between -grad and momenta
    new_cos = torch.cos(new_angle)
    new_sin = torch.sin(new_angle)
    m_frac = new_sin / sin_angle
    g_frac = (new_cos - m_frac*cos_angle)*mnorm/gnorm #we will add -grad*g_frac
    
    for i, param in enumerate(params):
        momenta[i].mul_(m_frac)
        momenta[i].add_(grads[i], alpha = -g_frac)
    
    delta_ln_luminosity = (total_params-1)*torch.log(torch.abs(sin_angle / new_sin))
    return delta_ln_luminosity

def UpdateV_HMC(params, momenta, grads, scale_likelihood, total_params, dt):
    mnorm = 0
    mnorm2 = 0
    for i, param in enumerate(params):
        mnorm += torch.sum(torch.square(momenta[i]))
        momenta[i].add_(grads[i], alpha=-scale_likelihood*dt)
        mnorm2 += torch.sum(torch.square(momenta[i]))
    return 0.5*(mnorm2 - mnorm)

def ScatterV(params, momenta, mom_norm, refresh_rate, dt):
    f=torch.exp(-torch.abs(torch.tensor(refresh_rate*dt)))
    for i, param in enumerate(params):
        momenta[i].mul_(f)
        diffusion = torch.zeros_like(momenta[i])
        diffusion.normal_(mean=0.0, std=mom_norm)
        momenta[i].add_(diffusion, alpha=torch.sqrt(1.0-f*f))

def sample_raytrace(
   params_init: Tensor, log_prob_fn, n_steps, n_leapfrog_steps,
    step_size, refresh_rate=0, metro_check=1, sample_hmc=False,
        device=None, samples_device=None, scale_likelihood=1.0):

    params = params_init.clone().to(device=device)
    momenta = torch.zeros_like(params)
    
    total_params = params.numel()
    if (total_params < 2 and sample_hmc==False):
        print('[Warning] Ray tracing requires 2 or more dimensions.  Defaulting to HMC instead.')
        sample_hmc=True
    samples = torch.empty((n_steps, total_params), device=samples_device)
    likelihoods = torch.empty((n_steps), device=samples_device)

    dt = step_size
    UpdateV_Func = UpdateV
    if (sample_hmc): UpdateV_Func = UpdateV_HMC
    accepted=n_steps

    #Note that the UpdateV functions assume that the gradient is of the loss
    # function, instead of the likelihood function, so we flip the sign of
    # scale_likelihood to compensate
    scale_likelihood = -scale_likelihood
    
    for j in range(n_steps):
        initial_ln_likelihood = 0
        final_ln_likelihood = 0
        delta_ln_luminosity = 0
        params.requires_grad_(False)
        last_params=params.clone().detach()
        momenta.normal_(mean=0.0, std=1.0)
        lp = log_prob_fn(params)
        initial_ln_likelihood = lp.clone().detach()
        for i in range(n_leapfrog_steps+1):
            if (refresh_rate):
                if (i==0): ScatterV([params], [momenta], 1.0, refresh_rate, dt/2.0)
                params.add_(momenta, alpha=dt/2.0)

                if (i>0 and i<n_leapfrog_steps):
                    ScatterV([params], [momenta], 1.0, refresh_rate, dt)
                    params.add_(momenta, alpha=dt/2.0)
        
                if (i==n_leapfrog_steps):
                    ScatterV([params], [momenta], 1.0, refresh_rate, dt/2.0)
                    
            else:
                if (i==0 or i==n_leapfrog_steps): dt *= 0.5
                params.add_(momenta, alpha=dt)
                if (i==0 or i==n_leapfrog_steps): dt = step_size

            if (i<n_leapfrog_steps):
                params.requires_grad_(True)
                lp = log_prob_fn(params)
                grads = torch.autograd.grad(lp, params)
                delta_ln_luminosity += UpdateV_Func([params], [momenta], grads, scale_likelihood, total_params, dt)
                params.requires_grad_(False)

        
        lp = log_prob_fn(params)
        final_ln_likelihood = lp.clone().detach()
        log_likelihood_diff = final_ln_likelihood - initial_ln_likelihood
        log_accept_prob = log_likelihood_diff - delta_ln_luminosity
        log_accept_prob = torch.nan_to_num(log_accept_prob, nan=-torch.inf)
        r=torch.rand(1)
        if (r[0]>torch.exp(log_accept_prob)): #Rejection
            params = last_params.clone().detach()
            final_ln_likelihood = initial_ln_likelihood
            accepted -= 1
        samples[j] = params.clone().detach().to(device=samples_device)
        likelihoods[j] = final_ln_likelihood.clone().detach().to(device=samples_device)

    print("Accepted: %f%%" % (100.0*accepted/n_steps))
    return (samples,likelihoods)


'''            
params.requires_grad_(True)
            lp = log_prob_fn(params)
            grads = torch.autograd.grad(lp, params)
            params.requires_grad_(False)
            if (refresh_rate):
                if (i==0): ScatterV([params], [momenta], 1.0, refresh_rate, dt/2.0)
                delta_ln_luminosity += UpdateV_Func([params], [momenta], grads, scale_likelihood, total_params, dt/2.0)

                if (i>0 and i<n_leapfrog_steps):
                    ScatterV([params], [momenta], 1.0, refresh_rate, dt)
                    delta_ln_luminosity += UpdateV_Func([params], [momenta], grads, scale_likelihood, total_params, dt/2.0)
        
                if (i==n_leapfrog_steps):
                    ScatterV([params], [momenta], 1.0, refresh_rate, dt/2.0)
                    
            else:
                if (i==0 or i==n_leapfrog_steps): dt *= 0.5
                delta_ln_luminosity += UpdateV_Func([params], [momenta], grads, scale_likelihood, total_params, dt)
                if (i==0 or i==n_leapfrog_steps): dt = step_size

            if (i<n_leapfrog_steps):
                params.add_(momenta, alpha=dt)
'''

def sample_hamiltonian(
   params_init: Tensor, log_prob_fn, n_steps, n_leapfrog_steps,
    step_size, refresh_rate=0, metro_check=1, sample_hmc=True,
        device=None, samples_device=None, scale_likelihood=1.0):
    return sample_raytrace(params_init, log_prob_fn, n_steps, n_leapfrog_steps,
                           step_size, refresh_rate, metro_check, sample_hmc,
                           device, samples_device, scale_likelihood)

            

def _single_tensor_raytracer(
    params: List[Tensor],
    grads: List[Tensor],
    momenta: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    has_complex: bool,
    mom_norm: float,
    norm_limit: float,
    dt: float,
    t:float,
    ln_luminosity:float,
    scale_likelihood: float,
    stochastic_hmc: bool,
    verbose: bool,
    mom_decay: float,
    refresh_rate: float,
    capturable: bool,
    differentiable: bool,
):

    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    delta_vol = 0
    param_norm = 0
    mom_length = 0
    raytracing = 1
    num_params = 0
    delta_ln_luminosity = 0
    mnorm2 = 0
    pnorm = 0
    if (stochastic_hmc==1): raytracing = 0
    
    UpdateV_Func = UpdateV
    if (not raytracing): UpdateV_Func = UpdateV_HMC
    total_params = 0
    total_params = sum(p.numel() for p in params if p.requires_grad)

    if (refresh_rate):
        if (t==0): ScatterV(params, momenta, mom_norm, refresh_rate, dt/2.0)

        delta_ln_luminosity += UpdateV_Func(params, momenta, grads, scale_likelihood, total_params, dt/2.0)

        if (t>0):
            ScatterV(params, momenta, mom_norm, refresh_rate, dt)
            delta_ln_luminosity += UpdateV_Func(params, momenta, grads, scale_likelihood, total_params, dt/2.0)
        
        if (t<0):
            ScatterV(params, momenta, mom_norm, refresh_rate, dt/2.0)
    else:
        if (t<=0): dt *= 0.5
        delta_ln_luminosity += UpdateV_Func(params, momenta, grads, scale_likelihood, total_params, dt)
        if (t<=0): dt *= 2.0
        

    for i, param in enumerate(params):    
        mnorm2 += torch.sum(torch.square(momenta[i]))
        pnorm += torch.sum(torch.square(param))
    mnorm2 = torch.sqrt(mnorm2)
    pnorm = torch.sqrt(pnorm)        

    for i, param in enumerate(params):
        grad = grads[i]
        step_t = state_steps[i]
        mom = momenta[i]
        num_params += len(mom)

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        if (weight_decay):
            param.mul_(1 - dt * weight_decay)
            delta_vol += torch.log1p(- dt * weight_decay)*torch.numel(param)

        #update position
        if (t>=0):
            param.add_(mom, alpha=dt)
                        
        # Perform momentum decay
        if (mom_decay):
            mom.mul_(1 - mom_decay)
            delta_vol += torch.log1p(-mom_decay)*torch.numel(mom)
        
        param_norm += torch.sum(torch.square(param))
        mom_length += torch.sum(torch.square(mom))

    extreme_mom_length = 0
    if (mom_length > 200.0*num_params*mom_norm): extreme_mom_length=1
    
    if (extreme_mom_length and (not raytracing)):
        if (verbose):
            print("Reset momentum due to extreme value")
            print(mom_length)
            print(200.0*num_params*mom_norm)
        new_mom_length = 0
        for i, param in enumerate(params):
            mom = momenta[i]
            mom.normal_(mean=0.0, std=mom_norm)
            new_mom_length += torch.sum(torch.square(mom))
        mom_length = new_mom_length

    if (verbose):
        print("Param norm: %e; Momentum Norm: %e; dt: %e" % (torch.sqrt(param_norm), torch.sqrt(mom_length), dt))
    return (delta_vol, mom_length/(mom_norm*mom_norm)*0.5, delta_ln_luminosity)

