# Instructions on How to Use the MCMC Mode  

## Introduction  
EBS's MCMC mode was created to survey of required integration time for reaching required SNR ($t_{req}$) in high dimensional spaces of coronagraphic states. Here, a "coronagraph state" is a vector of values specifying observational and instrument variables (e.g. spectral bandwidth, contrast as a function of angular separation, exo-zodi level) that determine $t_{req}$.  The survey algorithm utilizes a Markov-chain Monte Carlo (MCMC) approach, as implemented by the Python EMCEE package (https://emcee.readthedocs.io), to sample coronagraph states. A Markov chain has the property that, after a "burn in" period, the chain generates values representative of those drawn from a target (or "posterior") probability distribution; for multi-variable PDFs, the Markov chain generates samples of state vectors (as opposed to scalar values). In addition, MCMC is innately independent of the initial state; as such, one obtains the same final distribution regardless of initial input values. Our algorithm produces a PDF (probability density fucntions) that is the joint probability of input prior constraints on corongraph-state variables and a "likelihood" function based on $t_{req}$ (see below). In general, MCMC is far more efficient in sampling large parameter space than uniform grid sampling because MCMC concentrates on exploration of interesting regions in parameter space, as specified by the prior and likelihood functions.     
 

We define our target PDF ($Pr$) as follows:  

$$
Pr(\mathbf{x}|\mathbf{t}) \equiv L(\mathbf{t}|\mathbf{x})p(\mathbf{x})
$$

Here, $\mathbf{x}$ is a vector of user-selected variables specifying the coronagraph state, where EXOSIMS defines the full list of possible variables (see https://exosims.readthedocs.io/en/latest/opticalsystem.html). The vector $\mathbf{t}$ comprises target integration-time values as a function of planet-star angular separation. The equation's right-hand side contains two PDFs;  $p(\mathbf{x})$ is the prior probability of coronagraph-state variables, and $L(\mathbf{t}|\mathbf{x})$ is the likelihood of $\mathbf{t}$ given $\mathbf{x}$.  The prior function, $p$, specify ranges of parameter values to be explored in the form of a probability density function for each parameter; it represents the user's knowledge or judgment of the feasible realm of coronagraph states. The likelihood function, $L$, specifies the distribution of integration-time values to be of interest in the given MCMC run. After burn-in, the Markov chain process produces a population of coronagraph states that are within the bounds specified the prior function *and* yield required integration times that are of interest, as specified by the likelihood function; i.e. the population represents the joint probability of $L$ and $p$.    

We provide an example below as a tutorial on how to use EBS's MCMC mode.

## Example   
In this example, we 

### Create necessary input files

### Configuration YAML File
* Create a YAML file in the <inputs> subdirectory with the recommended 
nomenclature <config_x.cfg> (replace "x" with your own string).
* This YAML file specifies parameter values that set up the run.  
* See <config_mcmc_example.cfg> as an example with instructive comments.  

## 
* Create a 
