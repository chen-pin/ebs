# Instructions on How to Use the MCMC Mode  

## Introduction  
EBS's MCMC mode was created to survey required integration time in high
dimensional spaces of coronagraphic parameters.  It use a Markov-chain Monte
Carlo (MCMC) algorithm (https://emcee.readthedocs.io) to explore the space of
the parameters of interest. Moreover, the mode utilizes a Bayesian approach to
incorporate prior constraints
on state parameters. A Markov chain has the property that, after running
sufficiently long (after a "burn in" period), the chain generates values representative of 
those drawn from
a target probability distribution. One can regard it as a random-number 
generator per user-specified PDF (probability density function). For
multi-variable PDFs, the Markov chain generates samples of state vectors
(as opposed to scalar values).
Another important property of Markov chain is that the current state depends
on only the immediately preceding state, and is therefore independent of the
initial state.

We cast the exploration in a Bayesian formalism. The Bayes equation can be
expressed as follows:  

\begin{equation}
Pr(\mathbf{x}|\mathbf{y}) \propto L(\mathbf{y}|\mathbf{x})p(\mathbf{x})
\end{equation}

For our application, $\mathbf{x}$ is a vector of user-selected variables
specifying the state of the observational system (e.g. throughput,
contrast, bandwidth).  These variables consitute a subset of all variables 
that determine the integration time required to achieve a user-specified SNR
(signal to noise ratio). The vector $\mathbf{y}$ represents required integration
times as a function of planet-star angular separation. The equation contains three PDFs:  $p(\mathbf{x})$ is the (user-defined) prior 
PDF, $L(\mathbf{y}|\mathbf{x})$ is the likelihood PDF of $\mathbf{y}$
conditioned on $\mathbf{x}$; and $Pr(\mathbf{x}|\mathbf{y})$ is the posterior
PDF of $\mathbf{x}$ conditioned on $\mathbf{y}$.   

      

It does so through the
following steps:  
1. Utilize a Markov-chain Monte Carlo (MCMC): algorithm to sample parameter
vectors that specify coronagraph states, computes integration times associated 

## Create necessary input files

### Configuration YAML File
* Create a YAML file in the <inputs> subdirectory with the recommended 
nomenclature <config_x.cfg> (replace "x" with your own string).
* This YAML file specifies parameter values that set up the run.  
* See <config_mcmc_example.cfg> as an example with instructive comments.  

## 
* Create a 
