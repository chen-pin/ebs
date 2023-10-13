# The Error Budget Software (EBS)  
The Error Budget Software (EBS) package is an error-budgeting toolkit to aid the user in exploring combinations of technology performance parameters that can enable observations of Earth-size planets in the habitable zones of nearby stars ("exoEarths") using a telescope combined with a coronagraph instrument.  EBS accepts key parameters of a space observatory and coronagraph instrument that impact the system's ability to detect exoEarths, and it returns the required exposure time to reach user-specified SNR for the given target star; since any mission has a limited total amount of observation time, one can think of exposure time per target as a key cost quantity on mission science return.  EBS also returns count rates of signal and noise sources.  The computation uses the methodology described in Nemati et al. [DOI: 10.1117/1.JATIS.6.3.039002], which involves allocating flux-ratio noise among different system elements. Written in Python, EBS is essentially a wrapper around the open-source EXOSIMS package (https://exosims.readthedocs.io/en/latest/).  EBS enables the user to input parameters specifying dynamical wavefront aberrations (from the observatory to the coronagraph instrument) and parameters representing performance of a wavefront-sensing-and-control (WFS&C) subsystem; these are EBS's distinctive features.  
## For Developers
* Please use Git branching for code development to keep the main branch in a working state (ref. [Git Feature-Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow)).
  * If you would like to push branches to the repository, please request to become a collaborator.      

## Getting Started 
### Setting up a conda environment 

from the root directory in a terminal run

`conda env create -f environment.yml`

activate the environment by running

`conda activate ebs`

you should now have access to all of the dependencies necessary to run the EBS package, including EXOSIMS. 
### Installing EBS
From the command line in the root directory run

`python setup.py install`

### Running EBS from the command line 

EBS is run by formatting and generating input JSON files that are processed by EXOSIMS to yield exposure times for a 
variety of observing scenarios. These parameters are split between a `parameters.yml` file and .csv files containing the 
wavefront error, wavefront sensing and control factors, sensitivity factors, throughputs, and contrasts. Example files 
can be found and modified in the `inputs` folder.

It should be noted that any parameter which is not specified in the input JSON file that EBS passes to EXOSIMS will 
automatically receive the default EXOSIMS values and so users should familiarize themselves with all relevant EXOSIMS
variables or potentially receive unexpected results. 

To run ebs from the command line, users should first ensure that all of the previously described parameter files contain 
the desired values. Command line usage for parameter searches is supported for all EXOSIMS variables that can be
found in the pp_json file, as well as `contrast` and `throughput`. EXOSIMS parameters should be specified by both the 
subsystem they are a part of, as well as the parameter name (e.g. `scienceInstrument` and `QE`). The values that will 
be swept over should be specified in the `parameters.yml` file under `iter_paramaters`. All other values will be kept 
fixed. Performing sweeps over multiple parameters simultaneously will be added in a future version update.  

For example, to sweep over contrast you would use the following syntax:

`run_ebs <subsystem> <parameter> -c <path/to/config.yml>`

e.g. 

`run_ebs scienceInstruments QE -c inputs/parameters.yml`

`contrast` and `throughput` are special cases where these should just be given as the `subsystem` parameter with the 
second field left blank, e.g. 

`run_ebs contrast -c inputs/parameters.yml`

When completed this will display a plot of the expected exposure times for each of the observing scenarios in the 
`parameters.yml` as a function of the selected `iter_parameter`.



# Legal Notices
Copyright (c) 2023-24 California Institute of Technology (“Caltech”). U.S. Government
sponsorship acknowledged.  
All rights reserved.  
Redistribution and use in source and binary forms, with or without modification, are permitted provided
that the following conditions are met:  
• Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.  
• Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.  
• Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory, nor the
names of its contributors may be used to endorse or promote products derived from this software
without specific prior written permission.  
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
