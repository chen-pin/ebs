# The Error Budget Software (EBS)  
The Error Budget Software (EBS) package is an error-budgeting toolkit to aid the user in exploring combinations of technology performance parameters that can enable observations of Earth-size planets in the habitable zones of nearby stars ("exoEarths") using a telescope combined with a coronagraph instrument.  EBS accepts key parameters of a space observatory and coronagraph instrument that impact the system's ability to detect exoEarths, and it returns the required exposure time to reach user-specified SNR for the given target star; since any mission has a limited total amount of observation time, one can think of exposure time per target as a key cost quantity on mission science return.  EBS also returns count rates of signal and noise sources.  The computation uses the methodology described in Nemati et al. [DOI: 10.1117/1.JATIS.6.3.039002], which involves allocating flux-ratio noise among different system elements. Written in Python, EBS is essentially a wrapper around the open-source EXOSIMS package (https://exosims.readthedocs.io/en/latest/).  EBS enables the user to input parameters specifying dynamical wavefront aberrations (from the observatory to the coronagraph instrument) and parameters representing performance of a wavefront-sensing-and-control (WFS&C) subsystem; these are FREAK's distinctive features.  
## For Developers
* Please use Git branching for code development to keep the main branch in a working state (ref. [Git Feature-Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow)).  

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
