<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Terms used in HIP Documentation](#terms-used-in-hip-documentation)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# Terms used in HIP Documentation

- host,  host cpu : Executes the HIP runtime API and is capable of initiating kernel launches to one or more devices.
- default device : Each host thread maintains a "default device".  
Most HIP runtime APIs (including memory allocation, copy commands, kernel launches) do not use accept an explicit device
argument but instead implicitly use the default device.
The default device can be set with hipSetDevice.

- "active host thread" - the thread which is running the HIP APIs.   

- completion_future becomes ready.  "Completes"

- hcc = Heterogeneous Compute Compiler (https://bitbucket.org/multicoreware/hcc/wiki/Home).  

- hipify - tool to convert CUDA(R) code to portable C++ code.
- hipconfig - tool to report various confoguration properties of the target platform.




