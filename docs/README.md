# README

Welcome to NFT processing module  
It it written on python and uses modified versions of FNFT and FNFTpy.  
PJT use C++ to calculate discrete eigenvalues for NF spectrum and has python interface.
[FNFT](https://github.com/FastNFT/FNFT) is written in pure C and 
have python interface provided by [FNFTpy](https://github.com/xmhk/FNFTpy).


## Installation

Please follow the instructions in the file [INSTALL.md](INSTALL.md) to install all necessary libraries.  
Don't forget set proper path to FNFT library in FNFTpy function get_lib_path() in auxiliary.py. 

## Use

You can find some examples in [examples.md](examples.md)  

Module has three parts:
* NFT tools
* Signal processing tools  
* Notebooks

NFT part corresponds to the NFT analysis of the signal. It includes forward, inverse NFT, 
digital back propagation (DBP) tools and other NFT routines.  
Signal processing is about signal handling. It includes tools for cut signal for specific window mode etc.  
Notebooks contain some examples for NFT / signal processing.

First, you have to import modules.
```
import sys
# adding signal_handling to the system path
sys.path.insert(0, '../signal_handling/')
sys.path.insert(0, '../nft_handling/')

import signal_generation as sg
import signal_processing as sp
import ssfm
import nft_analyse as nft
```

If you want use FNFTpy directly, you have to import it:
```
import FNFTpy as fpy
```

To see examples, read [examples.md](examples.md)  