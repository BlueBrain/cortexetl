> [!WARNING]
> The Blue Brain Project concluded in December 2024, so development has ceased under the BlueBrain GitHub organization.
> Future development will take place at: https://github.com/openbraininstitute/cortexetl

# CortexETL

CortexETL is a repository of analyses for characterising and calibrating simulations of in silico cortical models. CortexETL uses [BlueETL](https://bbpgitlab.epfl.ch/nse/blueetl) for data processing and data handling, and uses functionality from BarrelETL [https://bbpgitlab.epfl.ch/circuits/personal/teska/bc-simulation-analysis] for analysis of evoked responses.  

The code was originally written for the calibration and characterization of activity in the Blue Brain Project's model of the rat non-barrel primary somatosensory model. These analyses are described in the preprint: [Modeling and Simulation of Neocortical Micro- and Mesocircuitry. Part II: Physiology and Experimentation](https://www.biorxiv.org/content/10.1101/2023.05.17.541168v3)

The codebase is is currently being refined for use in other contexts. This will include improved integration with BarrelETL beyond the current solution of including copies of files from the repository here.

Please contanct James Isbister for questions / advice on using.


## Getting started

After cloning the repository and creating a virtual environment (currently tested with Python 3.10.8) install the requirements using:

```
pip install -r requirements.txt

```


## Example notebooks

A number of example notebooks are in development. These correspond to examples of [BBP Workflow](https://bbpteam.epfl.ch/project/spaces/display/BBPNSE/Workflow) calibration and calibrated simulation campaigns in [SSCx-Workflows](https://bbpgitlab.epfl.ch/conn/configs/simulations/bbp-workflow/sscx-workflows/-/tree/master/spont_evoked_calibration_and_calibrated/examples).

These include:
- Characterising the effect of OU mean and std on unconnected FRs of different layerwise populations: 
	[notebooks/examples/O1/calibration/stage_1_unconnected_scan/unconnected_fr_analysis.ipynb](https://bbpgitlab.epfl.ch/conn/personal/isbister/cortex_etl/-/tree/main/notebooks/examples/O1/calibration/stage_1_unconnected_scan)
- Characterising spiking activity during or after calibration:<br />
	[notebooks/examples/O1/calibration/stage_2_or_3_connection/measure_spontaneous_actvity_and_connection.ipynb](https://bbpgitlab.epfl.ch/conn/personal/isbister/cortex_etl/-/tree/main/notebooks/examples/O1/calibration/stage_2_or_3_connection)


## Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright © 2024 Blue Brain Project/EPFL



<!-- ## Name

## Description

## Visuals

## Installation

## Usage

## Support

## Roadmap


## Contributing


## Authors and acknowledgment


## License


## Project status -->
