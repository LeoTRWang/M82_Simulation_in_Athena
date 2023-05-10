## M82_Simulation_in_Athena++
### Introduction
This code is written by Wang Tianrui. The main purpose of this code is to simulate the effect of different stellar feedback in M82 dwarf galaxy and to reproduce the outflow structure seen in observation.

### Features
In our study we present a new M82 mass model including
In our code star formation and the evolution of each individual star cluster are carefully resolved, which requires a particle solver to be implemented to Athena++, currently we used our own particle solver built upon user defined meshblock data and user mpi communication.

It will not be possible for us to simulate anything without helping hands, we used different codes to suit the need for simulation. The three major codes we use are
+ [Athena++](https://github.com/PrincetonUniversity/athena-public-version/wiki) main hydro dynamics solver
+ [Starburst99](https://www.stsci.edu/science/starburst99/docs/default.htm) for the purpose of generating stellar wind table
+ [GRACKLE](https://grackle.readthedocs.io/en/latest/) for radiative cooling and heating

The three main feedback processes included are
+ radiation, including radiation pressure as a momentum source and radiation heating as a heating term
+ stellar wind
+ supernova

### Current Progress
At this stage most of the development has been done, we are currently analysing data, including trying to reproduce X-ray image from hydrodynamics data and so much more. The article is under preparation and will be published soon in this year^_^

### Current Results

https://github.com/LeoTRWang/M82_Simulation_in_Athena/assets/87620687/efa334c7-97c9-48a9-9740-72a97dd0fb28

https://github.com/LeoTRWang/M82_Simulation_in_Athena/assets/87620687/bf34424a-1b62-48ca-a9cd-a4ee700f391e


https://github.com/LeoTRWang/M82_Simulation_in_Athena/assets/87620687/0c24fe30-2605-49e6-8788-57b9bb4e2876


