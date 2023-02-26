# M82_Simulation_in_Athena
This code is written by Wang Tianrui. The main purpose of this code is to simulate the effect of different stellar feedback in M82 dwarf galaxy and to reproduce the outflow structure seen in observation.
The three main code we use are starburst99(for the purpose of generating stellar wind table), GRACKLE(for radiative cooling and heating), Athena++(hydro dynamics solver)
The three main feedback process is radiation, stellar wind and supernova.
In our code star formation and the evolution of each individual star cluster are carefully resolved, which requires a particle solver to be implemented to Athena++, currently we used our own particle solver built upon user defined meshblock data and user mpi communication.
At this stage most of the development has been done, we are currently preparing the article^_^.
==========2023/2/27==========
