Complete the table below with your results, and then provide your interpretation at the end.

Note that:

- When calculating the parallel speed-up S, use the time output by the code, which corresponds
  to the parallel calculation and does not include reading in the file or performing the serial check.

- Take as the serial execution time the time output by the code when run with a single process
  (hence the speed-up for 1 process must be 1.0, as already filled in the table).


No. Process:                        Mean time (average of 3 runs)           Parallel speed-up, S:
===========                         ============================:           ====================
1                                   0.000249863                             1.0
2                                   0.000287533                             0.869 
4                                   0.000402451                             0.621

Architecture that the timing runs were performed on: 
MPICH on the University of Leeds DEC-10 machines -> accessed via ssh. Timing runs completed using "mpiexec".


A brief interpretation of these results (2-3 sentences should be enough):
The mean time increases proportionally with the number of processors being used, which results in a corresponding decrease in the Parallel speed-up, 
S. The result doesn't match the expected theoretical outcome where a propotional increase in parallel speed-up is preferred: this phenomenon has
most likely been caused by the usage of an equivalent number of serial processes (like loops) in parallel, as would be in the optimal serial calculation. 
Moreover, even though instances of collective communication and reduction have been applied to this project, the parallel processes might be limited by 
the memory available: a technical parameter that is restricted by hardware.
