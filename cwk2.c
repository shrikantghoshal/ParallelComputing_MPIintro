//
//Made by Shrikant Ghoshal (el18s2g) - 201265711
//For Coursework 2 of University of Leeds teaching module Parallel Computing (COMP3221) 
//
//Submitted on 22/03/2021
//
//Uses pre-included header file cwk2_extra.h -> to be changed for assesment
//Requires an MPI library to compile and run.
//



//
// Includes.
//

// Standard includes.
#include <stdio.h>
#include <stdlib.h>

// The MPI library.
#include <mpi.h>

// Some extra routines for this coursework. DO NOT MODIFY OR REPLACE THESE ROUTINES,
// as this file will be replaced with a different version for assessment.
#include "cwk2_extra.h"


//
// Main.
//
int main( int argc, char **argv )
{
    int i, p;

    //
    // Initialisation.
    //

    // Initialise MPI and get the rank of this process, and the total number of processes.
    int rank, numProcs;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &numProcs );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank     );

    // Check that the number of processes is a power of 2, but <=256, so the data set, which is a multiple of 256 in length,
    // is also a multiple of the number of processes. If using OpenMPI, you may need to add the argument '--oversubscribe'
    // when launnching the executable, to allow more processes than you have cores.
    if( (numProcs&(numProcs-1))!=0 || numProcs>256 )
    {
        // Only display the error message from one processes, but finalise and quit all of them.
        if( rank==0 ) printf( "ERROR: Launch with a number of processes that is a power of 2 (i.e. 2, 4, 8, ...) and <=256.\n" );

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Load the full data set onto rank 0.
    float *globalData = NULL;
    int globalSize = 0;
    if( rank==0 )
    {
        globalData = readDataFromFile( &globalSize );           // globalData must be free'd on rank 0 before quitting.
        if( globalData==NULL )
        {
            MPI_Finalize();                                     // Should really communicate to all other processes that they need to quit as well ...
            return EXIT_FAILURE;
        }

        printf( "Rank 0: Read in data set with %d floats.\n", globalSize );
    }

    // Calculate the number of floats per process. Note that only rank 0 has the correct value of localSize
    // at this point in the code. This will somehow need to be communicated to all other processes. Note also
    // that we can assume that globalSize is a multiple of numProcs.
    int localSize = globalSize / numProcs;          // = 0 at this point of the code for all processes except rank 0.

    // Start the timing now, after the data has been loaded (will only output on rank 0).
    double startTime = MPI_Wtime();

//=============================Coursework input start=============================//
    

    //
    // Task 1: Calculate the mean using all available processes.
    //

    //Broadcast the localSize variable to all processes using MPI_Bcast()
	if( rank==0 )
	{
		localSize = globalSize / numProcs;
        MPI_Bcast( &localSize , 1 , MPI_INT , 0, MPI_COMM_WORLD);
	}
	else
	{
        MPI_Bcast( &localSize , 1 , MPI_INT , 0, MPI_COMM_WORLD);
	}

    //Error reporting for a failure to allocate memory locally
    float *localData= (float*) malloc( localSize*sizeof(float) );
    if( !localData )
	{
		printf( "Could not allocate memory for the local data array on rank %d.\n", rank );
		MPI_Finalize();
		return EXIT_FAILURE;
	}

    //Distribute globalData to localData on all processes using MPI_Scatter()
    MPI_Scatter( globalData , localSize , MPI_FLOAT , localData , localSize , MPI_FLOAT , 0 , MPI_COMM_WORLD);
    
    //Local calculation - Mean -> Calculate local mean by adding all elements in a localData and dividing by the localSize.
    float localMean=0, localSum=0;
    for(p=0; p<localSize; p++)
    {
        localSum+=localData[p];
    }
    localMean = localSum/localSize;

    //Accumulate all local mean calculations to root 0 and store in an array using MPI_Gather()
    float globalMean = 0, globalMeanSum = 0;
    float globalMeanArray[numProcs-1];
	MPI_Gather( &localMean, 1, MPI_FLOAT , globalMeanArray , 1, MPI_FLOAT , 0 , MPI_COMM_WORLD);

    //Global calculation - Mean -> Add all of the locally calculated mean values and divide by the number of processes
    for( p=0; p<numProcs; p++ )
	{
		globalMeanSum+=globalMeanArray[p];
	}
    globalMean = globalMeanSum/numProcs;

//-----------------------------Task 1 END-----------------------------//
    
    //
    // Task 2. Calculate the variance using all processes.
    //

    //Broadcast the previously calculated globalMean to all processes using MPI_Bcast()
    if( rank==0 )
	{
        MPI_Bcast( &globalMean , 1 , MPI_FLOAT , 0, MPI_COMM_WORLD);
	}
	else
	{
        MPI_Bcast( &globalMean , 1 , MPI_FLOAT , 0, MPI_COMM_WORLD);
	}
    //Note: This process uses the global variable globalMean which was initialised with a value of 0 and later calculated in Task 1.

    //Local calculation - Variance: STEP 1 -> Calculating the values of the square of (difference between one instance of data and the total mean value), and adding them up for each process
    float localSumSq = 0;
    for(p=0;p<localSize; p++)
    {
        localSumSq += (localData[p] - globalMean)*(localData[p] - globalMean);
    }
    
    //Global Calculation - Variance: STEP 2 -> Reduction of all the locally calculated values to root 0 using MPI_Reduce() - operation: MPI_SUM
    float globalVariance = 0, globalSumSq = 0;
    MPI_Reduce( &localSumSq , &globalSumSq , 1 , MPI_FLOAT , MPI_SUM , 0 , MPI_COMM_WORLD);
    globalVariance = globalSumSq/globalSize;

//-----------------------------Task 2 END-----------------------------//

    //
    // Output the results alongside a serial check.
    //
    if( rank==0 )
    {
        // Output the results of the timing now, before moving onto other calculations.
        printf( "Total time taken: %g s\n", MPI_Wtime() - startTime );

        // Your code MUST call this function after the mean and variance have been calculated using your parallel algorithms.
        // Do not modify the function itself (which is defined in 'cwk2_extra.h'), as it will be replaced with a different
        // version for the purpose of assessing. Also, don't just put the values from serial calculations here or you will lose marks.
        finalMeanAndVariance( globalMean, globalVariance);
            // You should replace the first argument with your mean, and the second with your variance.

        // Check the answers against the serial calculations. This also demonstrates how to perform the calculations
        // in serial, which you may find useful. Note that small differences in floating point calculations between
        // equivalent parallel and serial codes are possible, as explained in Lecture 11.

        // Mean.
        float sum = 0.0;
        for( i=0; i<globalSize; i++ ) sum += globalData[i];
        float mean = sum / globalSize;

        // Variance.
        float sumSqrd = 0.0;
        for( i=0; i<globalSize; i++ ) sumSqrd += ( globalData[i]-mean )*( globalData[i]-mean );
        float variance = sumSqrd / globalSize;

        printf( "SERIAL CHECK: Mean=%g and Variance=%g.\n", mean, variance );

   }

    //
    // Free all resources (including any memory you have dynamically allocated), then quit.
    //
    if( rank==0 ) free( globalData );

    free( localData );
//=============================Coursework input finish=============================//
    MPI_Finalize();

    return EXIT_SUCCESS;
}
