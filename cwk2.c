//
// Starting code for the MPI coursework.
//
// See lectures and/or the worksheet corresponding to this part of the module for instructions
// on how to build and launch MPI programs. A simple makefile has also been included (usage optional).
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

    
	if( rank==0 )
	{
		localSize = globalSize / numProcs;

		// Note &localSize looks to the MPI function like an array of size 1.
		for( p=1; p<numProcs; p++ )
			MPI_Send( &localSize, 1, MPI_INT, p ,0, MPI_COMM_WORLD );
	}
	else
	{
		MPI_Recv( &localSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
	}

    //
    // Task 1: Calculate the mean using all available processes.
    //
    
    //send localSize to all ranks to allocate appropriate memory
    float *localData= (float*) malloc( localSize*sizeof(float) );
    if( !localData )
	{
		printf( "Could not allocate memory for the local data array on rank %d.\n", rank );
		MPI_Finalize();
		return EXIT_FAILURE;
	}
    
    //send localData to ranks
    //Collective communication
    MPI_Scatter( globalData , localSize , MPI_FLOAT , localData , localSize , MPI_FLOAT , 0 , MPI_COMM_WORLD);
    
    float localMean=0, localSum=0;
    for(p=0; p<localSize; p++)
    {
        localSum+=localData[p];
    }
    localMean = localSum/localSize;
    
    float globalMean = 0, globalMeanSum = 0;
    float globalMeanArray[numProcs-1];
	MPI_Gather( &localMean, 1, MPI_FLOAT , globalMeanArray , 1, MPI_FLOAT , 0 , MPI_COMM_WORLD);

    for( p=0; p<numProcs; p++ )
	{
		globalMeanSum+=globalMeanArray[p];
	}

    globalMean = globalMeanSum/numProcs;


   // //Non-collective communication - Sending data to all ranks for Mean
    // if( rank==0 )
	// {
        
		// // Copy first segment to own localData (nb. never 'send' to self!)
		// for( i=0; i<localSize; i++ ) localData[i] = globalData[i];

		// // Send the remaining segments.
		// for( p=1; p<numProcs; p++ )
		// 	MPI_Send( &globalData[p*localSize], localSize, MPI_INT, p, 0, MPI_COMM_WORLD );
	// }
	// else
	// {
	// 	MPI_Recv( localData, localSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
	// }

    //Non-collective communication - combining data from all ranks for Mean

    // if( rank==0 )
	// {
	// 	// Start the running total with rank 0's count.
	// 	globalMeanSum = localMean;

	// 	// Now add on all of the counts from the other processes.
	// 	for( p=1; p<numProcs; p++ )
	// 	{
	// 		float next;
	// 		MPI_Recv( &next, 1, MPI_FLOAT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
	// 		globalMeanSum += next;
	// 	}
    //     globalMean = globalMeanSum/numProcs;
	// }
	// else
	// {
	// 	MPI_Send( &localMean, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD );
	// }

//=============================Task 1 END=============================//

    //
    // Task 2. Calculate the variance using all processes.
    //

    if( rank==0 )
	{
        	// Note &localSize looks to the MPI function like an array of size 1.
		for( p=1; p<numProcs; p++ )
			MPI_Send( &globalMean, 1, MPI_FLOAT, p ,0, MPI_COMM_WORLD );
	}
	else
	{
		MPI_Recv( &globalMean, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
	}

    float localSumSq = 0;
    for(p=0;p<localSize; p++)
    {
        localSumSq += (localData[p] - globalMean)*(localData[p] - globalMean);
    }
    
    float globalVariance = 0, globalSumSq = 0;

    MPI_Reduce( &localSumSq , &globalSumSq , 1 , MPI_FLOAT , MPI_SUM , 0 , MPI_COMM_WORLD);
    globalVariance = globalSumSq/globalSize;
    
//check binary tree method

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

    MPI_Finalize();

    return EXIT_SUCCESS;
}
