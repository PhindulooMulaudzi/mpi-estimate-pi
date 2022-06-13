/* File:     mpi_trapezoid_1.c
 * Purpose:  Use MPI to implement a parallel version of the trapezoidal
 *           rule.
 *
 * Input:
 * Output:   Estimate of the integral from a to b of f(x)
 *           using the trapezoidal rule and n trapezoids.
 *
 * Compile:  mpicc -g -Wall -o mpi_trapezoid_1 mpi_trapezoid_1.c
 * Run:      mpiexec -n <number of processes> ./mpi_trapezoid_1 \
 *                <starting point> <end point> <no of trpezoids>
 *           such as mpiexec -n 4 ./mpi_trapezoid_1 0.0 1.0 1000000
 * Algorithm:
 *    1.  Each process calculates "its" interval of integration.
 *    2.  Each process estimates the integral of f(x)
 *        over its interval using the trapezoidal rule.
 *    3a. Each process != 0 sends its integral to 0.
 *    3b. Process 0 sums the calculations received from
 *        the individual processes and prints the result.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void usage(char prog_name[]);

int main(int argc, char* argv[]) {
	int currentProcess, nprocesses, n = 1000, local_n;
	double a = 0.0, b = 1.0, h, local_a, local_b;
	double local_sum =0.0, global_sum =0.0;
	double start, end;
	int source;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &currentProcess);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocesses);

	start = MPI_Wtime();
		
	// determine value of h
	h = (b-a)/n; 
	
	// divide the work into chunks for each process
	// this translate to the number of trapezoids each process handles
	local_n = n/nprocesses; 
	
	// this essentially serves as the index that tell each process where its chunk starts
	// this is given as part of the algorithm's pseudo code
	local_a = a + currentProcess*local_n*h;
	local_b = local_a + local_n*h;
	
	// we start the main function to compute the trapezoid
	local_sum = (f(local_a) + f(local_b))/2.0;
	double x = 0;
	for(int i = 1; i <= local_n - 1; i++) {
		x = local_a + i*h;
		local_sum += f(x);
	}
	local_sum = local_sum*h;
	// end of trapezoid computation
	
	// Perform reduction sum on all the local sums
	// Can also use Send and Receive, or use a Gather to send to a buffer then sum it
	MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
		 MPI_COMM_WORLD);
	
	end = MPI_Wtime();
	if(currentProcess == 0) {
		printf("With n = %d trapezoids, our local_sum of ", n);
		printf("the integral\n from %f to %f = %.15e ", a, b, global_sum);
		printf("in %.6fs\n", (end-start));
	}
	/* Shut down MPI */
	MPI_Finalize();
	return 0;
}/* main */

double f(double x){
	return 4/(1+x*x);
}

void usage(char prog_name[]) {
   fprintf(stderr, "usage: %s <left point> <right point> <number of tapezoids>\n",
         prog_name);
   exit(0);
} /* Usage */
