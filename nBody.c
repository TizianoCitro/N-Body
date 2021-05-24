#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define SOFTENING 1e-9f
#define DT 0.01f
#define SEED 42
#define MASTER 0

#define CORRECTLY_INVOKED 1
#define NOT_CORRECTLY_INVOKED 0
#define EXPECTED_ARGUMENT 4
#define PRINT_ARGUMENT 1
#define NUMBER_OF_BODIES_ARGUMENT 2
#define NUMBER_OF_ITERATIONS_ARGUMENT 3

#define BODY_FLOAT 6
#define BODY_NO_DIFFERENCE 0

#define PRINT_REQUIRED 1
#define PRINT_NOT_REQUIRED 0
#define EXECUTION_TIME_REQUIRED 1
#define EXECUTION_TIME_NOT_REQUIRED 0
#define NO_EXECUTION_TIME 0

// Body structure definition
typedef struct { 
    float x, y, z, vx, vy, vz; 
} Body;

// Functions definition
void randomizeBodies(float *bodies, int numberOfBodies);
void bodyForce(Body *bodies, float dt, int dependentStart, int dependentStop, int independentStart, int independentStop);
void updatePositions(Body *bodies, float dt, int start, int stop);
void buildBodiesPerProcessAndDispls(int numberOfBodies, int numberOfTasks, int *bodiesPerProcess, int *displs);
void printTimeAndBodies(Body *bodies, int numberOfBodies, int numberOfTasks, int iterations, double executionTime, int isExecutionTimeRequired, int isPrintRequired);
void printBodies(Body *bodies, int numberOfBodies, int numberOfTasks, int iterations, int isEnd);
void printHowToUse();

int main(int argc, char **argv) {

    // Initializes the MPI environment
    MPI_Init(NULL, NULL);

    int numberOfTasks, rank;

    // Gets the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfTasks);

    // Gets the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    /**
     * Handles input parameters.
     * Stops the execution if they're not provided properly.
     */
    int isCorrectlyInvoked = CORRECTLY_INVOKED;
    if (argc != EXPECTED_ARGUMENT) isCorrectlyInvoked = NOT_CORRECTLY_INVOKED;

    int isPrintRequired;
    if (isCorrectlyInvoked != NOT_CORRECTLY_INVOKED) {  
        if (strcmp(argv[PRINT_ARGUMENT], "-pY") == 0) isPrintRequired = PRINT_REQUIRED;
        else if (strcmp(argv[PRINT_ARGUMENT], "-pN") == 0) isPrintRequired = PRINT_NOT_REQUIRED;
        else isCorrectlyInvoked = NOT_CORRECTLY_INVOKED;
    }

    if (isCorrectlyInvoked == NOT_CORRECTLY_INVOKED) {
        if (rank == MASTER) printHowToUse();

        // Finalizes the MPI environment in case of error
        MPI_Finalize();

        return 0;
    }

    int numberOfBodies = atoi(argv[NUMBER_OF_BODIES_ARGUMENT]);
    int iterations = atoi(argv[NUMBER_OF_ITERATIONS_ARGUMENT]);
    srand(SEED);

    // Initializes bodies at random
    int bytes = numberOfBodies * sizeof(Body);
    float *buffer = (float*) malloc(bytes);
    Body *bodies = (Body*) buffer;
    if (rank == MASTER) {
        randomizeBodies(buffer, BODY_FLOAT * numberOfBodies);
        
        printTimeAndBodies(bodies, numberOfBodies, numberOfTasks, iterations, NO_EXECUTION_TIME, EXECUTION_TIME_NOT_REQUIRED, isPrintRequired);
    }

    // Creates the custom MPI datatype for bodies
    MPI_Datatype MPI_BODY; 
    MPI_Datatype oldTypes[1] = {MPI_FLOAT};
    int blocksCount[1] = {BODY_FLOAT};
    MPI_Aint offset[1] = {0};
    
    MPI_Type_create_struct(1, blocksCount, offset, oldTypes, &MPI_BODY);
    MPI_Type_commit(&MPI_BODY);

    /**
     * Contains the number of bodies to send to each of the processes
     * and the displacements where each segment begins respectively.
     * So, displs is about where to start and bodiesPerProcess is
     * about how far you have to go.
     */
    int *bodiesPerProcess = (int*) malloc(numberOfTasks * sizeof(int));
    int *displs = (int*) malloc(numberOfTasks * sizeof(int));
    buildBodiesPerProcessAndDispls(numberOfBodies, numberOfTasks, bodiesPerProcess, displs);

    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Request requests[numberOfTasks];
    MPI_Status status;

    MPI_Barrier(MPI_COMM_WORLD);
    int startTime = MPI_Wtime();

    // Master sends a portion of the bodies to each slave
    MPI_Scatterv(
        bodies, 
        bodiesPerProcess, 
        displs, 
        MPI_BODY,
        &bodies[displs[rank]], 
        bodiesPerProcess[rank], 
        MPI_BODY, 
        MASTER,
        MPI_COMM_WORLD);

    // Computes for the number of requested iterations
    for (int iteration = 0; iteration < iterations; iteration++) {
        for (int process = MASTER; process < numberOfTasks; process++)
            MPI_Ibcast(&bodies[displs[process]], bodiesPerProcess[process], MPI_BODY, process, MPI_COMM_WORLD, &requests[process]);
        
        // Each of the process computes its own part of the bodies
        int independentStop = displs[rank] + bodiesPerProcess[rank];
        bodyForce(bodies, DT, displs[rank], independentStop, displs[rank], independentStop);

        for (int waitedProcess = MASTER; waitedProcess < numberOfTasks; waitedProcess++) {
            if (waitedProcess != rank) {
                // Waits for the process with the same rank as waitedProcess
                MPI_Wait(&requests[waitedProcess], &status);
                
                /**
                 * Computes on its own particles compared with the particles 
                 * of the process with the same rank as waitedProcess.
                 */
                int dependentStop = displs[waitedProcess] + bodiesPerProcess[waitedProcess];
                bodyForce(bodies, DT, displs[waitedProcess], dependentStop, displs[rank], independentStop);
            }
        }

        // At the it integrates the positions
        updatePositions(bodies, DT, displs[rank], independentStop);
    }  

    // Gathers all the computation from slaves to master
    MPI_Gatherv(
        &bodies[displs[rank]], 
        bodiesPerProcess[rank], 
        MPI_BODY, 
        bodies, 
        bodiesPerProcess, 
        displs, 
        MPI_BODY, 
        MASTER, 
        MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    int finishTime = MPI_Wtime();

    int executionTime = finishTime - startTime;
    if (rank == MASTER) 
        printTimeAndBodies(bodies, numberOfBodies, numberOfTasks, iterations, executionTime, EXECUTION_TIME_REQUIRED, isPrintRequired);

    free(bodiesPerProcess);
    free(displs);
    MPI_Type_free(&MPI_BODY);

    // Finalizes the MPI environment
    MPI_Finalize();

    return 0;
}

void randomizeBodies(float *bodies, int numberOfBodies) {
    for (int body = 0; body < numberOfBodies; body++) 
        bodies[body] = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
}

void bodyForce(Body *bodies, float dt, int dependentStart, int dependentStop, int independentStart, int independentStop) {
    for (int i = independentStart; i < independentStop; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = dependentStart; j < dependentStop; j++) {
            float dx = bodies[j].x - bodies[i].x;
            float dy = bodies[j].y - bodies[i].y;
            float dz = bodies[j].z - bodies[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1.0f / sqrt(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        bodies[i].vx += dt * Fx;
        bodies[i].vy += dt * Fy;
        bodies[i].vz += dt * Fz;
    }
}

void updatePositions(Body *bodies, float dt, int start, int stop) {
    for (int body = start; body < stop; body++) { 
        bodies[body].x += bodies[body].vx * dt;
        bodies[body].y += bodies[body].vy * dt;
        bodies[body].z += bodies[body].vz * dt;
    }
}

void buildBodiesPerProcessAndDispls(int numberOfBodies, int numberOfTasks, int *bodiesPerProcess, int *displs) {
    int rest = numberOfBodies % numberOfTasks;
    int bodiesDifference = numberOfBodies / numberOfTasks;
    int startPosition = 0;

    /**
     * It's based on the fact that the rest
     * is always less than the divisor.
     */
    for (int process = MASTER; process < numberOfTasks; process++) {
        if (rest > BODY_NO_DIFFERENCE) {
            bodiesPerProcess[process] = bodiesPerProcess[process] = bodiesDifference + 1;
            rest--;
        } else bodiesPerProcess[process] = bodiesDifference;

        displs[process] = startPosition;
        startPosition += bodiesPerProcess[process];
    }
}

void printTimeAndBodies(
    Body *bodies, 
    int numberOfBodies, 
    int numberOfTasks,
    int iterations,
    double executionTime, 
    int isExecutionTimeRequired, 
    int isPrintRequired) {
    // If execution time is required then it's the end of computation
    if (isPrintRequired == 1) printBodies(bodies, numberOfBodies, numberOfTasks, iterations, isExecutionTimeRequired);

    if (isExecutionTimeRequired == 1) {
        printf(
            "With %d processors, %d bodies and %d iterations the execution time is %0.2f seconds\n", 
            numberOfTasks, numberOfBodies, iterations, executionTime);

        FILE *file = fopen("./nBodyExecutionTime.txt", "a");
        fprintf(
            file, 
            "With %d processors, %d bodies and %d iterations the execution time is %0.2f seconds\n\n", 
            numberOfTasks, numberOfBodies, iterations, executionTime);
    }
}

void printBodies(Body *bodies, int numberOfBodies, int numberOfTasks, int iterations, int isEnd) {
    FILE *file = fopen("./bodies.txt", "a");

    if (isEnd == 1) fprintf(file, "Bodies at the end with %d processors and %d iterations:\n", numberOfTasks, iterations);
    else fprintf(file, "Bodies at the beginning with %d processors and %d iterations:\n", numberOfTasks, iterations);

    for(int body = 0; body < numberOfBodies; body++)
		fprintf(
            file, 
            "Body[%d][%f, %f, %f, %f, %f, %f]\n", 
            body,
			bodies[body].x, bodies[body].y, bodies[body].z,
			bodies[body].vx, bodies[body].vy, bodies[body].vz);

    fprintf(file, "\n");
}

void printHowToUse() {
    printf("To correctly launch nBody run: mpirun -np P nBody [-pY | -pN] B I\n");
    printf("---> Where P is the number of processors\n");
    printf("---> Where [-pY | -pN] is pY if you want the bodies to be printed, pN otherwise\n");
    printf("---> Where B is the number of bodies\n");
    printf("---> Where I is the number of iterations\n");
    printf("---> Try it with mpirun -np 1 nBody -pY 12 3\n");
}