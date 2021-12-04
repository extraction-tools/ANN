/*
Grid Search - Local Fit
Author: Nathan Snyder
*/



#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "TVA1_UU.h"


#define NUM_INDEXES 45
#define NUM_REPLICAS 1000



static double k, QQ, x, t,
       F[NUM_INDEXES], errF[NUM_INDEXES],
       F1, F2,
       phi[NUM_INDEXES],
       dvcs,
       ReH_mean, ReE_mean, ReHtilde_mean,
       ReH_stddev, ReE_stddev, ReHtilde_stddev;

static int desiredSet;

static TVA1_UU tva1_uu;



// Read in the data
// Returns 0 if success, 1 if error
static bool readInData(const char * const restrict filename);

// Box-Muller Transform --> Turn a uniform distribution into a Gaussian distribution
static double boxMuller(void);

// Calculate the root mean squared percent error in F for a specific replica and the given CFFs
static double calcFError(const double replicas[], const double ReH, const double ReE, const double ReHtilde);

// Estimate the correct CFF values for a specific replica
static void calcCFFs(void);

// Fits CFFs to all of the replicas
static void localFit(void);



int main(int argc, char **argv) {
    srand((unsigned int) time(0));

    if (argc == 3) {
        desiredSet = atoi(argv[2]);

        if(readInData(argv[1])) {
            printf("Read failed.\n");
            return 1;
        }
    } else {
        printf("Please specify a data file (.csv) and a set number only.\n");
        return 1;
    }

    TVA1_UU_Init(&tva1_uu, QQ, x, t, k, F1, F2, (double *) &phi, 45);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    localFit();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    const long double timeLocalFitTook = (long double) (end.tv_sec - start.tv_sec) + ((long double) (end.tv_nsec - start.tv_nsec) / (long double) 1.0e9);

    FILE *f = fopen("local_fit_output.csv", "r");

    if (f == NULL) {
        f = fopen("local_fit_output.csv", "w");
        fprintf(f, "Set,k,QQ,x,t,ReH_mean,ReH_stddev,ReE_mean,ReE_stddev,ReHtilde_mean,ReHtilde_stddev,time\n");
    } else {
        fclose(f);
        f = fopen("local_fit_output.csv", "a");
    }

    fprintf(f, "%d,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf,%.3Lf\n", desiredSet, k, QQ, x, t, ReH_mean, ReH_stddev, ReE_mean, ReE_stddev, ReHtilde_mean, ReHtilde_stddev, timeLocalFitTook);

    fclose(f);

    TVA1_UU_Destruct(&tva1_uu);

    return 0;
}



// Read in the data
// returns 0 if success, 1 if error
static bool readInData(const char * const restrict filename) {
    FILE *f = fopen(filename, "r");
    char buff[1024] = {0};
    bool previouslyEncounteredSet = false;

    if (f == NULL) {
        printf("Error: could not open file %s.\n", filename);
        return 1;
    }

    fgets(buff, 1024, f);

    while (fgets(buff, 1024, f)) {
        int set;
        sscanf(buff, "%d,%*s\n", &set);

        if (set == desiredSet) {
            int index;
            sscanf(buff, "%*d,%d,%*s\n", &index);

            if (!previouslyEncounteredSet) {
                sscanf(buff, "%*d,%*d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%*lf,%lf,%lf,%lf\n",
                       &k, &QQ, &x, &t,
                       &phi[index], &F[index], &errF[index],
                       &F1, &F2, &dvcs);

                previouslyEncounteredSet = 1;
            } else {
                sscanf(buff, "%*d,%*d,%*lf,%*lf,%*lf,%*lf,%lf,%lf,%lf,%*lf,%*lf,%*lf,%*lf\n",
                       &phi[index], &F[index], &errF[index]);
            }
        }
    }

    fclose(f);

    if (!previouslyEncounteredSet) {
        printf("Error: could not find set %d in %s.\n", desiredSet, filename);
        return 1;
    }

    return 0;
}



// Box-Muller Transform --> Turn a uniform distribution into a Gaussian distribution
// Returns a random number following a Gaussian distribution with mean=0 and stddev=1
static double boxMuller(void) {
    // random numbers in the range [0, 1) following a uniform distribution
    const double rand1 = rand() / (double) RAND_MAX;
    const double rand2 = rand() / (double) RAND_MAX;

    return sqrt(-2 * log(rand1)) * sin(2 * M_PI * rand2);
}



// Calculate the root root mean squared percent error in F for a specific replica and the given CFFs
static double calcFError(const double replicas[], const double ReH, const double ReE, const double ReHtilde) {
    double sum_of_squared_percent_errors = 0.0;

    for (int index = 0; index < NUM_INDEXES; index++) {
        for (int replica = 0; replica < NUM_REPLICAS; replica++) {
            const double F_predicted = dvcs + TVA1_UU_getBHUU_plus_getIUU(&tva1_uu, phi[index], ReH, ReE, ReHtilde);
            const double F_actual = F[index] + (replicas[(index * NUM_REPLICAS) + replica] * errF[index]);
            const double percent_error = (F_actual - F_predicted) / F_actual;
            sum_of_squared_percent_errors += percent_error * percent_error;
        }
    }

    return sqrt(sum_of_squared_percent_errors / (NUM_INDEXES * NUM_REPLICAS));
}



// Estimate the correct CFF values for a specific replica
static void calcCFFs() {
    double ReHguess = 0.;
    double ReEguess = 0.;
    double ReHtildeguess = 0.;
    double bestError = DBL_MAX;

    // replicas: a list of replicas, where each replica represents the number of standard deviations that an index's F value will be off by
    double replicas[NUM_INDEXES * NUM_REPLICAS];
    for (int i = 0; i < NUM_INDEXES * NUM_REPLICAS; i++) {
        replicas[i] = boxMuller();
    }

    // num: number of points tested on either side of the guess
    const double num = 10.;

    // totalDist: the total distance being tested on either side of the guess
    const double totalDist = 1.;



    // dist: distance between each point being tested
    for (double dist = totalDist / num; dist >= 0.000001; dist /= num) {
        const double maxChange = dist * num;
        const double minChange = -1 * maxChange;

        double bestReHguess = 0.0;
        double bestReEguess = 0.0;
        double bestReHtildeguess = 0.0;

        double prevError = DBL_MAX;
        bestError = DBL_MAX;


        while ((prevError - bestError > 0.000001) || (DBL_MAX - bestError < 0.000001)) {
            prevError = bestError;

            for (double ReHchange = minChange; ReHchange <= maxChange; ReHchange += dist) {
                for (double ReEchange = minChange; ReEchange <= maxChange; ReEchange += dist) {
                    for (double ReHtildechange = minChange; ReHtildechange <= maxChange; ReHtildechange += dist) {
                        double error = calcFError(replicas, ReHguess + ReHchange, ReEguess + ReEchange, ReHtildeguess + ReHtildechange);

                        if (error < bestError) {
                            bestReHguess = ReHguess + ReHchange;
                            bestReEguess = ReEguess + ReEchange;
                            bestReHtildeguess = ReHtildeguess + ReHtildechange;
                            bestError = error;
                        }
                    }
                }
            }

            ReHguess = bestReHguess;
            ReEguess = bestReEguess;
            ReHtildeguess = bestReHtildeguess;
        }
    }



    ReH_mean = ReHguess;
    ReE_mean = ReEguess;
    ReHtilde_mean = ReHtildeguess;

    ReH_stddev = ReHguess * bestError;
    ReE_stddev = ReEguess * bestError;
    ReHtilde_stddev = ReHtildeguess * bestError;
}



// Fits CFFs to all of the replicas
static void localFit(void) {
    calcCFFs();
}
