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
#define NUM_REPLICAS 100



static struct {
    double ReH, ReE, ReHtilde;
} cffGuesses[NUM_REPLICAS];

static double k, QQ, x, t,
       F[NUM_INDEXES], errF[NUM_INDEXES],
       F1, F2,
       dvcs,
       ReH_mean, ReE_mean, ReHtilde_mean,
       ReH_stddev, ReE_stddev, ReHtilde_stddev;

static int desiredSet, phi[NUM_INDEXES];

static TVA1_UU tva1_uu;



// Read in the data
// Returns 0 if success, 1 if error
static bool readInData(const char * const restrict filename);

// Box-Muller Transform --> Turn a uniform distribution into a Gaussian distribution
static double boxMuller(void);

// Function that calculates the mean and standard deviation of the CFFs over all the replicas
static void calcMeanAndStdDev(void);

// Calculate the root mean squared percent error in F for a specific replica and the given CFFs
static double calcFError(const double replicas[], const double ReH, const double ReE, const double ReHtilde);

// Estimate the correct CFF values for a specific replica
static void calcCFFs(const int replicaNum);

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

    TVA1_UU_Init(&tva1_uu, QQ, x, t, k, (int *) (&phi), 45);

    localFit();

    FILE *f = fopen("local_fit_output.csv", "r");

    if (f == NULL) {
        f = fopen("local_fit_output.csv", "w");
        fprintf(f, "Set,ReH_mean,ReH_stddev,ReE_mean,ReE_stddev,ReHtilde_mean,ReHtilde_stddev\n");
    } else {
        fclose(f);
        f = fopen("local_fit_output.csv", "a");
    }

    fprintf(f, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", desiredSet, k, QQ, x, t, ReH_mean, ReH_stddev, ReE_mean, ReE_stddev, ReHtilde_mean, ReHtilde_stddev);

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
                sscanf(buff, "%*d,%*d,%lf,%lf,%lf,%lf,%d,%lf,%lf,%*lf,%lf,%lf,%lf\n",
                       &k, &QQ, &x, &t,
                       &phi[index], &F[index], &errF[index],
                       &F1, &F2, &dvcs);

                previouslyEncounteredSet = 1;
            } else {
                sscanf(buff, "%*d,%*d,%*lf,%*lf,%*lf,%*lf,%d,%lf,%lf,%*lf,%*lf,%*lf,%*lf\n",
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



// Function that calculates the mean and standard deviation of the CFFs over all the replicas
static void calcMeanAndStdDev(void) {
    ReH_mean = 0.0;
    ReE_mean = 0.0;
    ReHtilde_mean = 0.0;

    double ReH_sum_of_differences = 0.0;
    double ReE_sum_of_differences = 0.0;
    double ReHtilde_sum_of_differences = 0.0;


    for (int replicaNum = 0; replicaNum < NUM_REPLICAS; replicaNum++) {
        ReH_mean += cffGuesses[replicaNum].ReH;
        ReE_mean += cffGuesses[replicaNum].ReE;
        ReHtilde_mean += cffGuesses[replicaNum].ReHtilde;
    }


    ReH_mean /= NUM_REPLICAS;
    ReE_mean /= NUM_REPLICAS;
    ReHtilde_mean /= NUM_REPLICAS;


    for (int replicaNum = 0; replicaNum < NUM_REPLICAS; replicaNum++) {
        const double ReH_difference = cffGuesses[replicaNum].ReH - ReH_mean;
        const double ReE_difference = cffGuesses[replicaNum].ReE - ReE_mean;
        const double ReHtilde_difference = cffGuesses[replicaNum].ReHtilde - ReHtilde_mean;

        ReH_sum_of_differences += ReH_difference * ReH_difference;
        ReE_sum_of_differences += ReE_difference * ReE_difference;
        ReHtilde_sum_of_differences += ReHtilde_difference * ReHtilde_difference;
    }


    ReH_stddev = sqrt(ReH_sum_of_differences / (NUM_REPLICAS - 1));
    ReE_stddev = sqrt(ReE_sum_of_differences / (NUM_REPLICAS - 1));
    ReHtilde_stddev = sqrt(ReHtilde_sum_of_differences / (NUM_REPLICAS - 1));
}



// Calculate the root root mean squared percent error in F for a specific replica and the given CFFs
static double calcFError(const double replicas[], const double ReH, const double ReE, const double ReHtilde) {
    double sum_of_squared_percent_errors = 0.0;

    for (int index = 0; index < NUM_INDEXES; index++) {
        const double F_predicted = dvcs + TVA1_UU_getBHUU_plus_getIUU(&tva1_uu, phi[index], F1, F2, ReH, ReE, ReHtilde);
        const double F_actual = F[index] + (replicas[index] * errF[index]);
        const double percent_error = (F_actual - F_predicted) / F_actual;
        sum_of_squared_percent_errors += percent_error * percent_error;
    }

    return sqrt(sqrt(sum_of_squared_percent_errors / NUM_INDEXES));
}



// Estimate the correct CFF values for a specific replica
static void calcCFFs(const int replicaNum) {
    double ReHguess = 0.0;
    double ReEguess = 0.0;
    double ReHtildeguess = 0.0;
    double bestError = DBL_MAX;

    // replicas: a list of replicas, where each replica represents the number of standard deviations that an index's F value will be off by
    double replicas[NUM_INDEXES];
    for (int i = 0; i < NUM_INDEXES; i++) {
        replicas[i] = boxMuller();
    }

    // num: number of points tested on either side of the guess
    const int num = 10;

    // totalDist: the total distance being tested on either side of the guess
    const double totalDist = 1.;



    // dist: distance between each point being tested
    for (double dist = (double) totalDist / num; dist >= 0.000000001; dist /= num) {
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



    cffGuesses[replicaNum].ReH = ReHguess;
    cffGuesses[replicaNum].ReE = ReEguess;
    cffGuesses[replicaNum].ReHtilde = ReHtildeguess;
}



// Fits CFFs to all of the replicas
static void localFit(void) {
    const double percentCompleteEachReplica = 100 / (double) NUM_REPLICAS;
    double percentComplete = 0.0;

    printf("Set %d: %.2lf%% complete\n", desiredSet, percentComplete);

    for (int replicaNum = 0; replicaNum < NUM_REPLICAS; replicaNum++) {
        percentComplete += percentCompleteEachReplica;
        calcCFFs(replicaNum);
        printf("\033[ASet %d: %.2lf%% complete\n", desiredSet, percentComplete);
    }

    calcMeanAndStdDev();
}
