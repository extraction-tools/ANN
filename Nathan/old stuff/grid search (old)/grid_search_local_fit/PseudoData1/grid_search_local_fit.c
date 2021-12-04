#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <float.h>

#include "BHDVCS.h"

#define NUM_INDEXES 36
#define NUM_REPLICAS 100

typedef struct {
    double ReH, ReE, ReHtilde, error;
} CFFGuess;

double k, QQ, x_b, t,
       F[NUM_INDEXES], errF[NUM_INDEXES],
       F1, F2,
       dvcs,
       ReH_actual, ReE_actual, ReHtilde_actual,
       ReH_guess, ReE_guess, ReHtilde_guess,
       ReH_mean, ReE_mean, ReHtilde_mean,
       ReH_stddev, ReE_stddev, ReHtilde_stddev;

unsigned int phi[NUM_INDEXES];

BHDVCS bhdvcs;

CFFGuess cffGuesses[NUM_REPLICAS];

unsigned int desiredSet;

// Read in the data
// Returns 0 if success, 1 if error
bool readInData(char *filename);

// Box-Muller Transform --> Turn a uniform distribution into a Gaussian distribution
double boxMuller();

// Function that calculates the mean and standard deviation of the CFFs over all the replicas
void calcMeanAndStdDev(void);

// Calculate the mean percent error in F for a specific replica and the given CFFs
double calcFError(const double * restrict replicas, double ReH, double ReE, double ReHtilde);

// Estimate the correct CFF values for a specific replica
void calcCFFs(int replicaNum);

// Fits CFFs to all of the replicas
void localFit(void);



int main(int argc, char *argv[]) {
    srand(time(0));

    if (argc == 3) {
        desiredSet = atoi(argv[2]);
        if(readInData(argv[1]) == 1) return 1;
    } else {
        printf("Please specify a data file (.csv) and a set number only\n");
        return 1;
    }
/*
    for (int i = 0; i < NUM_REPLICAS; i += 2) {
        boxMuller(&cffGuesses[i].replica, &cffGuesses[i+1].replica);
    }
*/
    localFit();

    FILE *f = fopen("grid_search_local_fit_output.csv", "r");

    if (f == NULL) {
        f = fopen("grid_search_local_fit_output.csv", "w");
        fprintf(f, "set,ReH_actual,ReH_mean,ReH_stddev,ReE_actual,ReE_mean,ReE_stddev,ReHtilde_actual,ReHtilde_mean,ReHtilde_stddev\n");
    } else {
        fclose(f);
        f = fopen("grid_search_local_fit_output.csv", "a");
    }

    calcMeanAndStdDev();

/*
    printf("\nReH_mean: %lf, ReH_stddev: %lf\n", ReH_mean, ReH_stddev);
    printf("ReE_mean: %lf, ReE_stddev: %lf\n", ReE_mean, ReE_stddev);
    printf("ReHtilde_mean: %lf, ReHtilde_stddev: %lf\n\n\n", ReHtilde_mean, ReHtilde_stddev);
*/

    fprintf(f, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", desiredSet, ReH_actual, ReH_mean, ReH_stddev, ReE_actual, ReE_mean, ReE_stddev, ReHtilde_actual, ReHtilde_mean, ReHtilde_stddev);
    fclose(f);

    return 0;
}



// Read in the data
// returns 0 if success, 1 if error
bool readInData(char *filename) {
    FILE *f = fopen(filename, "r");
    char buff[1024];
    bool previouslyEncounteredSet = 0;

    if (f == NULL) return 1;

    fgets(buff, 1024, f);

    while (fgets(buff, 1024, f)) {
        unsigned int set, index;

        sscanf(buff, "%u,%*s\n", &set);

        if (set == desiredSet) {
            sscanf(buff, "%*u,%u,%*s\n", &index);

            if (!previouslyEncounteredSet) {
                sscanf(buff, "%*u,%*u,%lf,%lf,%lf,%lf,%u,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
                       &k, &QQ, &x_b, &t,
                       &phi[index], &F[index], &errF[index],
                       &F1, &F2,
                       &dvcs,
                       &ReH_actual, &ReE_actual, &ReHtilde_actual);

                previouslyEncounteredSet = 1;
            } else {
                sscanf(buff, "%*u,%*u,%*lf,%*lf,%*lf,%*lf,%u,%lf,%lf,%*lf,%*lf,%*lf,%*lf,%*lf,%*lf\n",
                       &phi[index], &F[index], &errF[index]);
            }
        }
    }

    fclose(f);

    return !previouslyEncounteredSet;
}



// Box-Muller Transform --> Turn a uniform distribution into a Gaussian distribution
// Returns a random number following a Gaussian distribution wheree mean=0 and stddev=1
double boxMuller() {
    return sqrt(-2. * log((double) rand() / RAND_MAX)) * sin(2. * M_PI * ((double) rand() / RAND_MAX));
}



// Function that calculates the mean and standard deviation of the CFFs over all the replicas
void calcMeanAndStdDev(void) {
    ReH_mean = ReE_mean = ReHtilde_mean = 0.0;

    double ReH_sum_of_differences = 0.0, ReE_sum_of_differences = 0.0, ReHtilde_sum_of_differences = 0.0;

    for (int replicaNum = 0; replicaNum < NUM_REPLICAS; replicaNum++) {
        ReH_mean += cffGuesses[replicaNum].ReH;
        ReE_mean += cffGuesses[replicaNum].ReE;
        ReHtilde_mean += cffGuesses[replicaNum].ReHtilde;
    }

    ReH_mean /= NUM_REPLICAS;
    ReE_mean /= NUM_REPLICAS;
    ReHtilde_mean /= NUM_REPLICAS;

    for (int replicaNum = 0; replicaNum < NUM_REPLICAS; replicaNum++) {
        ReH_sum_of_differences += (cffGuesses[replicaNum].ReH - ReH_mean) * (cffGuesses[replicaNum].ReH - ReH_mean);
        ReE_sum_of_differences += (cffGuesses[replicaNum].ReE - ReE_mean) * (cffGuesses[replicaNum].ReE - ReE_mean);
        ReHtilde_sum_of_differences += (cffGuesses[replicaNum].ReHtilde - ReHtilde_mean) * (cffGuesses[replicaNum].ReHtilde - ReHtilde_mean);
    }

    ReH_stddev = sqrt(ReH_sum_of_differences / (NUM_REPLICAS - 1));
    ReE_stddev = sqrt(ReE_sum_of_differences / (NUM_REPLICAS - 1));
    ReHtilde_stddev = sqrt(ReHtilde_sum_of_differences / (NUM_REPLICAS - 1));
}



// Calculate the mean percent error in F for a specific replica and the given CFFs
double calcFError(const double * restrict replicas, double ReH, double ReE, double ReHtilde) {
    double mean_percent_error = 0.0;

    for (int index = 0; index < NUM_INDEXES; index++) {
        double F_predicted = BHDVCS_TotalUUXS(&bhdvcs, phi[index], k, QQ, x_b, t,
                                              F1, F2, dvcs, ReH, ReE, ReHtilde);

        double F_actual = F[index] + (replicas[index] * errF[index]);

        double error = (F_actual - F_predicted) / F_actual;

        // Only care about the magnitude of the error, not the sign
        if (error < 0) error = -1 * error;

        mean_percent_error += error;
    }

    mean_percent_error /= NUM_INDEXES;

    // Return the mean percent error
    return mean_percent_error;
}



// Estimate the correct CFF values for a specific replica
void calcCFFs(int replicaNum) {
    double ReHguess, ReEguess, ReHtildeguess, bestError;

    const double replicas[] = {(replicaNum == -1) ? 0.0 : boxMuller()};

    // num: number of points tested on either side of the guess
    const unsigned int num = 10;

    // totalDist: the total distance being tested on either side of the guess
    // Value is arbitrary for correctness
    // 100:10 makes the algorithm faster for the pseudodata
    const unsigned int totalDist = (replicaNum == -1) ? 100 : 10;

    if (replicaNum == -1) {
        ReHguess = ReEguess = ReHtildeguess = 0;
    } else {
        ReHguess = ReH_guess;
        ReEguess = ReE_guess;
        ReHtildeguess = ReHtilde_guess;
    }

    // dist: distance between each point being tested
    for (double dist = (double) totalDist / (double) num; dist >= 0.001; dist /= num) {
        double maxChange = dist * num, minChange = -1 * dist * num;
        double bestReHguess = 0, bestReEguess = 0, bestReHtildeguess = 0;
        double prevError = DBL_MAX;

        bestError = DBL_MAX;

        while ((prevError != bestError) || 
               (prevError - bestError >= 0.000001 || prevError - bestError <= -0.000001) || 
               (bestError == DBL_MAX)) {

            prevError = bestError;

            for (double ReHchange = minChange; ReHchange <= maxChange; ReHchange += dist) {
                for (double ReEchange = minChange; ReEchange <= maxChange; ReEchange += dist) {
                    for (double ReHtildechange = minChange; ReHtildechange <= maxChange; ReHtildechange += dist) {
                        double error = calcFError(&replicas, ReHguess + ReHchange, ReEguess + ReEchange, ReHtildeguess + ReHtildechange);

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

    if (replicaNum == -1) {
        ReH_guess = ReHguess;
        ReE_guess = ReEguess;
        ReHtilde_guess = ReHtildeguess;
    } else {
        cffGuesses[replicaNum].ReH = ReHguess;
        cffGuesses[replicaNum].ReE = ReEguess;
        cffGuesses[replicaNum].ReHtilde = ReHtildeguess;
        cffGuesses[replicaNum].error = bestError;
    }
}



// Fits CFFs to all of the replicas
void localFit(void) {
    //double averageMPE_F = 0.0, averageMPE_F_actual = 0.0;

    BHDVCS_Init(&bhdvcs);
    BHDVCS_SetKinematics(&bhdvcs, QQ, x_b, t, k);

    printf("Starting set %d...\n", desiredSet);

    calcCFFs(-1);

    printf("\033[ASet %d: 0.00%% complete\n", desiredSet);

    for (int replicaNum = 0; replicaNum < NUM_REPLICAS; replicaNum++) {
        calcCFFs(replicaNum);

/*
        double F_actual_error = calcFError(replicaNum, ReH_actual, ReE_actual, ReHtilde_actual);

        averageMPE_F += cffGuesses[replicaNum].error;
        averageMPE_F_actual += F_actual_error;

        double ReH_guess = cffGuesses[replicaNum].ReH;
        double ReE_guess = cffGuesses[replicaNum].ReE;
        double ReHtilde_guess = cffGuesses[replicaNum].ReHtilde;

        printf("Set %d, ReplicaNum %d (value of %lf):\n", desiredSet, replicaNum, cffGuesses[replicaNum].replica);
        printf("\tReH:\n\t\tActual = %.5lf\n\t\tEstimate = %.3lf\n", ReH_actual, ReH_guess);
        printf("\tReE:\n\t\tActual = %.5lf\n\t\tEstimate = %.3lf\n", ReE_actual, ReE_guess);
        printf("\tReHtilde:\n\t\tActual = %.5lf\n\t\tEstimate = %.3lf\n", ReHtilde_actual, ReHtilde_guess);
        printf("\tEstimates' Mean Percent Error in F = %lf%%\n", cffGuesses[replicaNum].error * 100);
        printf("\tActual Values' Mean Percent Error in F = %lf%%\n\n", F_actual_error * 100);
*/

        printf("\033[ASet %d: %.2lf%% complete\n", desiredSet, (100. * (double) (replicaNum + 1) / (double) NUM_REPLICAS));
    }
}
