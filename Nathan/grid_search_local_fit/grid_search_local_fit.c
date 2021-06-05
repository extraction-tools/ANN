#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <float.h>

#include "TVA1_UU.h"

#define NUM_INDEXES 45
#define NUM_REPLICAS 100

typedef struct {
    double ReH, ReE, ReHtilde, error;
} CFFGuess;

static double k, QQ, x, t,
              F[NUM_INDEXES], errF[NUM_INDEXES],
              F1, F2,
              dvcs,
              ReH_guess, ReE_guess, ReHtilde_guess,
              ReH_mean, ReE_mean, ReHtilde_mean,
              ReH_stddev, ReE_stddev, ReHtilde_stddev;

static int desiredSet, phi[NUM_INDEXES];

static CFFGuess cffGuesses[NUM_REPLICAS];

// Read in the data
// Returns 0 if success, 1 if error
bool readInData(char *filename);

// Box-Muller Transform --> Turn a uniform distribution into a Gaussian distribution
double boxMuller(void);

// Function that calculates the mean and standard deviation of the CFFs over all the replicas
void calcMeanAndStdDev(void);

// Calculate the mean percent error in F for a specific replica and the given CFFs
double calcFError(double replicas[], double ReH, double ReE, double ReHtilde);

// Estimate the correct CFF values for a specific replica
void calcCFFs(int replicaNum);

// Fits CFFs to all of the replicas
void localFit(void);



int main(int argc, char *argv[]) {
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

    localFit();

/*
    printf("ReH_guess: %lf, ReH_mean: %lf, ReH_stddev: %lf\n", ReH_guess, ReH_mean, ReH_stddev);
    printf("ReE_guess: %lf, ReE_mean: %lf, ReE_stddev: %lf\n", ReE_guess, ReE_mean, ReE_stddev);
    printf("ReHtilde_guess: %lf, ReHtilde_mean: %lf, ReHtilde_stddev: %lf\n\n", ReHtilde_guess, ReHtilde_mean, ReHtilde_stddev);
*/

    FILE *f = fopen("grid_search_local_fit_output.csv", "r");

    if (f == NULL) {
        f = fopen("grid_search_local_fit_output.csv", "w");
        fprintf(f, "Set,ReH_mean,ReH_stddev,ReE_mean,ReE_stddev,ReHtilde_mean,ReHtilde_stddev\n");
    } else {
        fclose(f);
        f = fopen("grid_search_local_fit_output.csv", "a");
    }

    fprintf(f, "%d,%lf,%lf,%lf,%lf,%lf,%lf\n", desiredSet, ReH_mean, ReH_stddev, ReE_mean, ReE_stddev, ReHtilde_mean, ReHtilde_stddev);

    fclose(f);

    return 0;
}



// Read in the data
// returns 0 if success, 1 if error
bool readInData(char *filename) {
    FILE *f = fopen(filename, "r");
    char buff[1024] = {0};
    bool previouslyEncounteredSet = 0;

    if (f == NULL) {
        printf("Error: could not open file %s.\n", filename);
        return 1;
    }

    fgets(buff, 1024, f);

    while (fgets(buff, 1024, f)) {
        int set, index;

        sscanf(buff, "%d,%*s\n", &set);

        if (set == desiredSet) {
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
double boxMuller(void) {
    return sqrt(-2 * log(rand() / ((double) RAND_MAX))) * sin(2 * M_PI * (rand() / ((double) RAND_MAX)));
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
double calcFError(double replicas[], double ReH, double ReE, double ReHtilde) {
    double sum_of_percent_errors = 0.0;

    for (int index = 0; index < NUM_INDEXES; index++) {
        double F_predicted = TVA1_UU_GetBHUU(phi[index], F1, F2) +
                             TVA1_UU_GetIUU(phi[index], F1, F2, ReH, ReE, ReHtilde) +
                             dvcs;

        double F_actual = F[index] + (replicas[index] * errF[index]);


        double percent_error = (F_actual - F_predicted) / F_actual;

        // Only care about the magnitude of the error, not the sign
        if (percent_error < 0) percent_error *= -1;

        sum_of_percent_errors += percent_error;
    }

    return sum_of_percent_errors / NUM_INDEXES;
}



// Estimate the correct CFF values for a specific replica
void calcCFFs(int replicaNum) {
    double ReHguess, ReEguess, ReHtildeguess;

    double bestError = DBL_MAX;

    // replica: the number of standard deviations that the values of F will be off by
    double replicas[NUM_INDEXES];

    if (replicaNum == -1) {
        for (int i = 0; i < NUM_INDEXES; i++) replicas[i] = 0.0;
    } else {
        for (int i = 0; i < NUM_INDEXES; i++) replicas[i] = boxMuller();
    }

    // num: number of points tested on either side of the guess
    // Value is arbitrary for correctness but greatly impacts the speed of the program
    const int num = 10;

    // totalDist: the total distance being tested on either side of the guess
    // Value is arbitrary for correctness but greatly impacts the speed of the program
    const int totalDist = (replicaNum == -1) ? 100 : 10;

    if (replicaNum == -1) {
        ReHguess = 0.0;
        ReEguess = 0.0;
        ReHtildeguess = 0.0;
    } else {
        ReHguess = ReH_guess;
        ReEguess = ReE_guess;
        ReHtildeguess = ReHtilde_guess;
    }

    // dist: distance between each point being tested
    for (double dist = (double) totalDist / (double) num; dist >= 0.0001; dist /= num) {
        double maxChange = dist * num, minChange = -1 * dist * num;
        double bestReHguess = 0, bestReEguess = 0, bestReHtildeguess = 0;
        double prevError = DBL_MAX;

        bestError = DBL_MAX;

        while ((fabs(bestError - prevError) > 0.000001) || (DBL_MAX - bestError < 0.000001)) {
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
    TVA1_UU_SetKinematics(QQ, x, t, k);

    //printf("Starting set %d...\n", desiredSet);

    calcCFFs(-1);

    //printf("\033[ASet %d: 0.00%% complete\n", desiredSet);

    for (int replicaNum = 0; replicaNum < NUM_REPLICAS; replicaNum++) {
        calcCFFs(replicaNum);
        //printf("\033[ASet %d: %.2lf%% complete\n", desiredSet, (100. * (double) (replicaNum + 1) / (double) NUM_REPLICAS));
    }

    calcMeanAndStdDev();
}
