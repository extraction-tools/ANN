#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "vars.h"
#include "float.h"


struct FitParametersGuess {
    double p[9], error;
};

struct FitParametersGuess fitParametersGuesses[3];

// F-values
double replicas[NUM_SETS][NUM_INDEXES][10];

// Functions for the global fit
double calcCFFError(int cff, double p[]);
void calcFitParameters(int cff);
void globalFit(void);

int main() {
    srand(time(0));
    readInData("dvcs_xs_newsets_genCFFs.csv");

    for (int set = 0; set < NUM_SETS; set++) {
        for (int index = 0; index < NUM_INDEXES; index++) {
            for (int i = 0; i < 10; i += 2) {
                box_muller(F[set][index], errF[set][index], &replicas[set][index][i], &replicas[set][index][i+1]);
            }
        }
    }

    globalFit();
    return 0;
}

// param cff: 0 - ReH, 1 - ReE, 2 - ReHtilde
double calcCFFError(int cff, double *p) {
    double totalPercentError = 0.0;

    for (int set = 0; set < NUM_SETS; set++) {
        double CFF_actual =
            (cff == 0) ? cffGuesses[set].ReH :
            (cff == 1) ? cffGuesses[set].ReE :
            cffGuesses[set].ReHtilde;

        double CFF_predicted =
            p[0] +
            (p[1] * k[set]) +
            (p[2] * k[set] * k[set]) +
            (p[3] * QQ[set]) +
            (p[4] * QQ[set] * QQ[set]) +
            (p[5] * x_b[set]) +
            (p[6] * x_b[set] * x_b[set]) +
            (p[7] * t[set]) +
            (p[8] * t[set] * t[set]);

        double error = (CFF_actual - CFF_predicted) / CFF_actual;

        // Only care about the magnitude of the error, not the sign
        if (error < 0) error = -1 * error;

        totalPercentError += error;
    }

    // Return the Mean Percent error (MPE)
    return totalPercentError / NUM_SETS;
}



// param cff: 0 - ReH, 1 - ReE, 2 - ReHtilde
void calcFitParameters(int cff) {
    // CFF = [0] + [1]k + [2]k^2 + [3]QQ + [4]QQ^2 + [5]x_b + [6]x_b^2 + [7]t + [8]t^2

    double guesses[9], bestGuesses[9], bestError;

    for (int i = 0; i < 9; i++) {
        guesses[i] = rand() % 50;
    }

    // 2 and 3 work. 4 is unreasonable
    const int num = 10;

    bestError = DBL_MAX;

    // dist: distance between each point being tested
    // Range for all parameters is [-200, 200] (when num=2). tighter range when num=3
    for (double dist = 100 / num; dist >= 0.000001; dist /= num) {
        //bestError = DBL_MAX;

        double pchange[4];
        double maxChange = dist * num;
        double minChange = -1 * maxChange;

        int numSameBestError = 0;

        while (numSameBestError < 50) {
            int param1 = rand() % 9, param2 = rand() % 9, param3 = rand() % 9;
            while (param2 == param1) param2 = rand() % 9;
            while (param3 == param2 || param3 == param1) param3 = rand() % 9;

            double prevError = DBL_MAX;

            while (prevError != bestError || bestError == DBL_MAX) {
                prevError = bestError;

                for (pchange[0] = minChange; pchange[0] <= maxChange; pchange[0] += dist) {
                    for (pchange[1] = minChange; pchange[1] <= maxChange; pchange[1] += dist) {
                        for (pchange[2] = minChange; pchange[2] <= maxChange; pchange[2] += dist) {
                            double p[9];

                            for (int i = 0; i < 9; i++) p[i] = guesses[i];

                            p[param1] += pchange[param1];
                            p[param2] += pchange[param2];
                            p[param3] += pchange[param3];

                            double error = calcCFFError(cff, p);

                            if (error < bestError) {
                                for (int i = 0; i < 9; i++) bestGuesses[i] = guesses[i];

                                bestGuesses[param1] += pchange[param1];
                                bestGuesses[param2] += pchange[param2];
                                bestGuesses[param3] += pchange[param3];

                                bestError = error;

                                numSameBestError = 0;
                            }
                        }
                    }
                }

                for (int i = 0; i < 9; i++) guesses[i] = bestGuesses[i];

                printf("params: %d %d %d; dist = %lf\n", param1, param2, param3, dist);
                printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                    guesses[0], guesses[1], guesses[2], guesses[3], guesses[4],
                    guesses[5], guesses[6], guesses[7], guesses[8]);
                printf("bestError: %lf. prevError: %lf, numSameBestError: %d\n\n", bestError, prevError, numSameBestError);


            }

            numSameBestError++;
        }
    }

    for (int i = 0; i < 9; i++) fitParametersGuesses[cff].p[i] = guesses[i];
    fitParametersGuesses[cff].error = bestError;
}



// Calculates the fit parameters for all 3 CFFs
void globalFit() {
    // ReH
    calcFitParameters(0);
    printf("ReH = %lf + %lfk + %lfk^2 + %lfQQ + %lfQQ^2 + %lfx_b + %lfx_b^2 + %lft + %lft^2\n",
           fitParametersGuesses[0].p[0], fitParametersGuesses[0].p[1], fitParametersGuesses[0].p[2], fitParametersGuesses[0].p[3],
           fitParametersGuesses[0].p[4], fitParametersGuesses[0].p[5], fitParametersGuesses[0].p[6], fitParametersGuesses[0].p[7],
           fitParametersGuesses[0].p[8]);
    printf("Mean error over all sets: %lf%%\n\n", fitParametersGuesses[0].error * 100);

    // ReE
    calcFitParameters(1);
    printf("ReE = %lf + %lfk + %lfk^2 + %lfQQ + %lfQQ^2 + %lfx_b + %lfx_b^2 + %lft + %lft^2\n",
           fitParametersGuesses[1].p[0], fitParametersGuesses[1].p[1], fitParametersGuesses[1].p[2], fitParametersGuesses[1].p[3],
           fitParametersGuesses[1].p[4], fitParametersGuesses[1].p[5], fitParametersGuesses[1].p[6], fitParametersGuesses[1].p[7],
           fitParametersGuesses[1].p[8]);
    printf("Mean error over all sets: %lf%%\n\n", fitParametersGuesses[1].error * 100);

    // ReHtilde
    calcFitParameters(2);
    printf("ReHtilde = %lf + %lfk + %lfk^2 + %lfQQ + %lfQQ^2 + %lfx_b + %lfx_b^2 + %lft + %lft^2\n",
           fitParametersGuesses[2].p[0], fitParametersGuesses[2].p[1], fitParametersGuesses[2].p[2], fitParametersGuesses[2].p[3],
           fitParametersGuesses[2].p[4], fitParametersGuesses[2].p[5], fitParametersGuesses[2].p[6], fitParametersGuesses[2].p[7],
           fitParametersGuesses[2].p[8]);
    printf("Mean error over all sets: %lf%%\n\n", fitParametersGuesses[2].error * 100);
}



void box_muller(double mu, double sigma, double *X, double *Y) {
    // U and V are distributed uniformly in the range [0, 1]
    double U = (double) rand() / RAND_MAX;
    double V = (double) rand() / RAND_MAX;

    // X and Y follow a Gaussian distribution with mean=mu and std.dev=sigma
    *X = (mu * sqrt(-2 * log(U)) * cos(2 * M_PI * V)) + sigma;
    *Y = (mu * sqrt(-2 * log(U)) * sin(2 * M_PI * V)) + sigma;
}
