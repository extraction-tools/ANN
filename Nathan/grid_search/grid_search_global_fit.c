#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <float.h>

#define ReH 0
#define ReE 1
#define ReHtilde 2

typedef struct {
    double k, QQ, x, t, mean[3], error[3];
    uint16_t setNumber;
} SetData;

static SetData* sets[342] = {0};

static uint16_t numSets = 0;

static double coefficients[3][9];


// Read in the data
// Returns 0 if success, 1 if error
static bool readInData(char * restrict filename);

static double calcCFFError(uint8_t cff, double test_coefficients[9], uint16_t setToChange, double amountToChangeBy);

void calcCoefficients(uint8_t cff);

// Fits CFFs to all of the replicas
static void globalFit(void);

static void removeBadData(void);

static void sortData(uint8_t cff);



int main(int argc, char **argv) {
    srand((unsigned int) time(0));

    if (argc == 2) {
        if(readInData(argv[1])) {
            printf("Read failed.\n");
            return 1;
        }
    } else {
        printf("Please specify a data file (.csv) only.\n");
        return 1;
    }

    removeBadData();

    globalFit();

    for (uint16_t i = 0; i < numSets; ++i) free(sets[i]);

    return 0;
}



// Read in the data
// returns 0 if success, 1 if error
static bool readInData(char * restrict filename) {
    // Format of the file:
    // set#,k,QQ,x,t,ReH_mean,ReH_error,ReE_mean,ReE_error,ReHtilde_mean,ReHtilde_error

    FILE *f = fopen(filename, "r");
    char buff[128] = {0};

    if (f == NULL) {
        printf("Error: could not open file %s.\n", filename);
        return 1;
    }

    fgets(buff, 128, f);

    for (int set = 0; fgets(buff, 128, f); set++) {
        sets[set] = malloc(sizeof(SetData));
        ++numSets;

        sscanf(buff, "%hu,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
               &sets[set]->setNumber,
               &sets[set]->k, &sets[set]->QQ, &sets[set]->x, &sets[set]->t,
               &sets[set]->mean[ReH], &sets[set]->error[ReH],
               &sets[set]->mean[ReE], &sets[set]->error[ReE],
               &sets[set]->mean[ReHtilde], &sets[set]->error[ReHtilde]);
    }

    fclose(f);

    return 0;
}


static double calcCFFError(uint8_t cff, double test_coefficients[9], uint16_t setToChange, double amountToChangeBy) {
    double sum_of_squared_percent_errors = 0.0;

    for (uint16_t i = 0; i < numSets; i++) {
        double estimate =
            (test_coefficients[0] * sets[i]->k                ) +
            (test_coefficients[1] * sets[i]->k  * sets[i]->k  ) +
            (test_coefficients[2] * sets[i]->QQ               ) +
            (test_coefficients[3] * sets[i]->QQ * sets[i]->QQ ) +
            (test_coefficients[4] * sets[i]->x                ) +
            (test_coefficients[5] * sets[i]->x  * sets[i]->x  ) +
            (test_coefficients[6] * sets[i]->t                ) +
            (test_coefficients[7] * sets[i]->t  * sets[i]->t  ) +
            test_coefficients[8];

        double actual = sets[i]->mean[cff];

        if (i == setToChange) actual += amountToChangeBy * sets[i]->error[cff];

        //printf("actual=%lf\testimate=%lf\n", actual, estimate);

        // Take the absolute value because we only care about the magnitude of the error, not the sign
        double percent_error = (actual == 0.0) ? 0.0 : (actual - estimate) / actual;

        sum_of_squared_percent_errors += percent_error * percent_error;
    }

    return sqrt(sum_of_squared_percent_errors / numSets);
}



// Estimate the correct coefficient values for a specific replica
void calcCoefficients(uint8_t cff) {
    // num: number of points tested on either side of the guess
    const uint8_t num = 10;

    // totalDist: the total distance being tested on either side of the guess
    const double totalDist = 10.;

    const double amountsToChangeBy[11] = {0.0, -0.05, 0.05, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.5, 0.5};

    for (uint16_t setToChange = 0; setToChange < numSets; setToChange++) {
        double bestAmountToChangeBy = 0.0;
        uint8_t bestAmountToChangeByCount = 0;
        double bestAmountToChangeBysError = DBL_MAX;
        double amountToChangeBysCoefficients[21][9];
        for (uint8_t i = 0; i < 21; i++) {
            for (uint8_t j = 0; j < 9; j++) {
                amountToChangeBysCoefficients[i][j] = coefficients[cff][j];
            }
        }

        for (uint8_t amountToChangeByCount = 0; amountToChangeByCount < 11; amountToChangeByCount++) {
            double bestErrorForAmountToChangeBy = DBL_MAX;

            // dist: distance between each point being tested
            for (double dist = (double) totalDist / num; dist >= 0.01; dist /= num) {
                printf("dist=%lf\n", dist);
                double maxChange = dist * num;
                double prevErrorForAmountToChangeBy = DBL_MAX;

                bestErrorForAmountToChangeBy = DBL_MAX;

                while ((prevErrorForAmountToChangeBy - bestErrorForAmountToChangeBy > 0.001) || (DBL_MAX - bestErrorForAmountToChangeBy < 0.000001)) {
                    prevErrorForAmountToChangeBy = bestErrorForAmountToChangeBy;

                    for (double r = dist; r <= maxChange; r += dist) {
                        for (uint32_t i = 0; i < 3355443.2; i++) {
                            double phis[8];
                            double new_coefficients[9], coefficient_diffs[9];

                            for (uint8_t j = 0; j < 8; j++) {
                                phis[j] = M_PI * (rand() / (double) RAND_MAX);
                            }
                            phis[7] *= 2.;

                            coefficient_diffs[0] = r * cos(phis[0]);
                            coefficient_diffs[1] = r * sin(phis[0]) * cos(phis[1]);
                            coefficient_diffs[2] = r * sin(phis[0]) * sin(phis[1]) * cos(phis[2]);
                            coefficient_diffs[3] = r * sin(phis[0]) * sin(phis[1]) * sin(phis[2]) * cos(phis[3]);
                            coefficient_diffs[4] = r * sin(phis[0]) * sin(phis[1]) * sin(phis[2]) * sin(phis[3]) * cos(phis[4]);
                            coefficient_diffs[5] = r * sin(phis[0]) * sin(phis[1]) * sin(phis[2]) * sin(phis[3]) * sin(phis[4]) * cos(phis[5]);
                            coefficient_diffs[6] = r * sin(phis[0]) * sin(phis[1]) * sin(phis[2]) * sin(phis[3]) * sin(phis[4]) * sin(phis[5]) * cos(phis[6]);
                            coefficient_diffs[7] = r * sin(phis[0]) * sin(phis[1]) * sin(phis[2]) * sin(phis[3]) * sin(phis[4]) * sin(phis[5]) * sin(phis[6]) * cos(phis[7]);
                            coefficient_diffs[8] = r * sin(phis[0]) * sin(phis[1]) * sin(phis[2]) * sin(phis[3]) * sin(phis[4]) * sin(phis[5]) * sin(phis[6]) * sin(phis[7]);

                            //printf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", coefficient_diffs[0],coefficient_diffs[1],coefficient_diffs[2],coefficient_diffs[3],coefficient_diffs[4],coefficient_diffs[5],coefficient_diffs[6],coefficient_diffs[7], coefficient_diffs[8]);

                            for (uint8_t j = 0; j < 9; j++) {
                                new_coefficients[j] = coefficient_diffs[j] + amountToChangeBysCoefficients[amountToChangeByCount][j];
                            }

                            double error = calcCFFError(cff, new_coefficients, setToChange, amountsToChangeBy[amountToChangeByCount]);
                            //printf("error=%lf\n", error);

                            if (error < bestErrorForAmountToChangeBy) {
                                for (uint8_t j = 0; j < 9; j++) {
                                    amountToChangeBysCoefficients[amountToChangeByCount][j] = new_coefficients[j];
                                }
                                bestErrorForAmountToChangeBy = error;
                            }
                        }
                    }

                    printf("bestErrorForAmountToChangeBy=%.5lf\n", bestErrorForAmountToChangeBy);
                }
            }

            if (bestErrorForAmountToChangeBy < bestAmountToChangeBysError - 0.005) {
                bestAmountToChangeBy = amountsToChangeBy[amountToChangeByCount];
                bestAmountToChangeBysError = bestErrorForAmountToChangeBy;
                bestAmountToChangeByCount = amountToChangeByCount;
            }

            printf("setToChange=%d\tamountToChangeBy=%lf\tbestError=%lf\n", setToChange, amountsToChangeBy[amountToChangeByCount], bestErrorForAmountToChangeBy);
        }

        printf("\nSet #%d:\n", sets[setToChange]->setNumber);
        printf("\tBest amount to change by = %lf\n", bestAmountToChangeBy);
        printf("\tMinimum error = %lf\n", bestAmountToChangeBysError);
        printf("\tOld mean value = %lf\n", sets[setToChange]->mean[cff]);
        sets[setToChange]->mean[cff] += bestAmountToChangeBy * sets[setToChange]->error[cff];
        printf("\tNew mean value = %lf\n\n", sets[setToChange]->mean[cff]);

        for (uint8_t i = 0; i < 9; i++) {
            coefficients[cff][i] = amountToChangeBysCoefficients[bestAmountToChangeByCount][i];
        }

        FILE *f = fopen("grid_search_global_fit_output.csv", "a+");
        fprintf(f, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", sets[setToChange]->setNumber, sets[setToChange]->k, sets[setToChange]->QQ, sets[setToChange]->x, sets[setToChange]->t, sets[setToChange]->mean[0], sets[setToChange]->mean[1], sets[setToChange]->mean[2]);
        fclose(f);
    }
}



static void removeBadData(void) {
    double stddev_mean = 0.0, stddev_stddev = 0.0;

    for (uint16_t i = 0; i < numSets; i++) {
        stddev_mean += sets[i]->error[ReE];
    }

    stddev_mean /= numSets;

    for (uint16_t i = 0; i < numSets; i++) {
        stddev_stddev += (sets[i]->error[ReE] - stddev_mean) * (sets[i]->error[ReE] - stddev_mean);
    }

    stddev_stddev = sqrt(stddev_stddev / (numSets - 1));

    for (uint16_t i = 0; i < numSets; i++) {
        if (sets[i]->error[ReE] < stddev_mean - stddev_stddev) {
            printf("Removed set %d\n", i+1);

            sets[i]->k = 0.0;
            sets[i]->QQ = 0.0;
            sets[i]->x = 0.0;
            sets[i]->t = 0.0;
            sets[i]->mean[ReH] = 0.0;
            sets[i]->mean[ReE] = 0.0;
            sets[i]->mean[ReHtilde] = 0.0;
            sets[i]->error[ReH] = 0.0;
            sets[i]->error[ReE] = 0.0;
            sets[i]->error[ReHtilde] = 0.0;
        }
    }
}

static void sortData(uint8_t cff) {
    for (uint16_t i = 1; i < numSets; i++) { // go through each value from left to right (start on the second one so it can be compared to the first)
        for (uint16_t j = i; (j > 0) && (sets[j]->error[cff] > sets[j-1]->error[cff]); j--) {
            // go through all values BEFORE i (unlike bubble sort)
            // break the loop when this value is in its place (when a[j] >= a[j-1])
            // moves the value as far left as it can --> if already sorted, it breaks the loop immediately (unlike with bubble sort)
            void *temp = sets[j];
            sets[j] = sets[j-1];
            sets[j-1] = temp;
        }
    }

    for (; sets[numSets - 1]->error[ReH] < 0.0000001; --numSets) {
        free(sets[numSets - 1]);
        sets[numSets - 1] = NULL;
    }
}


// Fits CFFs to all of the replicas
static void globalFit(void) {
    for (uint8_t cff = ReH; cff <= ReH; ++cff) {

        sortData(cff);

        double cff_mean = 0.0;

        for (uint8_t j = 0; j < 9; j++) {
            coefficients[cff][j] = 0.0;
        }

        for (uint16_t j = 0; j < numSets; j++) {
            cff_mean += sets[j]->mean[cff];
        }

        cff_mean /= numSets;
        coefficients[cff][8] = cff_mean;

        //printf("Starting %s...\n", cffNames[i]);

        calcCoefficients(cff);

        printf("k:\t%lf\n",     coefficients[cff][0]);
        printf("k2:\t%lf\n",    coefficients[cff][1]);
        printf("QQ:\t%lf\n",    coefficients[cff][2]);
        printf("QQ2:\t%lf\n",   coefficients[cff][3]);
        printf("x:\t%lf\n",     coefficients[cff][4]);
        printf("x2:\t%lf\n",    coefficients[cff][5]);
        printf("t:\t%lf\n",     coefficients[cff][6]);
        printf("t2:\t%lf\n",    coefficients[cff][7]);
        printf("off:\t%lf\n",   coefficients[cff][8]);

        printf("\nerror=%lf\n\n", calcCFFError(cff, coefficients[cff], 0, 0.0));
    }
}
