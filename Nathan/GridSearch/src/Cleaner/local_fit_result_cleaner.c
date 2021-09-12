/*
Grid Search
Local Fit Result Cleaner
Nathan Snyder
*/

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define ReH 0
#define ReE 1
#define ReHtilde 2

#define NUM_SETS 342


// The SetData type stores a set's kinematic variables as well as the results of the local fit for that set
typedef struct {
    double k, QQ, x, t, mean[3], error[3];
    uint16_t setNumber;
} SetData;

// The maximum number of sets you can read is NUM_SETS
static SetData* sets[NUM_SETS] = {0};

// The total number of sets
static uint16_t numSets = 0;


// Read in the data
// Returns 0 if success, 1 if error
static bool readData(char * restrict filename);

// Removes bad data
static void removeBadData(void);

static void sortData(void);

static void writeData(void);



int main(int argc, char **argv) {
    if (argc == 2) {
        if(readData(argv[1])) {
            printf("Read failed.\n");
            return 1;
        }
    } else {
        printf("Please specify a data file (.csv) only.\n");
        return 1;
    }

    removeBadData();

    for (uint16_t i = 0; i < numSets; ++i) free(sets[i]);

    return 0;
}



// Read in the data
// returns 0 if success, 1 if error
static bool readData(char * restrict filename) {
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

static void sortData(void) {
    for (uint16_t i = 1; i < numSets; i++) {
        for (uint16_t j = i; (j > 0) && (sets[j]->setNumber > sets[j-1]->setNumber); j--) {
            void *temp = sets[j];
            sets[j] = sets[j-1];
            sets[j-1] = temp;
        }
    }
}
