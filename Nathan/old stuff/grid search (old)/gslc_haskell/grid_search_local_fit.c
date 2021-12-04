#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <HsFFI.h>

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

// Read in the data
// Returns 0 if success, 1 if error
bool readInData(char *filename);

// Fits CFFs to all of the replicas
extern void localFit(void);

int getNumIndexes(void);
int getNumReplicas(void);
double getDVCS(void);
double getF(int index);
double getErrF(int index);
int getPhi(int index);
double getBHUU(int phi_val);
double getIUU(int phi_val, double ReH, double ReE, double ReHtilde);
void writeMeanAndStddev(double reH_mean, double reH_stddev, double reE_mean, double reE_stddev, double reHtilde_mean, double reHtilde_stddev);
double getRand(unsigned int a);

// Helper functions that lets the Haskell algorithm read from C variables
int getNumIndexes(void) { return NUM_INDEXES; }
int getNumReplicas(void) { return NUM_REPLICAS; }
double getDVCS(void) { return dvcs; }

double getF(int index) { return F[index]; }
double getErrF(int index) { return errF[index]; }
int getPhi(int index) { return phi[index]; }

// Helper functions that lets the Haskell algorithm call TVA1_UU functions
double getBHUU(int phi_val) { return TVA1_UU_GetBHUU(phi_val, F1, F2); }
double getIUU(int phi_val, double ReH, double ReE, double ReHtilde) { return TVA1_UU_GetIUU(phi_val, F1, F2, ReH, ReE, ReHtilde); }

// Helper function that lets the Haskell algorithm write to C variables
void writeMeanAndStddev(double reH_mean, double reH_stddev, double reE_mean, double reE_stddev, double reHtilde_mean, double reHtilde_stddev) {
    printf("called writeMeanAndStddev with reH_mean = %lf\n", reH_mean);
    ReH_mean = reH_mean;
    ReH_stddev = reH_stddev;
    ReE_mean = reE_mean;
    ReE_stddev = reE_stddev;
    ReHtilde_mean = reHtilde_mean;
    ReHtilde_stddev = reHtilde_stddev;
}

// Produces a pseudorandom number uniformly distributed in the range [0, 1]
// The input `a` is here for other reasons (Haskell being shitty) but it does help with randomness
double getRand(unsigned int a) {
    uint32_t tsc = 0;

    // Read timestamp counter into EDX:EAX
    // Set `tsc` to only the value in EAX, since the value in EDX is practically constant
    __asm__ volatile ("rdtsc" : "=a" (tsc));

    // Fit `a` into the range [1, 20]
    a = (a % 19) + 1;

    // XOR `tsc` with itself shifted by `a` bits
    tsc ^= (tsc << a);

    // Take the last 20 bits (~6 digits) of `tsc` and fit it into the range [0, 1]
    return (double) (tsc & 0xFFFFF) / 0xFFFFF;
}



int main(int argc, char *argv[]) {
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

    hs_init(&argc, &argv);
    localFit();
    hs_exit();

    printf("finished local fit\n");

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
