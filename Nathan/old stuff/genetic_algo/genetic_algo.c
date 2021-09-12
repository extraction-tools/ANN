#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "TVA1_UU.h"

#define NUM_CANDIDATES 1000

#define NUM_CFFS 3

#define NUM_INDEXES 45

#define NUM_REPLICAS 100

// The algorithm will stop when the fitess score reaches this value
#define FITNESS_SCORE_THRESHOLD 0.99 // Range: [0, 1]

// The algorithm will stop when the number of epochs reaches this value
#define NUMBER_OF_EPOCHS_THRESHOLD 100

#define FRACTION_OF_CANDIDATES_TO_REMOVE 0.7 // Range: [0, 1]

#define MUTATION_PROBABILITY 0.5 // Range: [0, 1]

#define PERCENT_IMMUNE_TO_MUTATION 0.2 // Range: [0, 1]


#define ReH 0
#define ReE 1
#define ReHtilde 2


// Type `Candidate`: represents one candidate/genotype
typedef struct {
    // Genes
    double cffs[NUM_CFFS];

    // Tracking metadata
    double fitness;
} Candidate;

static Candidate *currentCandidates[NUM_CANDIDATES];
static Candidate *bestCandidatesByReplica[NUM_REPLICAS];

static double k, QQ, x, t,
              F[NUM_INDEXES], errF[NUM_INDEXES],
              F1, F2,
              dvcs,
              meanValues[NUM_CFFS], stddevValues[NUM_CFFS],
              ReH_mean, ReE_mean, ReHtilde_mean,
              ReH_stddev, ReE_stddev, ReHtilde_stddev;

static int desiredSet, phi[NUM_INDEXES], numEpochs = 0, currReplica = 0;



static bool readInData(void);

static double boxMuller(void);

static double fitness(const uint16_t candidateIndex, double *replicas);

static bool canStop(void);

// Calculate the fitness of each candidate and then sort the currentCandidates array by fitness in non-increasing order
static void evaluate(double *replicas);

static void reproduction(void);

static void mutation(void);

static void createNewCandidate(const uint16_t candidateIndex);

static void generateInitialPopulation(void);

static void calcMeanAndStdDev(void);



int main(int argc, char** argv) {
    srand((unsigned int) time(0));

    // Initialize the currentCandidates array
    for (uint16_t candidateIndex = 0; candidateIndex < NUM_CANDIDATES; ++candidateIndex) {
        currentCandidates[candidateIndex] = malloc(sizeof(Candidate));
    }

    // Initialize the bestCandidatesByReplica array
    for (uint16_t replicaNumber = 0; replicaNumber < NUM_REPLICAS; ++replicaNumber) {
        bestCandidatesByReplica[replicaNumber] = malloc(sizeof(Candidate));
    }

    // Read in data
    if (argc == 2) {
        desiredSet = (uint16_t) atoi(argv[1]);

        if(readInData()) {
            printf("Read failed.\n");
            goto exit;
        }
    } else {
        printf("Please specify a set number only.\n");
        goto exit;
    }

    TVA1_UU_SetKinematics(QQ, x, t, k);



    double replicas[NUM_INDEXES];
    for (currReplica = 0; currReplica < NUM_REPLICAS; ++currReplica) {
        numEpochs = 0;

        // Generate the initial population of currentCandidates
        generateInitialPopulation();

        for (int i = 0; i < NUM_INDEXES; i++) replicas[i] = boxMuller();

        // Run until one of the stopping criteria is fulfilled
        while (true) {
            evaluate(replicas);
            if (canStop()) break;
            reproduction();
            mutation();

            ++numEpochs;
        }

        printf("Replica %d:\n", currReplica);
        printf("\tBest candidate had fitness %lf\n", currentCandidates[0]->fitness);
        printf("\tReH = %lf\n", currentCandidates[0]->cffs[ReH]);
        printf("\tReE = %lf\n", currentCandidates[0]->cffs[ReE]);
        printf("\tReHtilde = %lf\n", currentCandidates[0]->cffs[ReHtilde]);

        bestCandidatesByReplica[currReplica]->cffs[ReH] = currentCandidates[0]->cffs[ReH];
        bestCandidatesByReplica[currReplica]->cffs[ReE] = currentCandidates[0]->cffs[ReE];
        bestCandidatesByReplica[currReplica]->cffs[ReHtilde] = currentCandidates[0]->cffs[ReHtilde];
        bestCandidatesByReplica[currReplica]->fitness = currentCandidates[0]->fitness;
    }

    calcMeanAndStdDev();

    printf("ReH_mean = %lf, ReH_stddev = %lf\n", ReH_mean, ReH_stddev);
    printf("ReE_mean = %lf, ReE_stddev = %lf\n", ReE_mean, ReE_stddev);
    printf("ReHtilde_mean = %lf, ReHtilde_stddev = %lf\n", ReHtilde_mean, ReHtilde_stddev);

    const char * const geneticAlgoOutputFilename = "genetic_algo_output.csv";
    FILE *f = fopen(geneticAlgoOutputFilename, "r");

    if (f == NULL) {
        f = fopen(geneticAlgoOutputFilename, "w");
        fprintf(f, "Set,k,QQ,x,t,ReH_mean,ReH_stddev,ReE_mean,ReE_stddev,ReHtilde_mean,ReHtilde_stddev\n");
    } else {
        fclose(f);
        f = fopen(geneticAlgoOutputFilename, "a");
    }

    fprintf(f, "%d,%lf,%lf,%lf,%lf,%lf,%lf\n", desiredSet, ReH_mean, ReH_stddev, ReE_mean, ReE_stddev, ReHtilde_mean, ReHtilde_stddev);

    fclose(f);

    FILE *f2 = fopen("idk.csv", "w");
    fprintf(f2, "ReH,ReE,ReHtilde,Fitness\n");
    for (uint16_t replicaNumber = 0; replicaNumber < NUM_REPLICAS; ++replicaNumber) {
        fprintf(f2, "%lf,%lf,%lf,%lf\n", bestCandidatesByReplica[replicaNumber]->cffs[ReH], bestCandidatesByReplica[replicaNumber]->cffs[ReE], bestCandidatesByReplica[replicaNumber]->cffs[ReHtilde], bestCandidatesByReplica[replicaNumber]->fitness);
    }

    fclose(f2);


exit:
    // Free dynamically allocated variables
    for (uint16_t candidateIndex = 0; candidateIndex < NUM_CANDIDATES; ++candidateIndex) {
        free(currentCandidates[candidateIndex]);
    }

    for (uint16_t replicaNumber = 0; replicaNumber < NUM_REPLICAS; ++replicaNumber) {
        free(bestCandidatesByReplica[replicaNumber]);
    }

    return 0;
}

// Read in the data
// returns 0 if success, 1 if error
static bool readInData() {
    const char * const pseudodataFilename = "dvcs_xs_May-2021_342_sets.csv";

    FILE *f = fopen(pseudodataFilename, "r");
    char buff[1024] = {0};
    bool previouslyEncounteredSet = 0;

    if (f == NULL) {
        printf("Error: could not open file %s.\n", pseudodataFilename);
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
        printf("Error: could not find set %d in %s.\n", desiredSet, pseudodataFilename);
        return 1;
    }

    // Format of local fit output file:
    // set#,k,QQ,x,t,ReH_mean,ReH_error,ReE_mean,ReE_error,ReHtilde_mean,ReHtilde_error
    const char * const localFitOutputFilename = "local_fit_output.csv";

    FILE *f2 = fopen(localFitOutputFilename, "r");
    char buff2[1024] = {0};

    if (f2 == NULL) {
        printf("Error: could not open file %s.\n", localFitOutputFilename);
        return 1;
    }

    fgets(buff2, 1024, f2);

    while (fgets(buff2, 1024, f2)) {
        int set;

        sscanf(buff2, "%d,%*s\n", &set);

        if (set == desiredSet) {
            

            sscanf(buff2, "%*hu,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
                &k, &QQ, &x, &t,
                &meanValues[ReH], &stddevValues[ReH],
                &meanValues[ReE], &stddevValues[ReE],
                &meanValues[ReHtilde], &stddevValues[ReHtilde]);

            fclose(f2);
            
            return 0;
        }
    }

    fclose(f2);

    return 1;
}


// Box-Muller Transform --> Turn a uniform distribution into a Gaussian distribution
// Returns a random number following a Gaussian distribution with mean=0 and stddev=1
static double boxMuller(void) {
    return sqrt(-2 * log(rand() / ((double) RAND_MAX))) * sin(2 * M_PI * (rand() / ((double) RAND_MAX)));
}




static void generateInitialPopulation(void) {
    // Generate initial values for the CFFs that are sampled from a uniform 
    // distribution in the range (minValues[cff], maxValues[cff])
    for (uint16_t candidateIndex = 0; candidateIndex < NUM_CANDIDATES; ++candidateIndex) {
        createNewCandidate(candidateIndex);
    }
}




// Calculate the root root mean squared percent error in F for a specific replica and the given CFFs
static double fitness(const uint16_t candidateIndex, double * const restrict replicas) {
    double sum_of_squared_percent_errors = 0.0;

    for (int index = 0; index < NUM_INDEXES; index++) {
        //printf("%d %lf %lf %lf %lf %lf %lf\n", phi[index], F1, F2, currentCandidates[candidateIndex]->cffs[ReH], currentCandidates[candidateIndex]->cffs[ReE], currentCandidates[candidateIndex]->cffs[ReHtilde], dvcs);

        double F_predicted = TVA1_UU_GetBHUU(phi[index], F1, F2) +
                             TVA1_UU_GetIUU(phi[index], F1, F2, currentCandidates[candidateIndex]->cffs[ReH], currentCandidates[candidateIndex]->cffs[ReE], currentCandidates[candidateIndex]->cffs[ReHtilde]) +
                             dvcs;

        //printf("%lf %lf\n", TVA1_UU_GetBHUU(phi[index], F1, F2), );

        double F_actual = F[index] + (replicas[index] * errF[index]);

        double percent_error = fabs((F_actual - F_predicted) / F_actual);

        //printf("%lf %lf %lf\n", F_predicted, F_actual, percent_error);

        sum_of_squared_percent_errors += percent_error * percent_error;
    }

    double rmse = sqrt(sqrt(sum_of_squared_percent_errors / NUM_INDEXES));

    //printf("rmse = %lf\n", rmse);

    return (rmse > 1.0) ? 0.0 : 1.0 - rmse;
}




static bool canStop(void) {
    // Return if any of the stopping criteria is fulfilled
    return 
        // fitness score threshold
        currentCandidates[0]->fitness > FITNESS_SCORE_THRESHOLD ||
        
        // number of epochs threshold
        numEpochs >= NUMBER_OF_EPOCHS_THRESHOLD;
}



static void evaluate(double * const restrict replicas) {
    // Find the fitness of the first candidate
    currentCandidates[0]->fitness = fitness(0, replicas);

    // Find the fitness of all other currentCandidates and perform insertion sort over the currentCandidates' fitness
    for (uint16_t candidateIndex = 1; candidateIndex < NUM_CANDIDATES; ++candidateIndex) {
        currentCandidates[candidateIndex]->fitness = fitness(candidateIndex, replicas);
        
        for (uint16_t j = candidateIndex; (j > 0) && (currentCandidates[candidateIndex]->fitness > currentCandidates[candidateIndex - 1]->fitness); j--) {
            Candidate * const temp = currentCandidates[candidateIndex];
            currentCandidates[candidateIndex] = currentCandidates[candidateIndex - 1];
            currentCandidates[candidateIndex - 1] = temp;
        }
    }
}



static void reproduction(void) {
    // replace the worst-performing candidates with new candidates that 
    // take genes from the best-performing currentCandidates
    const uint16_t endingIndexOfBestPerforming = (uint16_t) (NUM_CANDIDATES * FRACTION_OF_CANDIDATES_TO_REMOVE);
    const uint16_t startingIndexOfWorstPerforming = NUM_CANDIDATES - endingIndexOfBestPerforming;

    for (uint16_t index = startingIndexOfWorstPerforming; index < NUM_CANDIDATES; ++index) {
        for (uint8_t cff = 0; cff < NUM_CFFS; ++cff) {
            currentCandidates[index]->cffs[cff] = currentCandidates[rand() % endingIndexOfBestPerforming]->cffs[cff];
        }
    }
}



static void mutation(void) {
    const uint16_t uInt16TMax = 65535;
    const uint16_t startingIndex = (uint16_t) (NUM_CANDIDATES * PERCENT_IMMUNE_TO_MUTATION);
    const uint16_t thresholdForMutation = uInt16TMax - ((uint16_t) (MUTATION_PROBABILITY * uInt16TMax));

    for (uint16_t candidateIndex = startingIndex; candidateIndex < NUM_CANDIDATES; ++candidateIndex) {
        for (uint8_t cff = 0; cff < NUM_CFFS; ++cff) {
            if (rand() % uInt16TMax > thresholdForMutation) {
                currentCandidates[candidateIndex]->cffs[cff] = meanValues[cff] + (boxMuller() * stddevValues[cff] / 2.);
            }
        }
    }
}



static void createNewCandidate(const uint16_t candidateIndex) {
    for (uint8_t cff = 0; cff < NUM_CFFS; ++cff) {
        currentCandidates[candidateIndex]->cffs[cff] = meanValues[cff] + (boxMuller() * stddevValues[cff] / 2.);
    }
}

// Function that calculates the mean and standard deviation of the CFFs over all the replicas
static void calcMeanAndStdDev(void) {
    ReH_mean = ReE_mean = ReHtilde_mean = 0.0;

    double ReH_sum_of_differences = 0.0, ReE_sum_of_differences = 0.0, ReHtilde_sum_of_differences = 0.0;

    for (uint16_t replicaNumber = 0; replicaNumber < NUM_REPLICAS; ++replicaNumber) {
        ReH_mean += bestCandidatesByReplica[replicaNumber]->cffs[ReH];
        ReE_mean += bestCandidatesByReplica[replicaNumber]->cffs[ReE];
        ReHtilde_mean += bestCandidatesByReplica[replicaNumber]->cffs[ReHtilde];
    }

    ReH_mean /= NUM_REPLICAS;
    ReE_mean /= NUM_REPLICAS;
    ReHtilde_mean /= NUM_REPLICAS;

    for (uint16_t replicaNumber = 0; replicaNumber < NUM_REPLICAS; ++replicaNumber) {
        ReH_sum_of_differences += (bestCandidatesByReplica[replicaNumber]->cffs[ReH] - ReH_mean) * (bestCandidatesByReplica[replicaNumber]->cffs[ReH] - ReH_mean);
        ReE_sum_of_differences += (bestCandidatesByReplica[replicaNumber]->cffs[ReE] - ReE_mean) * (bestCandidatesByReplica[replicaNumber]->cffs[ReE] - ReE_mean);
        ReHtilde_sum_of_differences += (bestCandidatesByReplica[replicaNumber]->cffs[ReHtilde] - ReHtilde_mean) * (bestCandidatesByReplica[replicaNumber]->cffs[ReHtilde] - ReHtilde_mean);
    }

    ReH_stddev = sqrt(ReH_sum_of_differences / (NUM_REPLICAS - 1));
    ReE_stddev = sqrt(ReE_sum_of_differences / (NUM_REPLICAS - 1));
    ReHtilde_stddev = sqrt(ReHtilde_sum_of_differences / (NUM_REPLICAS - 1));
}
