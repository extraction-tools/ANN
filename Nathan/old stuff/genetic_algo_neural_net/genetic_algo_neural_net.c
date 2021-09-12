#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "TVA1_UU.h"

#define NUM_SETS 1

#define NUM_CANDIDATES 100

#define NUM_CFFS 3

#define NUM_INDEXES 45

#define NUM_REPLICAS 100

// The algorithm will stop when the fitess score reaches this value
//#define FITNESS_SCORE_THRESHOLD 99 // Range: [0, 1]

// The algorithm will stop when the number of epochs reaches this value
#define NUMBER_OF_EPOCHS_THRESHOLD 1000

#define FRACTION_OF_CANDIDATES_TO_REMOVE 0.7 // Range: [0, 1]

#define MUTATION_PROBABILITY 0.1 // Range: [0, 1]

#define PERCENT_IMMUNE_TO_MUTATION 0.2 // Range: [0, 1]


#define ReH 0
#define ReE 1
#define ReHtilde 2

// k, QQ, x, t, F1, F2, dvcs, bias
#define NUM_INPUTS 8
#define NUM_HIDDEN_LAYER_NEURONS 6





typedef struct {
    double values[NUM_CFFS];
} CFFs;



// Type `Candidate`: represents one candidate/genotype
typedef struct {
    // Genes
    double hiddenLayerNeuronWeights[NUM_HIDDEN_LAYER_NEURONS][NUM_INPUTS];
    double outputLayerNeuronWeights[NUM_CFFS][NUM_HIDDEN_LAYER_NEURONS];

    // Tracking metadata
    double fitness;
} Candidate;


static Candidate *currentCandidates[NUM_CANDIDATES];
static Candidate *bestCandidatesByReplica[NUM_REPLICAS];





typedef struct {
    double k, QQ, x, t, 
           F[NUM_INDEXES], errF[NUM_INDEXES], 
           F1, F2, 
           dvcs,
           ReH_mean, ReE_mean, ReHtilde_mean,
           ReH_error, ReE_error, ReHtilde_error;
    
    int phi[NUM_INDEXES];
} SetData;

static SetData setData[NUM_SETS + 1];





static int numEpochs = 0, currReplica = 0;





static bool readInData(void);

static double boxMuller(void);

static double fitness(const uint16_t candidateIndex, double * const restrict replicas);

static bool canStop(void);

// Calculate the fitness of each candidate and then sort the currentCandidates array by fitness in non-increasing order
static void evaluate(double * const restrict replicas);

static void reproduction(void);

static void mutation(void);

static void createNewCandidate(const uint16_t candidateIndex);

static void generateInitialPopulation(void);

static void printCandidate(const uint16_t candidateIndex);

static CFFs generateCFFs(const uint16_t candidateIndex, const uint8_t set);






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
    if (argc == 1) {
        if(readInData()) {
            printf("Read failed.\n");
            goto exit;
        }
    } else {
        printf("Do not specify any arguments.\n");
        goto exit;
    }

    



    double replicas[NUM_INDEXES];
    FILE *f = fopen("genetic_algo_neural_net_output.csv", "a");
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
            printf("Finished replica %d, epoch %d\n", currReplica, numEpochs);
        }

        //printf("Replica %d:\n", currReplica);
        //printf("\tBest candidate had fitness %lf\n", currentCandidates[0]->fitness);

        //printCandidate(0);

        
        //fprintf(f, "set,ReH,ReE,ReHtilde\n");
        for (uint8_t set = 1; set <= NUM_SETS; ++set) {
            CFFs cffs = generateCFFs(0, set);
            fprintf(f, "%hhu,%lf,%lf,%lf\n", set, cffs.values[ReH], cffs.values[ReE], cffs.values[ReHtilde]);
        }

        for (int hiddenLayerNeuron = 0; hiddenLayerNeuron < NUM_HIDDEN_LAYER_NEURONS; ++hiddenLayerNeuron) {
            for (int input = 0; input < NUM_INPUTS; ++input) {
                bestCandidatesByReplica[currReplica]->hiddenLayerNeuronWeights[hiddenLayerNeuron][input] = currentCandidates[0]->hiddenLayerNeuronWeights[hiddenLayerNeuron][input];
            }

            for (int cff = 0; cff < NUM_CFFS; ++cff) {
                bestCandidatesByReplica[currReplica]->outputLayerNeuronWeights[cff][hiddenLayerNeuron] = currentCandidates[0]->outputLayerNeuronWeights[cff][hiddenLayerNeuron];
            }
        }
    }


/*
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
*/

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

    if (f == NULL) {
        printf("Error: could not open file %s.\n", pseudodataFilename);
        return 1;
    }

    fgets(buff, 1024, f);

    while (fgets(buff, 1024, f)) {
        int set = -1;
        int index = -1;
        sscanf(buff, "%d,%d,%*s", &set, &index);
        if (set < 0 || set > NUM_SETS || index < 0 || index >= NUM_INDEXES) {
            break;
        }

        if (index == 0) {
            sscanf(
                buff,
                "%*d,%*d,%lf,%lf,%lf,%lf,%*d,%*lf,%*lf,%*lf,%lf,%lf,%lf\n",
                &setData[set].k, &setData[set].QQ, &setData[set].x, &setData[set].t, 
                &setData[set].F1, &setData[set].F2, &setData[set].dvcs
            );
        }

        sscanf(buff, "%*d,%*d,%*lf,%*lf,%*lf,%*lf,%d,%lf,%lf,%*lf,%*lf,%*lf,%*lf\n",
            &setData[set].phi[index], &setData[set].F[index], &setData[set].errF[index]
        );
    }

    fclose(f);

    return 0;
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
    double sumOfRMSEs = 0.0;

    for (uint8_t set = 1; set <= NUM_SETS; ++set) {
        TVA1_UU_SetKinematics(setData[set].QQ, setData[set].x, setData[set].t, setData[set].k);

        CFFs cffs = generateCFFs(candidateIndex, set);

        double sum_of_squared_percent_errors = 0.0;

        for (int index = 0; index < NUM_INDEXES; index++) {
            double F_predicted = 
                TVA1_UU_GetBHUU(setData[set].phi[index], setData[set].F1, setData[set].F2) +
                TVA1_UU_GetIUU(setData[set].phi[index], setData[set].F1, setData[set].F2, cffs.values[ReH], cffs.values[ReE], cffs.values[ReHtilde]) +
                setData[set].dvcs;

            double F_actual = setData[set].F[index];
            //if (replicas != NULL) F_actual += (replicas[index] * setData[set].errF[index]);

            double percent_error = fabs((F_actual - F_predicted) / F_actual);

            sum_of_squared_percent_errors += percent_error * percent_error;
        }

        sumOfRMSEs += sqrt(sum_of_squared_percent_errors / NUM_INDEXES);
    }

    
    // return average rmse
    return -1 * (sumOfRMSEs / NUM_SETS);
}




static bool canStop(void) {
    // Return if any of the stopping criteria is fulfilled
    return 
        // fitness score threshold
        //currentCandidates[0]->fitness > FITNESS_SCORE_THRESHOLD ||
        
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
    // replace the worst-performing candidates with new randomly-generated candidates
    const uint16_t endingIndexOfBestPerforming = (uint16_t) (NUM_CANDIDATES * FRACTION_OF_CANDIDATES_TO_REMOVE);
    const uint16_t startingIndexOfWorstPerforming = NUM_CANDIDATES - endingIndexOfBestPerforming;

    for (uint16_t index = startingIndexOfWorstPerforming; index < NUM_CANDIDATES; ++index) {
        createNewCandidate(index);
    }
}



static void mutation(void) {
    const uint16_t uInt16TMax = 65535;
    const uint16_t startingIndex = (uint16_t) (NUM_CANDIDATES * PERCENT_IMMUNE_TO_MUTATION);
    const uint16_t thresholdForMutation = uInt16TMax - ((uint16_t) (MUTATION_PROBABILITY * uInt16TMax));

    for (uint16_t candidateIndex = startingIndex; candidateIndex < NUM_CANDIDATES; ++candidateIndex) {
        for (int hiddenLayerNeuron = 0; hiddenLayerNeuron < NUM_HIDDEN_LAYER_NEURONS; ++hiddenLayerNeuron) {
            for (int input = 0; input < NUM_INPUTS; ++input) {
                if (rand() % uInt16TMax > thresholdForMutation) {
                    // random number in the range [-10, 10]
                    currentCandidates[candidateIndex]->hiddenLayerNeuronWeights[hiddenLayerNeuron][input] = ((rand() / (double) RAND_MAX) * 20) - 10;
                }
            }

            for (int cff = 0; cff < NUM_CFFS; ++cff) {
                if (rand() % uInt16TMax > thresholdForMutation) {
                    // random number in the range [-10, 10]
                    currentCandidates[candidateIndex]->outputLayerNeuronWeights[cff][hiddenLayerNeuron] = ((rand() / (double) RAND_MAX) * 20) - 10;
                }
            }
        }
    }
}



static void createNewCandidate(const uint16_t candidateIndex) {
    for (int hiddenLayerNeuron = 0; hiddenLayerNeuron < NUM_HIDDEN_LAYER_NEURONS; ++hiddenLayerNeuron) {
        for (int input = 0; input < NUM_INPUTS; ++input) {
            // random number in the range [-10, 10]
            currentCandidates[candidateIndex]->hiddenLayerNeuronWeights[hiddenLayerNeuron][input] = ((rand() / (double) RAND_MAX) * 20) - 10;
        }

        for (int cff = 0; cff < NUM_CFFS; ++cff) {
            // random number in the range [-10, 10]
            currentCandidates[candidateIndex]->outputLayerNeuronWeights[cff][hiddenLayerNeuron] = ((rand() / (double) RAND_MAX) * 20) - 10;
        }
    }
}


static void printCandidate(const uint16_t candidateIndex) {
    printf("Candidate %hu:\n", candidateIndex);
    printf("Fitness: %lf\n", currentCandidates[candidateIndex]->fitness);
    printf("\thiddenLayerNeuronWeights:\n");
    for (int hiddenLayerNeuron = 0; hiddenLayerNeuron < NUM_HIDDEN_LAYER_NEURONS; ++hiddenLayerNeuron) {
        printf("\t\thiddenLayerNeuronWeights[%d]:\n", hiddenLayerNeuron);
        for (int input = 0; input < NUM_INPUTS; ++input) {
            printf("\t\t\thiddenLayerNeuronWeights[%d][%d]: %lf\n", 
                hiddenLayerNeuron, 
                input, 
                currentCandidates[candidateIndex]->hiddenLayerNeuronWeights[hiddenLayerNeuron][input]
            );
        }
    }

    printf("\toutputLayerNeuronWeights:\n");
    for (int cff = 0; cff < NUM_CFFS; ++cff) {
        printf("\t\toutputLayerNeuronWeights[%d]:\n", cff);
        for (int hiddenLayerNeuron = 0; hiddenLayerNeuron < NUM_HIDDEN_LAYER_NEURONS; ++hiddenLayerNeuron) {
            printf("\t\t\toutputLayerNeuronWeights[%d][%d]: %lf\n", 
                cff,
                hiddenLayerNeuron, 
                currentCandidates[candidateIndex]->outputLayerNeuronWeights[cff][hiddenLayerNeuron]
            );
        }
    }
}



static CFFs generateCFFs(const uint16_t candidateIndex, const uint8_t set) {
    // inputs: k, QQ, x, t, F1, F2, dvcs, bias
    double inputs[NUM_INPUTS] = {
        setData[set].k,
        setData[set].QQ,
        setData[set].x,
        setData[set].t,
        setData[set].F1,
        setData[set].F2,
        setData[set].dvcs,
        1.0
    };
    double hiddenLayerOutputs[NUM_HIDDEN_LAYER_NEURONS] = {0.0};
    double outputLayerOutputs[NUM_CFFS] = {0.0};

    for (int hiddenLayerNeuron = 0; hiddenLayerNeuron < NUM_HIDDEN_LAYER_NEURONS; ++hiddenLayerNeuron) {
        for (int input = 0; input < NUM_INPUTS; ++input) {
            hiddenLayerOutputs[hiddenLayerNeuron] += 
                inputs[input] * 
                currentCandidates[candidateIndex]->hiddenLayerNeuronWeights[hiddenLayerNeuron][input];
        }
    }

    for (int outputLayerNeuron = 0; outputLayerNeuron < NUM_CFFS; ++outputLayerNeuron) {
        for (int hiddenLayerNeuron = 0; hiddenLayerNeuron < NUM_HIDDEN_LAYER_NEURONS; ++hiddenLayerNeuron) {
            outputLayerOutputs[outputLayerNeuron] += 
                hiddenLayerOutputs[hiddenLayerNeuron] * 
                currentCandidates[candidateIndex]->outputLayerNeuronWeights[outputLayerNeuron][hiddenLayerNeuron];
        }
    }

    CFFs cffs;
    for (int cff = 0; cff < NUM_CFFS; ++cff) {
        cffs.values[cff] = outputLayerOutputs[cff];
    }

    return cffs;
}
