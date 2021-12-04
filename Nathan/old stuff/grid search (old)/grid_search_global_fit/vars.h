#ifndef VARS_H_
#define VARS_H_

#include "BHDVCS.h"

#define NUM_SETS 15
#define NUM_INDEXES 36

typedef struct {
    double ReH, ReE, ReHtilde, error;
} CFFGuess;

typedef struct {
    double p[9], error;
} FitParametersGuess;

extern const int numSets;

extern double k[], QQ[], x_b[], t[],
       F[][NUM_INDEXES], errF[][NUM_INDEXES],
       F1[], F2[],
       dvcs,
       ReH[], ReE[], ReHtilde[];

extern BHDVCS bhdvcs[];
extern CFFGuess cffGuesses[];
extern FitParametersGuess fitParametersGuesses[];

#endif // VARS_H_
