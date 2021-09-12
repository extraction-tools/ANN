package main

import (
	"math"
)

// Calculate the root root mean squared percent error in F for a specific attempt and the given CFFs
func calcFError(phiValues [NUM_INDEXES_IN_ONE_ATTEMPT]int, fValues [NUM_INDEXES_IN_ONE_ATTEMPT]float64, ReH float64, ReE float64, ReHtilde float64) float64 {

	sumOfSquaredPercentErrors := 0.0

	for index := 0; index < NUM_INDEXES_IN_ONE_ATTEMPT; index++ {
		F_predicted := TVA1_UU_GetBHUU(float64(phiValues[index]), F1, F2) + TVA1_UU_GetIUU(float64(phiValues[index]), F1, F2, ReH, ReE, ReHtilde) + dvcs
		F_actual := fValues[index]
		percentError := (F_actual - F_predicted) / F_actual
		sumOfSquaredPercentErrors += percentError * percentError
	}

	return math.Sqrt(sumOfSquaredPercentErrors / float64(NUM_INDEXES_IN_ONE_ATTEMPT))
}

// Estimate the correct CFF values for a specific replica
func gridSearch(phiValues [NUM_INDEXES_IN_ONE_ATTEMPT]int, fValues [NUM_INDEXES_IN_ONE_ATTEMPT]float64) [4]float64 {
	var ReHguess, ReEguess, ReHtildeguess float64 = 0.0, 0.0, 0.0
	var bestError float64 = math.MaxFloat64

	// num: number of points tested on either side of the guess
	const num int = 10

	// totalDist: the total distance being tested on either side of the guess
	const totalDist float64 = 1.0

	// dist: distance between each point being tested
	for dist := totalDist / float64(num); dist >= 0.000000001; dist /= float64(num) {
		var maxChange float64 = dist * float64(num)
		var minChange float64 = -1 * maxChange

		var bestReHguess, bestReEguess, bestReHtildeguess float64 = 0.0, 0.0, 0.0

		var prevError float64 = math.MaxFloat64
		bestError = math.MaxFloat64

		for (prevError-bestError > 0.000001) || (math.MaxFloat64-bestError < 0.000100) {
			prevError = bestError

			for ReHchange := minChange; ReHchange <= maxChange; ReHchange += dist {
				for ReEchange := minChange; ReEchange <= maxChange; ReEchange += dist {
					for ReHtildechange := minChange; ReHtildechange <= maxChange; ReHtildechange += dist {
						var error float64 = calcFError(phiValues, fValues, ReHguess+ReHchange, ReEguess+ReEchange, ReHtildeguess+ReHtildechange)

						if error < bestError {
							bestReHguess = ReHguess + ReHchange
							bestReEguess = ReEguess + ReEchange
							bestReHtildeguess = ReHtildeguess + ReHtildechange
							bestError = error
						}
					}
				}
			}

			ReHguess = bestReHguess
			ReEguess = bestReEguess
			ReHtildeguess = bestReHtildeguess
		}
	}

	return [4]float64{ReHguess, ReEguess, ReHtildeguess, bestError}
}
