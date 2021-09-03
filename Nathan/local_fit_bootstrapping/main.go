package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

const NUM_INDEXES int = 45
const NUM_ATTEMPTS int = 1000
const NUM_INDEXES_IN_ONE_ATTEMPT int = 50

var k, QQ, x, t, F1, F2, dvcs float64
var F, errF [NUM_INDEXES]float64
var phi [NUM_INDEXES]int

const ReH_true float64 = 0.743515
const ReE_true float64 = -1.58545
const ReHtilde_true float64 = 3.34929

var F_true float64

var r *rand.Rand = rand.New(rand.NewSource(time.Now().UnixNano()))

const desiredSet int = 1

/*
func boxMuller() float64 {
	return math.Sqrt(-2.0*math.Log(r.Float64())) * math.Sin(2.0*math.Pi*(r.Float64()))
}

func getAttemptFValue(index int) float64 {
	return F[index] + (errF[index] * boxMuller())
}
*/

func getBestCFFs(indexes [NUM_INDEXES_IN_ONE_ATTEMPT]int, cffsChan chan<- [4]float64) {
	var phiValues [NUM_INDEXES_IN_ONE_ATTEMPT]int
	var fValues [NUM_INDEXES_IN_ONE_ATTEMPT]float64

	for index, value := range indexes {
		phiValues[index] = phi[value]
		fValues[index] = F[value] //getAttemptFValue(value)
	}

	cffsChan <- gridSearch(phiValues, fValues)
}

func main() {
	var cffsChannel = make(chan [4]float64)
	var numAttemptsCompleted int = 0

	var ReH_values [NUM_ATTEMPTS]float64
	var ReE_values [NUM_ATTEMPTS]float64
	var ReHtilde_values [NUM_ATTEMPTS]float64

	csvfile, err := os.Open("dvcs_xs_May-2021_342_sets.csv")
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}

	// Parse the file
	csvReader := csv.NewReader(csvfile)

	numIndexesCompleted := 0

	csvReader.Read()

	// Iterate through the records
	for {
		if numIndexesCompleted >= NUM_INDEXES {
			break
		}

		// Read each record from csv
		record, err := csvReader.Read()
		if err == io.EOF {
			fmt.Println(err)
			break
		}
		if err != nil {
			fmt.Println(err)
			log.Fatal(err)
		}

		val, err := strconv.Atoi(record[0])
		if err != nil || val > desiredSet {
			fmt.Println(err)
			break
		} else if val == desiredSet {
			if numIndexesCompleted == 0 {
				k, _ = strconv.ParseFloat(record[2], 64)
				QQ, _ = strconv.ParseFloat(record[3], 64)
				x, _ = strconv.ParseFloat(record[4], 64)
				t, _ = strconv.ParseFloat(record[5], 64)

				F1, _ = strconv.ParseFloat(record[10], 64)
				F2, _ = strconv.ParseFloat(record[11], 64)
				dvcs, _ = strconv.ParseFloat(record[12], 64)
			}

			phi[numIndexesCompleted], _ = strconv.Atoi(record[6])
			F[numIndexesCompleted], _ = strconv.ParseFloat(record[7], 64)
			errF[numIndexesCompleted], _ = strconv.ParseFloat(record[8], 64)

			numIndexesCompleted++
		}
	}

	TVA1_UU_SetKinematics(QQ, x, t, k)

	for i := 0; i < NUM_INDEXES; i++ {
		F[i] = TVA1_UU_GetBHUU(float64(phi[i]), F1, F2) + TVA1_UU_GetIUU(float64(phi[i]), F1, F2, ReH_true, ReE_true, ReHtilde_true) + dvcs
		errF[i] = 0.0
	}

	for i := 0; i < NUM_ATTEMPTS; i++ {
		var indexes [NUM_INDEXES_IN_ONE_ATTEMPT]int
		for i := 0; i < NUM_INDEXES_IN_ONE_ATTEMPT; i++ {
			indexes[i] = r.Intn(NUM_INDEXES)
		}

		go getBestCFFs(indexes, cffsChannel)
	}

	for {
		select {
		case cffs := <-cffsChannel:
			ReH_values[numAttemptsCompleted] = cffs[0]
			ReE_values[numAttemptsCompleted] = cffs[1]
			ReHtilde_values[numAttemptsCompleted] = cffs[2]
			numAttemptsCompleted++
			fmt.Printf("%.6f,%.6f,%.6f,%.6f\n", cffs[0], cffs[1], cffs[2], cffs[3])
		default:
			time.Sleep(50 * time.Millisecond)
		}

		if numAttemptsCompleted == NUM_ATTEMPTS {
			var cffMeans, cffStddevs [3]float64 = calcMeanAndStddevOfCFFs(ReH_values, ReE_values, ReHtilde_values)

			fmt.Printf("ReH = %.6f +/- %.6f\n", cffMeans[0], cffStddevs[0])
			fmt.Printf("ReE = %.6f +/- %.6f\n", cffMeans[1], cffStddevs[1])
			fmt.Printf("ReHtilde = %.6f +/- %.6f\n", cffMeans[2], cffStddevs[2])
			break
		}
	}
}

func calcMeanAndStddevOfCFFs(ReH_values, ReE_values, ReHtilde_values [NUM_ATTEMPTS]float64) ([3]float64, [3]float64) {
	cffMeanValues, cffStddevValues, cffSumsOfDifferences := [3]float64{0.0, 0.0, 0.0}, [3]float64{0.0, 0.0, 0.0}, [3]float64{0.0, 0.0, 0.0}

	for attemptNum := 0; attemptNum < NUM_ATTEMPTS; attemptNum++ {
		cffMeanValues[0] += ReH_values[attemptNum]
		cffMeanValues[1] += ReE_values[attemptNum]
		cffMeanValues[2] += ReHtilde_values[attemptNum]
	}

	cffMeanValues[0] /= float64(NUM_ATTEMPTS)
	cffMeanValues[1] /= float64(NUM_ATTEMPTS)
	cffMeanValues[2] /= float64(NUM_ATTEMPTS)

	for attemptNum := 0; attemptNum < NUM_ATTEMPTS; attemptNum++ {
		cffSumsOfDifferences[0] += (ReH_values[attemptNum] - cffMeanValues[0]) * (ReH_values[attemptNum] - cffMeanValues[0])
		cffSumsOfDifferences[1] += (ReE_values[attemptNum] - cffMeanValues[1]) * (ReE_values[attemptNum] - cffMeanValues[1])
		cffSumsOfDifferences[2] += (ReHtilde_values[attemptNum] - cffMeanValues[2]) * (ReHtilde_values[attemptNum] - cffMeanValues[2])
	}

	cffStddevValues[0] = math.Sqrt(cffSumsOfDifferences[0] / float64(NUM_ATTEMPTS-1))
	cffStddevValues[1] = math.Sqrt(cffSumsOfDifferences[1] / float64(NUM_ATTEMPTS-1))
	cffStddevValues[2] = math.Sqrt(cffSumsOfDifferences[2] / float64(NUM_ATTEMPTS-1))

	return cffMeanValues, cffStddevValues
}
