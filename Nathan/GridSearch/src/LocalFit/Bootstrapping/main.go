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

const NUM_INDEXES int = 36
const NUM_ATTEMPTS int = 100
const NUM_INDEXES_IN_ONE_ATTEMPT int = 100

var k, QQ, x, t, F1, F2, dvcs float64
var F, errF [NUM_INDEXES]float64
var phi [NUM_INDEXES]int

var desiredSet int

func boxMuller(r *rand.Rand) float64 {
	return math.Sqrt(-2.0*math.Log(r.Float64())) * math.Sin(2.0*math.Pi*(r.Float64()))
}

func getBestCFFs(indexes [NUM_INDEXES_IN_ONE_ATTEMPT]int, bhdvcs BHDVCS, cffsChan chan<- [4]float64) {
	var phiValues [NUM_INDEXES_IN_ONE_ATTEMPT]float64
	var fValues [NUM_INDEXES_IN_ONE_ATTEMPT]float64

	var r *rand.Rand = rand.New(rand.NewSource(time.Now().UnixNano()))

	for index, value := range indexes {
		phiValues[index] = float64(phi[value])
		fValues[index] = F[value] + (errF[value] * boxMuller(r))
	}

	cffsChan <- gridSearch(phiValues, fValues, bhdvcs)
}

func main() {
	desiredSet, _ = strconv.Atoi(os.Args[1])
	var cffsChannel = make(chan [4]float64)
	var numAttemptsCompleted int = 0

	var ReH_values [NUM_ATTEMPTS]float64
	var ReE_values [NUM_ATTEMPTS]float64
	var ReHtilde_values [NUM_ATTEMPTS]float64

	csvfile, err := os.Open("lib/dvcs_xs_newsets_withCFFs_2.csv") // Pseudodata 1
	//csvfile, err := os.Open("lib/dvcs_xs_May-2021_342_sets.csv") // Pseudodata 2
	//csvfile, err := os.Open("lib/dvcs_bkm2002_June2021_4pars.csv") // Pseudodata 3

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

	var bhdvcs BHDVCS

	var phiValuesArr []float64 = make([]float64, len(phi))
	for index, value := range phi {
		phiValuesArr[index] = float64(value)
	}
	bhdvcs.init(QQ, x, t, k, F1, F2, phiValuesArr)

	var r *rand.Rand = rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < NUM_ATTEMPTS; i++ {
		var indexes [NUM_INDEXES_IN_ONE_ATTEMPT]int
		for i := 0; i < NUM_INDEXES_IN_ONE_ATTEMPT; i++ {
			indexes[i] = r.Intn(NUM_INDEXES)
		}

		go getBestCFFs(indexes, bhdvcs, cffsChannel)
	}

	f, err := os.OpenFile("local_fit_bootstrapping_output.csv", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Println(err)
	}
	defer f.Close()

	for {
		select {
		case cffs := <-cffsChannel:
			ReH_values[numAttemptsCompleted] = cffs[0]
			ReE_values[numAttemptsCompleted] = cffs[1]
			ReHtilde_values[numAttemptsCompleted] = cffs[2]
			numAttemptsCompleted++
		default:
			time.Sleep(50 * time.Millisecond)
		}

		if numAttemptsCompleted == NUM_ATTEMPTS {
			var cffMeans, cffStddevs [3]float64 = calcMeanAndStddevOfCFFs(ReH_values, ReE_values, ReHtilde_values)

			f.WriteString(fmt.Sprintf("%d,%d,%d,%f,%f,%f,%f,%f,%f\n",
				desiredSet,
				NUM_ATTEMPTS, NUM_INDEXES_IN_ONE_ATTEMPT,
				cffMeans[0], cffStddevs[0],
				cffMeans[1], cffStddevs[1],
				cffMeans[2], cffStddevs[2]))

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
