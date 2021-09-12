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
	"strings"
	"time"
	"unicode"
)

const NUM_INDEXES int = 45

var setNumToSetIndex map[int]int
var sets []int
var k, QQ, x, t, F1, F2, dvcs []float64
var F, errF [][NUM_INDEXES]float64
var phi [][NUM_INDEXES]int

var numSets, numReplicasPerSet int

func main() {
	parseArgs(os.Args)

	readInData()

	// Schema: set #, ReH mean, ReH stddev, ReE mean, ReE stddev, ReHtilde mean, ReHtilde stddev
	gridSearchBySetChannel := make(chan [7]float64)

	for setIndex := 0; setIndex < numSets; setIndex++ {
		go gridSearchBySet(setIndex, gridSearchBySetChannel)
	}

	var numSetsLeftToComplete int = numSets
	for numSetsLeftToComplete > 0 {
		select {
		case gridSearchSetResult := <-gridSearchBySetChannel:
			numSetsLeftToComplete--
			fmt.Printf("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
				int(gridSearchSetResult[0]),
				gridSearchSetResult[1],
				gridSearchSetResult[2],
				gridSearchSetResult[3],
				gridSearchSetResult[4],
				gridSearchSetResult[5],
				gridSearchSetResult[6],
			)
		default:
			time.Sleep(500 * time.Millisecond)
		}
	}
}

func parseArgs(args []string) {
	var err error = nil

	if len(args) != 3 {
		if len(args) > 3 {
			fmt.Println("Error: Too many command-line arguments provided")
		} else {
			fmt.Println("Error: Not enough command-line arguments provided")
		}

		fmt.Println("Include (1) the list of sets to run the local fit on and (2) the number of replicas per set")
		os.Exit(1)
	}

	var inputtedSetsList string = os.Args[1]
	numReplicasPerSet, err = strconv.Atoi(os.Args[2])
	if err != nil || numReplicasPerSet <= 0 {
		fmt.Println("Error: Illegal number of replicas per set")
		fmt.Println("The number of replicas per set must be a positive integer")
		os.Exit(1)
	}

	var commaRune rune = rune(',')
	var dashRune rune = rune('-')
	for index, char := range inputtedSetsList {
		if !unicode.IsDigit(char) && char != commaRune && char != dashRune {
			fmt.Printf("Error: Character %c at position %d is invalid in the sets list\n", char, index)
			fmt.Println("Acceptable characters are 0-9, ',', and '-'")
			os.Exit(1)
		}
	}

	var setsListSplitByCommas []string = strings.Split(inputtedSetsList, ",")
	setNumToSetIndex = make(map[int]int)
	for index, str := range setsListSplitByCommas {
		var dashCount int = strings.Count(str, "-")

		if dashCount == 0 {
			num, err := strconv.Atoi(str)

			if err == nil {
				setNumToSetIndex[num] = -1
			} else {
				fmt.Printf("Error: list of sets index %d: invalid number\n", index)
				os.Exit(1)
			}
		} else if dashCount == 1 {
			var numsAsStrings []string = strings.Split(str, "-")

			num1, err1 := strconv.Atoi(numsAsStrings[0])
			num2, err2 := strconv.Atoi(numsAsStrings[1])

			if err1 == nil && err2 == nil {
				for i := num1; i <= num2; i++ {
					setNumToSetIndex[i] = -1
				}
			} else {
				fmt.Printf("Error: list of sets index %d: invalid number\n", index)
				os.Exit(1)
			}
		} else {
			fmt.Printf("Error: list of sets index %d: cannot use multiple dashes in one range expression\n", index)
			os.Exit(1)
		}
	}

	numSets = len(setNumToSetIndex)

	sets = make([]int, numSets)
	var setIndex int = 0
	for setNum := range setNumToSetIndex {
		sets[setIndex] = setNum
		setNumToSetIndex[setNum] = setIndex
		setIndex++
	}

	k = make([]float64, numSets)
	QQ = make([]float64, numSets)
	x = make([]float64, numSets)
	t = make([]float64, numSets)
	F1 = make([]float64, numSets)
	F2 = make([]float64, numSets)
	dvcs = make([]float64, numSets)

	F = make([][NUM_INDEXES]float64, numSets)
	errF = make([][NUM_INDEXES]float64, numSets)
	phi = make([][NUM_INDEXES]int, numSets)
}

func readInData() {
	executablePath, err := os.Executable()
	if err != nil {
		log.Fatalln("Error: Could not find path of executable")
		return
	}

	trimmedExecutablePath := strings.TrimSuffix(executablePath, "GridSearch/bin/LocalFitParallel")
	if trimmedExecutablePath == executablePath {
		log.Fatalln("Error: Executable should be named LocalFitParallel and be in the directory GridSearch/bin/")
		return
	}

	inputFilePath := trimmedExecutablePath + "GridSearch/lib/dvcs_xs_May-2021_342_sets.csv"
	inputFile, err := os.Open(inputFilePath)
	if err != nil {
		log.Fatalf("Error: Cannot locate the data file \"%s\"\n", inputFilePath)
		return
	}

	// Parse the input file as a csv
	var csvReader *csv.Reader = csv.NewReader(inputFile)

	// Discard first line of csv file
	csvReader.Read()

	var numIndexesCompleted []int = make([]int, numSets)
	// Iterate through the records
	for {
		// Read each record from csv
		record, err := csvReader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			fmt.Println(err)
			log.Fatal(err)
		}

		val, err := strconv.Atoi(record[0])
		setIndex, valInSetMap := setNumToSetIndex[val]
		if err != nil {
			log.Fatalln(err)
			return
		} else if err == nil && valInSetMap {
			if numIndexesCompleted[setIndex] == 0 {
				k[setIndex], _ = strconv.ParseFloat(record[2], 64)
				QQ[setIndex], _ = strconv.ParseFloat(record[3], 64)
				x[setIndex], _ = strconv.ParseFloat(record[4], 64)
				t[setIndex], _ = strconv.ParseFloat(record[5], 64)

				F1[setIndex], _ = strconv.ParseFloat(record[10], 64)
				F2[setIndex], _ = strconv.ParseFloat(record[11], 64)
				dvcs[setIndex], _ = strconv.ParseFloat(record[12], 64)
			}

			phi[setIndex][numIndexesCompleted[setIndex]], _ = strconv.Atoi(record[6])
			F[setIndex][numIndexesCompleted[setIndex]], _ = strconv.ParseFloat(record[7], 64)
			errF[setIndex][numIndexesCompleted[setIndex]], _ = strconv.ParseFloat(record[8], 64)

			numIndexesCompleted[setIndex]++
		}
	}
}

func gridSearchBySet(setIndex int, gridSearchBySetChannel chan<- [7]float64) {
	var ret [7]float64
	ret[0] = float64(sets[setIndex])

	ReH_values := make([]float64, numReplicasPerSet)
	ReE_values := make([]float64, numReplicasPerSet)
	ReHtilde_values := make([]float64, numReplicasPerSet)

	gridSearchByReplicaChannel := make(chan [3]float64)

	var tva1_uu TVA1_UU
	var phis []int = make([]int, len(phi[setIndex]))
	for index, value := range phi[setIndex] {
		phis[index] = value
	}
	tva1_uu.init(QQ[setIndex], x[setIndex], t[setIndex], k[setIndex], phis)

	for i := 0; i < numReplicasPerSet; i++ {
		go gridSearchByReplica(setIndex, &tva1_uu, gridSearchByReplicaChannel)
	}

	var numReplicasCompleted int = 0
	for numReplicasCompleted < numReplicasPerSet {
		select {
		case cffs := <-gridSearchByReplicaChannel:
			ReH_values[numReplicasCompleted] = cffs[0]
			ReE_values[numReplicasCompleted] = cffs[1]
			ReHtilde_values[numReplicasCompleted] = cffs[2]
			numReplicasCompleted++
		default:
			time.Sleep(50 * time.Millisecond)
		}
	}

	var cffMeans, cffStddevs [3]float64 = calcMeanAndStddevOfCFFs(ReH_values, ReE_values, ReHtilde_values)

	gridSearchBySetChannel <- [7]float64{float64(sets[setIndex]), cffMeans[0], cffStddevs[0], cffMeans[1], cffStddevs[1], cffMeans[2], cffStddevs[2]}
}

func gridSearchByReplica(setIndex int, tva1_uu *TVA1_UU, gridSearchByReplicaChannel chan<- [3]float64) {
	var ReHguess, ReEguess, ReHtildeguess float64 = 0.0, 0.0, 0.0
	var bestError float64 = math.MaxFloat64

	var fReplicaValues []float64 = make([]float64, NUM_INDEXES)
	var r *rand.Rand = rand.New(rand.NewSource(time.Now().UnixNano()))
	for index := 0; index < NUM_INDEXES; index++ {
		fReplicaValues[index] = F[setIndex][index] + (errF[setIndex][index] * boxMuller(r))
	}

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

		for (prevError-bestError > 0.000001) || (math.MaxFloat64-bestError < 0.000001) {
			prevError = bestError

			for ReHchange := minChange; ReHchange <= maxChange; ReHchange += dist {
				for ReEchange := minChange; ReEchange <= maxChange; ReEchange += dist {
					for ReHtildechange := minChange; ReHtildechange <= maxChange; ReHtildechange += dist {
						var error float64 = calcFError(setIndex, tva1_uu, ReHguess+ReHchange, ReEguess+ReEchange, ReHtildeguess+ReHtildechange, fReplicaValues)

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

	gridSearchByReplicaChannel <- [3]float64{ReHguess, ReEguess, ReHtildeguess}
}

// Calculate the root root mean squared percent error in F for a specific attempt and the given CFFs
func calcFError(setIndex int, tva1_uu *TVA1_UU, ReH float64, ReE float64, ReHtilde float64, F_actual []float64) float64 {
	sumOfSquaredPercentErrors := 0.0

	for index := 0; index < NUM_INDEXES; index++ {
		F_predicted := dvcs[setIndex] + tva1_uu.getBHUU_plus_getIUU(phi[setIndex][index], F1[setIndex], F2[setIndex], ReH, ReE, ReHtilde)
		percentError := (F_actual[index] - F_predicted) / F_actual[index]
		sumOfSquaredPercentErrors += percentError * percentError
	}

	return math.Sqrt(sumOfSquaredPercentErrors / float64(NUM_INDEXES))
}

func boxMuller(r *rand.Rand) float64 {
	return math.Sqrt(-2.0*math.Log(r.Float64())) * math.Sin(2.0*math.Pi*(r.Float64()))
}

func calcMeanAndStddevOfCFFs(ReH_values, ReE_values, ReHtilde_values []float64) ([3]float64, [3]float64) {
	cffMeanValues, cffStddevValues, cffSumsOfDifferences := [3]float64{0.0, 0.0, 0.0}, [3]float64{0.0, 0.0, 0.0}, [3]float64{0.0, 0.0, 0.0}

	for replicaNum := 0; replicaNum < numReplicasPerSet; replicaNum++ {
		cffMeanValues[0] += ReH_values[replicaNum]
		cffMeanValues[1] += ReE_values[replicaNum]
		cffMeanValues[2] += ReHtilde_values[replicaNum]
	}

	cffMeanValues[0] /= float64(numReplicasPerSet)
	cffMeanValues[1] /= float64(numReplicasPerSet)
	cffMeanValues[2] /= float64(numReplicasPerSet)

	for replicaNum := 0; replicaNum < numReplicasPerSet; replicaNum++ {
		cffSumsOfDifferences[0] += (ReH_values[replicaNum] - cffMeanValues[0]) * (ReH_values[replicaNum] - cffMeanValues[0])
		cffSumsOfDifferences[1] += (ReE_values[replicaNum] - cffMeanValues[1]) * (ReE_values[replicaNum] - cffMeanValues[1])
		cffSumsOfDifferences[2] += (ReHtilde_values[replicaNum] - cffMeanValues[2]) * (ReHtilde_values[replicaNum] - cffMeanValues[2])
	}

	cffStddevValues[0] = math.Sqrt(cffSumsOfDifferences[0] / float64(numReplicasPerSet-1))
	cffStddevValues[1] = math.Sqrt(cffSumsOfDifferences[1] / float64(numReplicasPerSet-1))
	cffStddevValues[2] = math.Sqrt(cffSumsOfDifferences[2] / float64(numReplicasPerSet-1))

	return cffMeanValues, cffStddevValues
}
