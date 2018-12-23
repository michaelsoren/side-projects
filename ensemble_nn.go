/*ensemble_nn: This file runs either the parallel or linear versions
              of an ensemble NN.*/
package main

import (
  "fmt"
  "os"
  "io/ioutil"
  "strings"
  "strconv"
  "encoding/csv"
  "nn"
  "nn_utils"
  "parse"
  "sync"
)

/*This is the standard thread arguments.
  Stores everything from data to hyperparameters for training*/
type threadArgs struct {
  id int
  xTrain [][]float64
  yTrain [][]float64
  xVal [][]float64
  yVal [][]float64

  numHiddenLayers int
  numClasses int
  numThreads int
  numIterations int
  reduceEtaAt int
  reduceEtaBy float64
  eta float64
  batchSize int
  hiddenLayerNodes int
  dropout float64

  classifiers []nn.NN
  start int
  end int

  wg *sync.WaitGroup
}

/*
 * Reads the input and sets up for training/prediction
 */
func main() {
  argsWithoutProg := os.Args[1:]

  if len(argsWithoutProg) != 6 && len(argsWithoutProg) != 7 {
    fmt.Println("Too few parameters.")
    return
  }

  trainFp := argsWithoutProg[0]
  testFp := argsWithoutProg[1]

  if argsWithoutProg[2][0:3] != "-n=" {
    fmt.Println("Incorrect input for number of classifiers")
    return
  }
  numClassifiers, _ := strconv.Atoi(argsWithoutProg[2][3:])

  if argsWithoutProg[3][0:3] != "-l=" {
    fmt.Println("Incorrect input for number of layers")
    return
  }
  numHiddenLayers, _ := strconv.Atoi(argsWithoutProg[3][3:])

  if argsWithoutProg[4][0:3] != "-e=" {
    fmt.Println("Incorrect input for eta")
    return
  }
  eta, _ := strconv.ParseFloat(argsWithoutProg[4][3:], 64)

  if argsWithoutProg[5][0:3] != "-c=" {
    fmt.Println("Incorrect input for number of classes")
    return
  }
  numClasses, _ := strconv.Atoi(argsWithoutProg[5][3:])

  numThreads := 1
  if len(argsWithoutProg) == 7 {
    /*Parallel*/
    fmt.Println("Parallel")
    if argsWithoutProg[6][0:3] != "-p=" {
      fmt.Println("Incorrect input for number of threads")
      return
    }
    numThreads, _ = strconv.Atoi(argsWithoutProg[6][3:])
  }
  setupAndRun(trainFp, testFp, numClassifiers, numHiddenLayers, numThreads,
              numClasses, eta)
}

func setupAndRun(trainFp, testFp string, numClassifiers, numHiddenLayers,
                 numThreads, numClasses int, eta float64) {

  /*Open data files*/
  trainF, err := ioutil.ReadFile(trainFp)
  if (err != nil) {
    fmt.Println("Training read failed")
    return
  }
  testF, err := ioutil.ReadFile(testFp)
  if (err != nil) {
    fmt.Println("Test read failed")
    return
  }

  /*Have to convert data files to be [][]floats and 1 hot vector*/
  trainReader := csv.NewReader(strings.NewReader(string(trainF)))
  trainStrings, err := trainReader.ReadAll()
  testReader := csv.NewReader(strings.NewReader(string(testF)))
  testStrings, err := testReader.ReadAll()

  xTrain, yTrain := parse.ParseCsv(trainStrings, numClasses, numThreads)
  /*Grabs final 30% as validation dataset*/
  xVal := xTrain[len(xTrain) / 10 * 7:]
  yVal := yTrain[len(yTrain) / 10 * 7:]
  /*subsets the train data*/
  xTrain = xTrain[:len(xTrain) / 10 * 7]
  yTrain = yTrain[:len(yTrain) / 10 * 7]
  /*grabs the test data*/
  xTest, yTest := parse.ParseCsv(testStrings, numClasses, numThreads)

  classifiers := make([]nn.NN, numClassifiers)

  numIterations := 2000
  reduceEtaAt := numIterations / 5
  reduceEtaBy := 0.8
  batchSize := 100
  threadsPerClassifier := 4

  hiddenLayerNodes := 100
  dropout := 0.09

  /*Print out hyperparameter information*/
  fmt.Println("numIterations", numIterations)
  fmt.Println("reduceEtaAt", reduceEtaAt)
  fmt.Println("reduceEtaBy", reduceEtaBy)
  fmt.Println("Batch size", batchSize)
  fmt.Println("hiddenLayerNodes", hiddenLayerNodes)
  fmt.Println("dropout", dropout)

  var wg sync.WaitGroup

  numThreadGroups := numThreads / threadsPerClassifier
  if numThreadGroups == 0 {
    fmt.Println("Fewer threads than", threadsPerClassifier)
    /*Fewer than threadsPerClassifier threads, used for
      linear and small thread counts*/
    wg.Add(1)
    newArgs := threadArgs{id:0, xTrain:xTrain, yTrain:yTrain,
                          xVal:xVal, yVal:yVal,
                          numHiddenLayers:numHiddenLayers,
                          numClasses:numClasses,
                          numThreads:numThreads,
                          numIterations:numIterations,
                          reduceEtaAt:reduceEtaAt,
                          reduceEtaBy:reduceEtaBy, eta:eta,
                          batchSize:batchSize,
                          hiddenLayerNodes:hiddenLayerNodes,
                          dropout:dropout,
                          classifiers:classifiers,
                          start:0, end:numClassifiers,
                          wg:&wg}
    runTraining(&newArgs)
    wg.Wait()
  } else {
    fmt.Println("As many or more threads than ", threadsPerClassifier)
    numThreadsLeft := numThreads - numThreadGroups * threadsPerClassifier

    /*Use all leftover threads*/
    if numThreadsLeft > 0 {
      numThreadGroups += 1
    }

    start := 0
    incr := numClassifiers / numThreadGroups
    end := incr

    numThreadsLeft = numThreads
    wg.Add(numThreadGroups)
    for i := 1; i < numThreadGroups; i++ {
      newArgs := threadArgs{id:i, xTrain:xTrain, yTrain:yTrain,
                            xVal:xVal, yVal:yVal,
                            numHiddenLayers:numHiddenLayers,
                            numClasses:numClasses,
                            numThreads:threadsPerClassifier,
                            numIterations:numIterations,
                            reduceEtaAt:reduceEtaAt,
                            reduceEtaBy:reduceEtaBy, eta:eta,
                            batchSize:batchSize,
                            hiddenLayerNodes:hiddenLayerNodes,
                            dropout:dropout,
                            classifiers:classifiers, start:start,
                            end:end, wg:&wg}
      go runTraining(&newArgs)
      start += incr
      end += incr
      numThreadsLeft -= threadsPerClassifier
    }
    /*At this point, 4 or fewer threads remain, use them on
      remaining classifiers*/
    newArgs := threadArgs{id:0, xTrain:xTrain, yTrain:yTrain,
                          xVal:xVal, yVal:yVal,
                          numHiddenLayers:numHiddenLayers,
                          numClasses:numClasses,
                          numThreads:numThreadsLeft,
                          numIterations:numIterations,
                          reduceEtaAt:reduceEtaAt,
                          reduceEtaBy:reduceEtaBy, eta:eta,
                          batchSize:batchSize,
                          hiddenLayerNodes:hiddenLayerNodes,
                          dropout:dropout,
                          classifiers:classifiers,
                          start:start, end:numClassifiers,
                          wg:&wg}
    runTraining(&newArgs)
    wg.Wait()
  }
  fmt.Println("All classifiers are trained")

  /*Get matrix of every probability of every class for every example*/
  probabilities := nn_utils.CreateMatrix(len(yTest), numClasses)

  for i := 0; i < numClassifiers; i++ {
    /*For each classifier, get the probabilities for each training example*/
    theseProbs := classifiers[i].Forward(xTest) /*Matrix*/

    /*For each prob, add it to the overall prob average*/
    for j := 0; j < len(theseProbs); j++ {
      for k := 0; k < numClasses; k++ {
        /*accumulates the average probability over all the classes*/
        probabilities[j][k] += theseProbs[j][k] / float64(numClassifiers)
      }
    }
  }

  /*Get the predictions from the probabilities (greatest probabilities)*/
  yHatTest := nn_utils.PredsFromProbs(probabilities, numThreads)

  overallAccuracy := nn_utils.Accuracy(yHatTest, yTest, numThreads)
  fmt.Println("Overall Accuracy", (overallAccuracy) * 100)
}

func runTraining(args *threadArgs) {
  fmt.Println("RunTraining id:", args.id, "start:", args.start,
              "end:", args.end, "threads:", args.numThreads)
  /*Create the NN's and train them*/
  for i := args.start; i < args.end; i++ {
    newClassifier := nn.NewNN(args.numHiddenLayers, args.numClasses,
                              args.xTrain, args.yTrain, args.xVal, args.yVal,
                              args.numThreads, args.hiddenLayerNodes,
                              args.dropout)
    fmt.Println("New classifier", i, "created by", args.id)
    newClassifier.Train(args.numIterations, args.reduceEtaAt,
                        args.eta, args.reduceEtaBy, args.batchSize)
    fmt.Println("New classifier trained by", args.id)
    args.classifiers[i] = newClassifier
  }
  fmt.Println("All classifiers are trained by group", args.id)
  args.wg.Done()
}
