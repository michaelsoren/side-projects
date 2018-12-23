/*loss: This file contains the parse struct and it's primary/helper function*/
package parse

import (
  "fmt"
  "strconv"
  "nn_utils"
  "sync"
)

/*
 * Takes in a csv of strings, which stores label and data, and outputs just
 * the data and labels split up. Parallelized.
 */
func ParseCsv(input [][]string, numClasses, numThreads int) ([][]float64, [][]float64) {

  labels := nn_utils.CreateMatrix(len(input), numClasses)
  /*-1 removes the class slot*/
  data := nn_utils.CreateMatrix(len(input), len(input[0]) - 1)
  if (input[0][0] == "label") {
    fmt.Println("NOTE: Removing first row")
    input = input[1:] /*removes the first row if it is the titles*/
  }

  var wg sync.WaitGroup
  wg.Add(numThreads)
  /*Divide up the data*/
  xStart := 0
  xIncr := len(input) / numThreads
  xEnd := xIncr
  for i := 1; i < numThreads; i++ {
    /*Calls goroutine to compute sub function*/
    go parseSubCsv(input, labels, data, xStart, xEnd, i, &wg)
    xStart += xIncr
    xEnd += xIncr
  }
  parseSubCsv(input, labels, data, xStart, len(input), 0, &wg)
  wg.Wait()
  return data, labels
}

/*Parallelizes the parsing of the CSV*/
func parseSubCsv(input [][]string, labels, data [][]float64, xStart, xEnd,
                id int, wg *sync.WaitGroup) {
  for rowIndex := xStart; rowIndex < xEnd; rowIndex++ {
    /*set the 1 hot vector*/
    label, _ := strconv.Atoi(input[rowIndex][0])
    labels[rowIndex][label] = 1
    for colId := 1; colId < len(input[rowIndex]); colId++ {
      intVal, _ := strconv.Atoi(input[rowIndex][colId])
      /*csv stores images 0-255, converts to decimal*/
      data[rowIndex][colId - 1] = float64(intVal) / 255.0
    }
  }
  wg.Done()
}
