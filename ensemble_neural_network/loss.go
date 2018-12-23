/*loss: This file contains the loss struct and associated function*/
package loss

import (
  "math"
  "nn_utils"
)

/*Loss struct is simple, stores input and logProbs for later use*/
type Loss struct {
  logProbs [][]float64
  input [][]float64
}

/*
 * This creates a new loss, implemented here for symmetry
 * with other classes
 */
func NewLoss() Loss {
  return Loss{}
}

/*
 * Calculates the forward direction loss based on the inputs and labels.
 * This has the most involved math, and most of it is needed to prevent
 * underflow.
 */
func (l *Loss) ForwardLoss(input [][]float64, labels [][]float64) float64 {
  output := make([]float64, len(labels))
  l.input = input
  /*Modify input to prevent underflow (round-off error issues)*/
  for i := 0; i < len(labels); i++ {
    rowMax := findMax(input, i)

    for j := 0; j < len(labels[i]); j++ {
      input[i][j] -= rowMax
    }
  }

  /*Init new log probs*/
  l.logProbs = nn_utils.CreateMatrix(len(labels), len(labels[0]))

  /*Calculate log probs*/
  for i := 0; i < len(labels); i++ {
    for j := 0; j < len(labels[i]); j++ {
      l.logProbs[i][j] = input[i][j] - math.Log(expSumArray(input[i]))
    }
  }

  tempCol := make([]float64, len(labels[0]))
  /*Calculate output*/
  for i := 0; i < len(labels); i++ {
    for j := 0; j < len(labels[0]); j++ {
      tempCol[j] = labels[i][j] * -1 * l.logProbs[i][j]
    }
    output[i] = sumArray(tempCol)
  }
  return mean(output)
}

/*
 * Computes the loss on the given probabilities, using the labels
 */
func (l *Loss) BackwardLoss(labels [][]float64) [][]float64 {
  ret := nn_utils.CreateMatrix(len(labels), len(labels[0]))

  for i := 0; i < len(labels); i++ {
    for j := 0; j < len(labels[j]); j++ {
      ret[i][j] = (math.Exp(l.logProbs[i][j]) - labels[i][j]) / float64(len(l.input))
    }
  }
  return ret
}

/*Helper functions used only for loss below*/

/*
 * Small helper function used exclusively for the loss.
 * Computes the sum of an array
 */
func sumArray(m []float64) float64 {
  total := 0.0
  for i := 0; i < len(m); i++ {
    total += m[i]
  }
  return total
}

/*
 * Computes the sum, but with every term as exp(term)
 */
func expSumArray(m []float64) float64 {
  total := 0.0
  for i := 0; i < len(m); i++ {
    total += math.Exp(m[i])
  }
  return total
}

/*
 * Compute the average value of an array
 */
func mean(m []float64) float64 {
  return sumArray(m) / float64(len(m))
}

/*
 * Computes the largest value in a given row.
 */
func findMax(m [][]float64, whichRow int) float64 {
  rowMax := m[whichRow][0]

  for i := 1; i < len(m[whichRow]); i++ {
    if rowMax < m[whichRow][i] {
      rowMax = m[whichRow][i]
    }
  }
  return rowMax
}
