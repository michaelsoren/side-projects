/*nn_utils: This file contains matrix math, relu applications, and
            similar methods.*/
package nn_utils

import (
  "fmt"
  "math/rand"
  "sync"
)

/*
 * Compute the predictions by predicting the class with largest probability.
 * Parallelized.
 */
func PredsFromProbs(probs [][]float64, numThreads int) [][]float64 {
  predsFromProbsThread := func(probs, ret [][]float64, start, end int,
                               wg *sync.WaitGroup) {
    for i := start; i < end; i++ {
      largestSeen := probs[i][0]
      indexOf := 0
      for j := 1; j < len(probs[i]); j++ {
        if probs[i][j] > largestSeen {
          largestSeen = probs[i][j]
          indexOf = j
        }
      }
      if indexOf >= 0 && indexOf < len(probs[0]) {
        ret[i][indexOf] = 1.0
      }
    }
    wg.Done()
  }

  var wg sync.WaitGroup
  wg.Add(numThreads)
  /*defines storage and start/end*/
  ret := CreateMatrix(len(probs), len(probs[0]))
  start := 0
  incr := len(probs) / numThreads
  end := incr
  /*spin off correct number of sub threads to compute max*/
  for id := 1; id < numThreads; id++ {
    go predsFromProbsThread(probs, ret, start, end, &wg)
    start += incr
    end += incr
  }
  /*Do the rest with the main thread*/
  predsFromProbsThread(probs, ret, start, len(probs), &wg)
  wg.Wait()

  return ret
}

/*
 * If neural network is not trained, accuracy will be meaningless.
 * Computes accuracy versus labels by direct comparison.
 * Parallelized
 */
func Accuracy(yHat, y[][]float64, numThreads int) float64 {
  threadSums := make(chan float64, numThreads)

  var wg sync.WaitGroup
  wg.Add(numThreads)

  /*Function only used inside Accuracy, used for threading*/
  accuracyThread := func(yHat, y [][]float64, start, end int, c chan float64,
                         wg *sync.WaitGroup) {

      numCorrect := 0.0
      for i := start; i < end; i++ {
        colsEqual := true
        for j := 0; j < len(yHat[i]); j++ {
          if yHat[i][j] != y[i][j] {
            colsEqual = false
          }
        }
        if colsEqual {
          numCorrect += 1.0
        }
      }

      c<-numCorrect
      wg.Done()
  }
  numCorrect := 0.0
  start := 0
  incr := len(yHat) / numThreads
  end := incr
  /*Spin off the child threads*/
  for id := 1; id < numThreads; id++ {
    //fmt.Println("id", id)
    go accuracyThread(yHat, y, start, end, threadSums, &wg)
    start += incr
    end += incr
  }

  accuracyThread(yHat, y, start, len(yHat), threadSums, &wg)

  wg.Wait()
  close(threadSums)
  for subCorrect := range threadSums {

    numCorrect += subCorrect
  }
  return numCorrect / float64(len(yHat))
}

/*
 * Add a 1d vector column to a 2d matrix.
 */
func AddVToMatrix(x [][]float64, y []float64) [][]float64 {
  ret := CreateMatrix(len(x), len(x[0]))

  for i := 0; i < len(x); i++ {
    for j := 0; j < len(x[i]); j++ {
      ret[i][j] = x[i][j] + y[j]
    }
  }

  return ret
}

/*
 * Sum two 1d vectors.
 */
func AddVectors(x, y []float64) []float64 {
  ret := make([]float64, len(x))

  for i := 0; i < len(x); i++ {
    ret[i] = x[i] + y[i]
  }
  return ret
}

/*
 * Compute the dot product between two 2d matrixes. Whichtranspose marks which
 * matrix to transpose. 0 is none, 1 is the first, 2 is the second.
 * Calls relevant function from there.
 * Parallelized
 */
func DotM(x, y [][]float64, whichTranspose, numThreads int) [][]float64 {
  dotMThread := func(x, y, ret [][]float64, whichTranspose, start, end int,
                     wg *sync.WaitGroup) {
    for i := start; i < end; i++ {
      for j := 0; j < len(ret[i]); j++ {
        if whichTranspose == 0 {
          ret[i][j] = DotVec(x, y, i, j)
        } else if whichTranspose == 1 {
          ret[i][j] = DotVecTX(x, y, i, j)
        } else if whichTranspose == 2 {
          ret[i][j] = DotVecTY(y, x, j, i)
        }
      }
    }
    wg.Done()
  }
  var wg sync.WaitGroup
  wg.Add(numThreads)
  /*Create ret and start/end*/
  ret := CreateMatrix(len(x), len(y))
  if whichTranspose == 0 {
    ret = CreateMatrix(len(x), len(y[0]))
  } else if whichTranspose == 1 {
    ret = CreateMatrix(len(x[0]), len(y[0]))
  }
  start := 0
  incr := len(ret) / numThreads
  end := incr
  /*spin off correct number of sub threads to compute max*/
  for id := 1; id < numThreads; id++ {
    go dotMThread(x, y, ret, whichTranspose, start, end, &wg)
    start += incr
    end += incr
  }
  /*Do the rest with the main thread*/
  dotMThread(x, y, ret, whichTranspose, start, len(ret), &wg)
  wg.Wait()
  return ret
}

/*
 * Dot the ith row of x and jth column of y together
 */
func DotVec(x, y [][]float64, i, j int) float64 {
  sum := 0.0
  if len(x[i]) != len(y) {
    fmt.Println("Dimensions do not align, error")
  }

  for k := 0; k < len(x[i]); k++ {
    sum += x[i][k] * y[k][j] /*ith row dotted with jth column*/
  }
  return sum
}

/*
 * Dot the ith row of x transpose and the jth column of y
 */
func DotVecTX(x, y [][]float64, i, j int) float64 {
  sum := 0.0

  for k := 0; k < len(x); k++ {
    sum += x[k][i] * y[k][j] /*ith row of transpose dotted with jth column*/
  }

  return sum
}

/*
 * Dot the ith row of x and the jth column of y transpose
 */
func DotVecTY(x, y [][]float64, i, j int) float64 {
  sum := 0.0

  for k := 0; k < len(x[i]); k++ {
    sum += x[i][k] * y[j][k] /*ith row dotted with jth column of transpose*/
  }

  return sum
}

/*
 * Also known as elementwise product of a matrix
 * Parallelized
 */
func HadamardProduct(x, y [][]float64, numThreads int) [][]float64 {  /*Defines thread computation function*/
  hadamardProductThread := func(x, y, ret [][]float64, start, end int,
                                wg *sync.WaitGroup) {
    for i := start; i < end; i++ {
      for j := 0; j < len(x[0]); j++ {
        ret[i][j] = x[i][j] * y[i][j]
      }
    }
    wg.Done()
  }
  var wg sync.WaitGroup
  wg.Add(numThreads)
  /*Creates ret storage and start/end*/
  ret := CreateMatrix(len(x), len(x[0]))
  start := 0
  incr := len(x) / numThreads
  end := incr
  /*spin off numThreads - 1 threads to do sub problem*/
  for id := 1; id < numThreads; id++ {
    go hadamardProductThread(x, y, ret, start, end, &wg)
    start += incr
    end += incr
  }
  /*Main thread finishes the rest of them*/
  hadamardProductThread(x, y, ret, start, len(x), &wg)
  wg.Wait()

  return ret
}

/*
 * Computes the max of a value and each value of a matrix.
 * Parallelized
 */
func Max(x [][]float64, val float64, numThreads int) [][]float64 {
  /*Defines thread computation function*/
  maxThread := func(x, ret [][]float64, val float64, start, end int,
                    wg *sync.WaitGroup) {
    for i := start; i < end; i++ {
      for j := 0; j < len(x[i]); j++ {
        if x[i][j] > val {
          ret[i][j] = x[i][j]
        } else {
          ret[i][j] = val
        }
      }
    }
    wg.Done()
  }
  var wg sync.WaitGroup
  wg.Add(numThreads)
  /*create storage and start/end*/
  ret := CreateMatrix(len(x), len(x[0]))
  start := 0
  incr := len(x) / numThreads
  end := incr
  /*spin off correct number of sub threads to compute max*/
  for id := 1; id < numThreads; id++ {
    go maxThread(x, ret, val, start, end, &wg)
    start += incr
    end += incr
  }
  /*Do the rest with the main thread*/
  maxThread(x, ret, val, start, len(x), &wg)

  wg.Wait()
  return ret
}

/*
 * Sums along a 2d array (by column) and returns the sums.
 * Parallelized.
 */
func SumAlongAxis(input [][]float64, numThreads int) []float64 {
  /*Defines thread computation function*/
  sumAlongAxisThread := func(input [][]float64, ret []float64, start, end int,
                             wg *sync.WaitGroup) {
    for i := start; i < end; i++ {
      sum := 0.0
      for j := 0; j < len(input[i]); j++ {
        sum += input[i][j]
      }
      ret[i] = sum
    }
    wg.Done()
  }
  var wg sync.WaitGroup
  wg.Add(numThreads)
  /*defines storage and start/end*/
  ret := make([]float64, len(input))
  start := 0
  incr := len(input) / numThreads
  end := incr
  /*spin off correct number of sub threads to compute max*/
  for id := 1; id < numThreads; id++ {
    go sumAlongAxisThread(input, ret, start, end, &wg)
    start += incr
    end += incr
  }
  /*Do the rest with the main thread*/
  sumAlongAxisThread(input, ret, start, len(input), &wg)

  wg.Wait()
  return ret
}

/*
 * Generates a mask (1's where input positive, 0's where negative)
 * Parallelized
 */
func GenerateReluMask(input [][]float64, numThreads int) [][]float64 {
  generateReluMaskThread := func(input, ret [][]float64, start, end int,
                                 wg *sync.WaitGroup) {
    for i := start; i < end; i++ {
      for j := 0; j < len(input[0]); j++ {
        if input[i][j] < 0 {
          ret[i][j] = 0
        } else {
          ret[i][j] = 1
        }
      }
    }
    wg.Done()
  }
  var wg sync.WaitGroup
  wg.Add(numThreads)
  /*defines storage and start/end*/
  ret := CreateMatrix(len(input), len(input[0]))
  start := 0
  incr := len(input) / numThreads
  end := incr
  /*spin off correct number of sub threads to compute max*/
  for id := 1; id < numThreads; id++ {
    go generateReluMaskThread(input, ret, start, end, &wg)
    start += incr
    end += incr
  }
  /*Do the rest with the main thread*/
  generateReluMaskThread(input, ret, start, len(input), &wg)

  wg.Wait()
  return ret
}

/*
 * Generate the dropout mask based on a threshold.
 * Parallelized
 */
func GenerateDropoutMask(dropout float64, rowLen, colLen, numThreads int) [][]float64 {
  GenerateDropoutMaskThread := func(ret [][]float64, dropout float64, start,
                                    end int, wg *sync.WaitGroup) {
    for i := start; i < end; i++ {
      for j := 0; j < len(ret[i]); j++ {
        threshold := rand.Float64()
        if threshold < dropout {
          ret[i][j] = 0
        } else {
          ret[i][j] = 1
        }
      }
    }
    wg.Done()
  }
  var wg sync.WaitGroup
  wg.Add(numThreads)
  /*defines storage and start/end */
  ret := CreateMatrix(rowLen, colLen)
  start := 0
  incr := rowLen / numThreads
  end := incr
  /*spin off correct number of sub threads to compute max */
  for id := 1; id < numThreads; id++ {
    go GenerateDropoutMaskThread(ret, dropout, start, end, &wg)
    start += incr
    end += incr
  }
  /*Do the rest with the main thread*/
  GenerateDropoutMaskThread(ret, dropout, start, rowLen, &wg)
  wg.Wait()

  return ret
}

/*
 * Randomly samples samples x's and y's of the sample size
 */
func DrawSample(x, y [][]float64, sampleSize int) ([][]float64, [][]float64) {
  retX := CreateMatrix(sampleSize, len(x[0]))
  retY := CreateMatrix(sampleSize, len(y[0]))

  for i := 0; i < sampleSize; i++ {
    randIndex := rand.Intn(len(x))
    retX[i] = x[randIndex]
    retY[i] = y[randIndex]
  }
  return retX, retY
}

/*
 * Convenience function because I create so many 2d matrixes. This function just
 * creates a 2d array of the desires size.
 */
func CreateMatrix(rows, cols int) [][]float64 {
  ret := make([][]float64, rows)
  for i := range ret {
      ret[i] = make([]float64, cols)
  }
  return ret
}
