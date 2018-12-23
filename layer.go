/*layer: This file contains the layer struct and its
         associated methods.*/
package layer

import (
  "math/rand"
  "math"
  "nn_utils"
  "sync"
)

/*An entry is used in my pipelining to pass along a
  computed matrix entry*/
type entry struct {
  val float64
  row int
  col int
}

/*This is what a layer can do*/
type layer interface {
  Init(inputSize, outputSize int)
  Forward(input [][]float64) [][]float64
  Backward(outputGrad [][]float64) [][]float64
  setTraining(x bool)
}

/*Stores all the relevant info for a layer*/
type Layer struct {
  /*Information about layer. isHidden and training enables dropout*/
  training bool
  isHiddenLayer bool

  /*Weights and biases are for fully connected layer*/
  Weights [][]float64
  Biases []float64
  WeightsGrad [][]float64
  BiasesGrad []float64

  /*Gradients*/
  gradInput [][]float64
  gradReluInput [][]float64
  gradDropoutInput [][]float64

  /*fully connected (FC) input and output*/
  input [][]float64
  output [][]float64

  /*relu input and output*/
  reluInput [][]float64
  reluOutput [][]float64

  /*dropout input and output*/
  dropoutInput [][]float64
  dropoutOutput [][]float64

  /*Only used for dropout layer*/
  dropoutRate float64
  dropoutMask [][]float64
}

/*
 * This creates a new layer from the given info.
 */
func NewLayer(inputSize, outputSize int, dropoutRate float64,
              training, isHiddenLayer bool) Layer {
  newLayer := Layer{training:training,
                    isHiddenLayer:isHiddenLayer}

  newLayer.initFullyConnected(inputSize, outputSize)
  newLayer.dropoutRate = dropoutRate

  return newLayer
}

/*
 * Init initializes the weights and biases with random values
 * and creates the storage for other fields.
 */
func (l *Layer) initFullyConnected(inputSize, outputSize int) {
  newWeights := nn_utils.CreateMatrix(inputSize, outputSize)
  newBiases := make([]float64, outputSize)

  /*Initialize the weights and biases*/
  for j := 0; j < outputSize; j++ {
    newBiases[j] = rand.Float64() / 20 - 0.025
    for i := 0; i < inputSize; i++ {
      /*Between -0.1 and 0.1, uniform*/
      newWeights[i][j] = rand.Float64() / 5 - 0.1
    }
  }

  newWeightsGrad := make([][]float64, inputSize)
  for i := range newWeightsGrad {
      newWeightsGrad[i] = make([]float64, outputSize)
  }
  newBiasesGrad := make([]float64, outputSize)

  l.Weights = newWeights
  l.Biases = newBiases
  l.WeightsGrad = newWeightsGrad
  l.BiasesGrad = newBiasesGrad
}

/*
 * Forward takes the given input and calculates the outputs
 * with the given weights and biases. Uses relu
 */
func (l *Layer) Forward(input [][]float64,
                        numThreads int) [][]float64 {
  if numThreads >= 4 {
    return l.ForwardParallel(input, numThreads)
  } else {
    /*Compute fully conected layer*/
    l.input = input
    l.output =
        nn_utils.AddVToMatrix(nn_utils.DotM(l.input,
                                            l.Weights, 0,
                                            numThreads),
                              l.Biases)
    /*Store fully connected layer*/
    l.reluInput = l.output
    l.reluOutput = nn_utils.Max(l.reluInput, 0, numThreads)
    /*Computes dropout if appropriate, otherwise returns*/
    if (l.training && l.isHiddenLayer) {
      l.dropoutInput = l.reluOutput
      l.dropoutMask =
          nn_utils.GenerateDropoutMask(l.dropoutRate, len(input),
                                       len(input[0]), numThreads)

      l.dropoutOutput =
          nn_utils.HadamardProduct(l.reluOutput, l.dropoutMask,
                                   numThreads)
      return l.dropoutOutput
    } else {
      return l.reluOutput
    }
  }
}

/*
 * Backwards computes the relevant gradients in reverse order
 * (dropout, relu, then fully connected)
 */
func (l *Layer) Backward(outputGrad [][]float64, numThreads int) [][]float64 {
  if numThreads >= 4 {
    return l.BackwardParallel(outputGrad, numThreads)
  } else {
    if l.isHiddenLayer {
      /*applies dropout layer if relevant*/
      outputGrad =
          nn_utils.HadamardProduct(l.dropoutMask, outputGrad, numThreads)
    }
    mask := nn_utils.GenerateReluMask(l.reluOutput, numThreads)
    l.gradReluInput = nn_utils.HadamardProduct(mask, outputGrad, numThreads)

    /*Create new Grads*/
    weightsGrad :=
        nn_utils.CreateMatrix(len(l.WeightsGrad), len(l.WeightsGrad[0]))
    biasesGrad := make([]float64, len(l.BiasesGrad))
    /*Created empty biases/weights gradient matricies*/

    /*zeroes the current gradients*/
    l.WeightsGrad = weightsGrad
    l.BiasesGrad = biasesGrad
    /*Updates the current gradient values*/
    l.WeightsGrad = nn_utils.DotM(l.input, l.gradReluInput, 1, numThreads)
    l.BiasesGrad = nn_utils.SumAlongAxis(l.gradReluInput, numThreads)
    l.gradInput = nn_utils.DotM(l.gradReluInput, l.Weights, 2, numThreads)

    return l.gradInput
  }
}

/*
 * Basic function that sets the training flag
 */
func (l *Layer) SetTraining(x bool) {
  l.training = x
}

/*
 * Forward parallel is the pipeline based parallel
 * We are guaranteed here that numThreads is
 * greater than or equal to 4
 */
func (l *Layer) ForwardParallel(input [][]float64, numThreads int) [][]float64 {
  /*Creates the pipe for communication between FC and ReLU/dropout*/
  pipe := make(chan entry)
  /*Sets l.input for use in fc layer*/
  l.input = input
  /*Sends half the threads to work on computing fully connected layer*/
  /*FC splits data in half and computes. For each computation, loads result
    into channel*/
  go l.fcForwardParallel(pipe, numThreads / 2 + numThreads % 2)
  /*Relu applies relu to output of pipe and also applies dropout if relevant*/
  return l.reluForwardParallel(pipe, numThreads / 2)
}

/*
 * Pushes out the fully connected layer operations into the pipe.
 * Parallelized
 */
func (l *Layer) fcForwardParallel(pipe chan entry, numThreads int) {
  newOutput := nn_utils.CreateMatrix(len(l.input), len(l.Weights[0]))
  /*Thread function*/
  fcParallelThread := func(l *Layer, newOutput [][]float64,
                           pipe chan entry, start, end int,
                           wg *sync.WaitGroup) {
    /*Computes dot, adds bias, pipelines it forward*/
    for row := 0; row < len(l.input); row++ {
      for col := 0; col < len(l.Weights[0]); col++ {
        dot := 0.0
        for k := 0; k < len(l.input[0]); k++ {
          dot += l.input[row][k] * l.Weights[k][col]
        }
        /*Store the values*/
        newOutput[row][col] = dot + l.Biases[col]
        /*Forward the value*/
        pipe <-entry{val:newOutput[row][col], row:row, col:col}
      }
    }
    wg.Done()
  }
  var wg sync.WaitGroup
  start := 0
  incr := len(l.input) / numThreads
  end := incr
  wg.Add(numThreads)
  for id := 1; id < numThreads; id++ {
    go fcParallelThread(l, newOutput, pipe, start, end, &wg)
    start += incr
    end += incr
  }
  fcParallelThread(l, newOutput, pipe, start, len(l.input), &wg)
  /*Waits for threads to finish working*/
  wg.Wait()
  /*Close the pipe now that forward layer is computed*/
  close(pipe)
  /*load the output into the layer for later use in backward*/
  l.output = newOutput
  l.reluInput = newOutput
}

/*
 * Computes the relu and dropout parts, fc values pipelined in
 */
func (l *Layer) reluForwardParallel(pipe chan entry,
                                    numThreads int) [][]float64 {
  reluForwardParallelThread := func(pipe chan entry, reluRet,
                                 dropoutMask, dropoutRet [][]float64,
                                 wg *sync.WaitGroup) {
    for pipeOut := range pipe {
      /*Compute this relu value*/
      reluRet[pipeOut.row][pipeOut.col] = math.Max(pipeOut.val, 0.0)
      /*If we're doing a dropout layer, computes dropout mask and layer here */
      if l.training && l.isHiddenLayer {
        threshold := rand.Float64()
        if threshold < l.dropoutRate {
          dropoutMask[pipeOut.row][pipeOut.col] = 0
        } else {
          dropoutMask[pipeOut.row][pipeOut.col] = 1
        }
        dropoutRet[pipeOut.row][pipeOut.col] =
            dropoutMask[pipeOut.row][pipeOut.col] *
            reluRet[pipeOut.row][pipeOut.col]
      }
    }
    wg.Done()
  }
  var wg sync.WaitGroup
  wg.Add(numThreads)
  /*Creates matrixes*/
  reluRet := nn_utils.CreateMatrix(len(l.input), len(l.Weights[0]))
  dropoutMask := nn_utils.CreateMatrix(len(l.input), len(l.Weights[0]))
  dropoutRet := nn_utils.CreateMatrix(len(l.input), len(l.Weights[0]))

  /*Spin off n-1 threads to receive from pipe*/
  for id := 1; id < numThreads; id++ {
    go reluForwardParallelThread(pipe, reluRet, dropoutMask, dropoutRet, &wg)
  }
  /*Send main thread to get from channel*/
  reluForwardParallelThread(pipe, reluRet, dropoutMask, dropoutRet, &wg)

  /*All finished processing, load up the values and return the correct thing*/
  l.reluOutput = reluRet
  if l.training && l.isHiddenLayer {
    l.dropoutInput = l.reluOutput
    l.dropoutMask = dropoutMask
    l.dropoutOutput = dropoutRet
    return l.dropoutOutput
  }
  wg.Wait()
  return l.reluOutput
}

/*Note: Pipelining is not used for back prop because back prop requires full
        rows of data instead of just values, making it much more efficient
        to handle by using data/functional decomp to compute relu/dropout,
        then data decomp for the fully connected gradients*/

/*
 * Guaranteed to have 4 or more threads
 */
func (l *Layer) BackwardParallel(outputGrad [][]float64,
                                 numThreads int) [][]float64 {
  l.reluBackwardParallel(outputGrad, numThreads / 2)
  return l.fcBackwardParallel(numThreads / 2 + numThreads % 2)
}

/*
 * Computes the relu and dropout parts with functional/data decomp and n threads
 */
func (l *Layer) reluBackwardParallel(outputGrad [][]float64,
                                     numThreads int) {
  reluBackwardParallelThread := func(outputGrad, newGradReluInput [][]float64,
                                     start, end int, wg *sync.WaitGroup) {
    for row := start; row < end; row++ {
      for col := 0; col < len(outputGrad[0]); col++ {
        thisGrad := outputGrad[row][col]
        if l.isHiddenLayer {
          thisGrad *= l.dropoutMask[row][col]
        }
        if l.reluOutput[row][col] < 0 {
          thisGrad = 0
        }
        /*Loads the ret matrix*/
        newGradReluInput[row][col] = thisGrad
      }
    }
    wg.Done()
  }
  var wg sync.WaitGroup
  wg.Add(numThreads)
  newGradReluInput := nn_utils.CreateMatrix(len(outputGrad), len(outputGrad[0]))
  start := 0
  incr := len(outputGrad) / numThreads
  end := incr

  for id := 1; id < numThreads; id++ {
    go reluBackwardParallelThread(outputGrad, newGradReluInput,
                                  start, end, &wg)
    start += incr
    end += incr
  }
  reluBackwardParallelThread(outputGrad, newGradReluInput,
                             start, len(outputGrad), &wg)
  wg.Wait()
  l.gradReluInput = newGradReluInput //may not need, try without
}

/*
 * Computes the weights, biases, and input gradients with the
 * given number of threads
 */
func (l *Layer) fcBackwardParallel(numThreads int) [][]float64 {
  weightsGrad := nn_utils.CreateMatrix(len(l.WeightsGrad),
                                       len(l.WeightsGrad[0]))
  biasesGrad := make([]float64, len(l.BiasesGrad))

  l.WeightsGrad = weightsGrad
  l.BiasesGrad = biasesGrad

  l.WeightsGrad = nn_utils.DotM(l.input, l.gradReluInput, 1, numThreads)
  l.BiasesGrad = nn_utils.SumAlongAxis(l.gradReluInput, numThreads)
  l.gradInput = nn_utils.DotM(l.gradReluInput, l.Weights, 2, numThreads)

  return l.gradInput
}
