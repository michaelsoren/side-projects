/*nn: This file contains the neural network struct and its associated methods.*/
package nn

import (
  "fmt"
  "layer"
  "math/rand"
  "loss"
  "nn_utils"
)

/*Neural network structure. Stores the layers, the data, and side info*/
type NN struct {
  layers []layer.Layer
  x [][]float64
  y [][]float64
  xVal [][]float64
  yVal [][]float64

  dropout float64
  /*How many threads can this thread create, including self*/
  numThreads int
}

/*
 * This creates an empty NN, with all the right layer sizes
 * and also sets default layer information
 */
func NewNN(numLayers, numClasses int, x, y, xVal, yVal [][]float64,
           numThreads, hiddenLayerNodes int, dropout float64) NN {
  numLayers += 2 /*Input and output layers*/

  layers := make([]layer.Layer, numLayers)

  dataSize := len(x[0])

  dropout += rand.Float64() / 100.0

  /*Input layer*/
  layers[0] = layer.NewLayer(dataSize, hiddenLayerNodes,
                             dropout, true, false)
  /*Hidden layers*/
  for thisLayer := 1; thisLayer < numLayers - 1; thisLayer++ {
    layers[thisLayer] = layer.NewLayer(hiddenLayerNodes, hiddenLayerNodes,
                                             dropout, true, true)
  }
  /*Output layer*/
  layers[numLayers - 1] = layer.NewLayer(hiddenLayerNodes, numClasses,
                                         dropout, true, false)
  /*Other default information*/
	return NN{layers:layers, x:x, y:y, xVal:xVal, yVal:yVal, dropout:dropout,
            numThreads:numThreads}
}

/*
 * Train the network with the given training hyperparemters
 */
func (nn *NN) Train(maxIterations, dropAfter int,
                    eta, etaDrop float64, batchSize int) {
  nn.setTraining(true)

  lossLayer := loss.NewLoss()
  /*Max iteration stoppingn criteria*/
  for iteration := 1; iteration < maxIterations + 1; iteration++ {

    if iteration % dropAfter == 0 {
      /*Reduces Eta when you reach the drop amount*/
      eta *= etaDrop
      /*Print current accuracy*/
      fmt.Println("Accuracy at", iteration, ":",
                  nn.accuracy(nn.xVal, nn.yVal) * 100)
    }
    /*Randomly sample the data for the batch*/
    sampleX, sampleY := nn_utils.DrawSample(nn.x, nn.y, batchSize)

    /*Forward pass, get the score/output*/
    score := nn.Forward(sampleX)

    /*Calculate the loss with a given loss function*/
    lossLayer.ForwardLoss(score, sampleY)

    /*Calculate gradient of loss*/
    dScore := lossLayer.BackwardLoss(sampleY)

    /*backprop the gradient*/
    nn.backward(dScore)

    /*Do gradient descent*/
    nn.sgd(eta)
  }
  /*Turn model off of training mode*/
  nn.setTraining(false)
}

/*
 * Forward calculates the output given the current weights for a given x.
 * Must be trained first, otherwise prediction is meaningless/random
 */
func (nn *NN) Forward(x [][]float64) [][]float64 {
  input := x
  for i := 0; i < len(nn.layers); i++ {
    input = nn.layers[i].Forward(input, nn.numThreads)
  }
  output := input
  return output
}

/*
 * Accuracy calculates the accuracy of the nn with current weights.
 * If neural network is not trained, accuracy will be meaningless.
 */
func (nn *NN) accuracy(x , y[][]float64) float64 {
  /*Runs the neural network forward to get the probabilities for this X*/
  probs := nn.Forward(x)
  /*Predicts the label with the highest probability*/
  yHat := nn_utils.PredsFromProbs(probs, nn.numThreads)
  return nn_utils.Accuracy(yHat, y, nn.numThreads)
}

/*
 * Goes through neural network layers and sets the training boolean
 */
func (nn *NN) setTraining(x bool) {
  for i := 0; i < len(nn.layers); i++ {
    nn.layers[i].SetTraining(x)
  }
}

/*
 * Backward passes the loss back through each of the layers
 * so they can compute the gradient.
 */
func (nn *NN) backward(loss [][]float64) [][]float64 {
  output := loss
  for i := len(nn.layers) - 1; i >= 0; i-- {
    output = nn.layers[i].Backward(output, nn.numThreads)
  }
  input := output
  return input
}

/*
 * SGD loops through every layer and adds the gradients
 * for the weights and biases to the weights and biases.
 */
func (nn *NN) sgd(eta float64) {
  for i := 0; i < len(nn.layers); i++ {
    thisLayer := nn.layers[i]
    for k := 0; k < len(thisLayer.Weights[0]); k++ {
      /*Step in the opposite direction of the gradient time seta for bias*/
      thisLayer.Biases[k] -= eta * thisLayer.BiasesGrad[k]
      for j := 0; j < len(thisLayer.Weights); j++ {
        /*move in the direction of the gradient times eta for weights*/
        thisLayer.Weights[j][k] -= eta * thisLayer.WeightsGrad[j][k]
      }
    }
  }
}
