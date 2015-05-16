// Package svm provides a set of svm types and functions.
package svm

// SVM holds all the information needed to run a support vector machine algorithm.
type SVM struct {
	TrainingPoints int         // number of training points.
	Xn             [][]float64 // data set of points for training.
	Yn             []float64   // output, evaluation of each Xi based on input data/ training examples.
	Wn             []float64   // weight vector.

}
