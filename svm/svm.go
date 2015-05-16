// Package svm provides a set of svm types and functions.
package svm

import (
	"math"
	"math/rand"

	"github.com/santiaago/ml"
)

// SVM holds all the information needed to run a support vector machine algorithm.
//
type SVM struct {
	TrainingPoints int         // number of training points.
	Xn             [][]float64 // data set of points for training.
	Yn             []float64   // output, evaluation of each Xi based on input data/ training examples.
	Wn             []float64   // weight vector.
	VectorSize     int         // size of vectors Xi and Wi.
	Lambda         float64     // used in the learning algorithm.
	Eta            float64     // used in the learning algorithm.
	K              int         // used in the learning algorithm, it is the block size to use when running Pegasos
}

// NewSVM creates a support vector machine object.
//
func NewSVM() *SVM {
	return &SVM{K: 1}
}

// InitializeFromData reads a 2 dimentional array with the following format:
// x1 x2 y
// x1 x2 y
// x1 x2 y
// And sets Xn and Yn accordingly
//
func (svm *SVM) InitializeFromData(data [][]float64) error {

	n := 0
	svm.Yn = make([]float64, len(data))
	svm.Xn = make([][]float64, len(data))

	for i, sample := range data {

		svm.Xn[i] = make([]float64, len(sample))
		svm.Xn[i] = []float64{1}
		svm.Xn[i] = append(svm.Xn[i], sample[:len(sample)-1]...)

		svm.Yn[i] = sample[len(sample)-1]
		n++
	}

	svm.TrainingPoints = n
	svm.VectorSize = len(svm.Xn[0])
	svm.Wn = make([]float64, svm.VectorSize)

	return nil
}

// Learn will update the weight vector with the output of a svm algorithm,
// with respect of the training examples and labels (svm.Xn, svm.Yn).
//
// We use an implementation of the Mini-Batch Pegasos Algorithm.
//
// The implementation of the Mini-Batch Pegasos Algorithm is based on the work done by:
// Shalev-Shwartz, Shai and Singer, Yoram and Srebro, Nathan.
// Pegasos: Primal Estimated sub-GrAdient SOlver for SVM.
//
// Note: We might support other methods in the future like SMO, ...
//
func (svm *SVM) Learn() error {

	var err error
	svm.TrainingPoints = len(svm.Xn)
	svm.Wn = make([]float64, svm.TrainingPoints)

	for t := range svm.Xn {

		// choose At where |At| = k, uniformly at random
		var At []int // vector of the selected indexes of size K
		for i := 0; i < svm.K; i++ {
			at := rand.Intn(svm.TrainingPoints)
			At = append(At, at)
		}

		// set At+ = {i in At : yi<wt, xi> < 1}
		var Atplus []int
		for _, i := range At {

			xi, yi := ml.Vector(svm.Xn[i]), svm.Yn[i]
			wt := ml.Vector(svm.Wn)
			var dot float64

			if dot, err = wt.Dot(xi); err != nil {
				return err
			}

			if yi*dot < 1 {
				Atplus = append(Atplus, i)
			}
		}

		// update eta = 1 / lambda*t
		svm.Eta = float64(1) / (svm.Lambda * float64(t+1))

		// set wt+1 = (1 - eta*lambda)*wt + eta/k * sum(for i in At+ of yixi)
		term1 := ml.Vector(svm.Wn).Scale(1 - svm.Eta*svm.Lambda)
		term2 := ml.Vector(make([]float64, svm.VectorSize))
		for _, i := range Atplus {
			xi, yi := ml.Vector(svm.Xn[i]), svm.Yn[i]
			xiyi := xi.Scale(yi)
			if term2, err = term2.Add(xiyi); err != nil {
				return err
			}
		}
		if svm.Wn, err = term1.Add(term2); err != nil {
			return err
		}

		// wt+1 = min{1, (1/sqrt(lambda))/||wt+1||} wt+1
		var norm float64
		if norm, err = ml.Vector(svm.Wn).Norm(); err != nil {
			return err
		}
		projection := float64(1) / (math.Sqrt(svm.Lambda) * norm)
		if 1 > projection {
			svm.Wn = ml.Vector(svm.Wn).Scale(projection)
		}
	}
	return nil
}
