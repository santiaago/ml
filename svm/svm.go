// Package svm provides a set of svm types and functions.
package svm

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/santiaago/ml"
)

// SVM holds all the information needed to run a support vector machine algorithm.
//
type SVM struct {
	TrainingPoints    int           // number of training points.
	TransformFunction TransformFunc // transformation function.
	HasTransform      bool          // determines if linear regression uses a transformation funtion, in which case 'TransformationFunction' should be defined.
	Xn                [][]float64   // data set of points for training.
	Yn                []float64     // output, evaluation of each Xi based on input data/ training examples.
	Wn                []float64     // weight vector.
	VectorSize        int           // size of vectors Xi and Wi.
	Lambda            float64       // used in the learning algorithm.
	Eta               float64       // used in the learning algorithm.
	K                 int           // used in the learning algorithm, it is the block size to use when running Pegasos
	T                 int           // number of iterations that Pegasos should run.
}

// NewSVM creates a support vector machine object.
//
func NewSVM() *SVM {
	return &SVM{Lambda: 0.01, K: 1, VectorSize: 3, T: 1000}
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
	svm.Wn = make([]float64, svm.VectorSize)

	for t := 0; t <= svm.T; t++ {

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

// Ein returns the in sample error of the current svm model.
// It is the fraction of in sample points which got misclassified.
//
func (svm *SVM) Ein() float64 {

	// XnWn
	gInSample := make([]float64, len(svm.Xn))
	for i := 0; i < len(svm.Xn); i++ {
		gi, err := svm.Predict(svm.Xn[i])
		if err != nil {
			continue
		}
		gInSample[i] = ml.Sign(gi)
	}

	nEin := 0
	for i := 0; i < len(gInSample); i++ {
		if gInSample[i] != svm.Yn[i] {
			nEin++
		}
	}
	ein := float64(nEin) / float64(len(gInSample))
	return ein
}

// Ecv returns the leave one out cross validation
// in sample error of the current svm model.
//
func (svm *SVM) Ecv() float64 {

	trainingPoints := svm.TrainingPoints
	x := svm.Xn
	y := svm.Yn
	nEcv := 0
	for out := range svm.Xn {
		fmt.Printf("\rLeave %v out of %v", out, len(svm.Xn))
		outx, outy := x[out], y[out]
		nsvm := NewSVM()
		*nsvm = *svm
		nsvm.TrainingPoints = svm.TrainingPoints - 1

		nsvm.Xn = [][]float64{}
		nsvm.Yn = []float64{}
		for i := range x {
			if i == out {
				continue
			}
			nsvm.Xn = append(nsvm.Xn, x[i])
			nsvm.Yn = append(nsvm.Yn, y[i])
		}

		if err := nsvm.Learn(); err != nil {
			log.Println("Learn error", err)
			trainingPoints--
			continue
		}

		gi, err := nsvm.Predict(outx)
		if err != nil {
			log.Println("Predict error", err)
			trainingPoints--
			continue
		}

		if ml.Sign(gi) != outy {
			nEcv++
		}

	}
	ecv := float64(nEcv) / float64(trainingPoints)
	return ecv
}

// Predict returns the result of the dot product between the x vector passed as param
// and the linear regression vector of weights.
//
func (svm *SVM) Predict(x []float64) (float64, error) {
	if len(x) != len(svm.Wn) {
		return 0, fmt.Errorf("SVM.Predict, size of x and Wn vector are different")
	}
	var p float64
	for j := 0; j < len(x); j++ {
		p += x[j] * svm.Wn[j]
	}
	return p, nil
}

// Predictions returns the prediction of each row of the 'data' passed in.
// It make a prediction by calling svm.Predict on each row of the data.
// If it fails to make a prediction it arbitrarly sets the result to 0
//
func (svm *SVM) Predictions(data [][]float64) ([]float64, error) {
	var err error
	var predictions []float64
	for i := 0; i < len(data); i++ {

		x := []float64{}
		// append x0
		x = append(x, 1)

		x = append(x, data[i]...)

		if svm.HasTransform {
			if x, err = svm.TransformFunction(x); err != nil {
				return nil, err
			}
		}

		gi, err := svm.Predict(x)
		if err != nil {
			predictions = append(predictions, 0)
			continue
		}

		if ml.Sign(gi) == float64(1) {
			predictions = append(predictions, 1)
		} else {
			predictions = append(predictions, 0)
		}
	}
	return predictions, nil
}

// ApplyTransformation sets Transform flag to true
// and transforms the Xn vector into Xtrans = TransformationFunction(Xn).
// It Sets Wn size to the size of Xtrans.
//
func (svm *SVM) ApplyTransformation() error {
	svm.HasTransform = true

	for i := 0; i < svm.TrainingPoints; i++ {
		if Xtrans, err := svm.TransformFunction(svm.Xn[i]); err == nil {
			svm.Xn[i] = Xtrans
		} else {
			return err
		}
	}
	svm.VectorSize = len(svm.Xn[0])
	svm.Wn = make([]float64, svm.VectorSize)
	return nil
}

// TransformFunc type is used to define transformation functions.
//
type TransformFunc func([]float64) ([]float64, error)
