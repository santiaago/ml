// Package svm provides a set of svm types and functions.
package svm

// SVM holds all the information needed to run a support vector machine algorithm.
//
type SVM struct {
	TrainingPoints int         // number of training points.
	Xn             [][]float64 // data set of points for training.
	Yn             []float64   // output, evaluation of each Xi based on input data/ training examples.
	Wn             []float64   // weight vector.
	VectorSize     int         // size of vectors Xi and Wi.
}

// NewSVM creates a support vector machine object.
//
func NewSVM() *SVM {
	return &SVM{}
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
