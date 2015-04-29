package logreg

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/santiaago/ml/linear"
)

// LogisticRegression holds all the information needed to run the LogisticRegression algorithm.
type LogisticRegression struct {
	TrainingPoints       int             // number of training points.
	Dimension            int             // dimension.
	Interval             linear.Interval // interval in which the points, outputs and function are defined.
	RandomTargetFunction bool            // flag to know if target function is generated at random or defined by user.
	Eta                  float64         // learning rate.
	Epsilon              float64         // small value used to determined when is it converging.
	Equation             linear.Equation // random equation that defines the random linear function: targetFunction.
	TargetFunction       linear.Function // random linear function.
	Xn                   [][]float64     // data set of points for training (if defined at random, they are uniformly chosen in Interval).
	Yn                   []float64       // output, evaluation of each Xi based on the linear function.
	Wn                   []float64       // weight vector.
	VectorSize           int             // size of vectors Xi and Wi
	Epochs               int             // number of epochs
}

// NewLogisticRegression creates a logistic regression object.
// TrainingPoints = 100
// Interval [-1 : 1]
// Learning rate: 0.01
// Epsilon: 0.01
func NewLogisticRegression() *LogisticRegression {
	lr := LogisticRegression{
		TrainingPoints:       100,
		Interval:             linear.Interval{-1, 1},
		RandomTargetFunction: true,
		Eta:                  0.01,
		Epsilon:              0.01,
		VectorSize:           3,
	}
	return &lr
}

// Initialize will set up the LogisticRegression structure with the following:
// * the random linear function
// * vector Xn with X0 = 1 and X1 and X2 random points in the defined input space.
// * vector Yn the output of the random linear function on each point Xi. either -1 or +1 based on the linear function.
// * vector Wn is set zeros.
func (lr *LogisticRegression) Initialize() {

	// generate random target function if asked to
	if lr.RandomTargetFunction {
		lr.Equation = linear.RandEquation(lr.Interval)
		lr.TargetFunction = lr.Equation.Function()
	}

	lr.Xn = make([][]float64, lr.TrainingPoints)
	for i := 0; i < lr.TrainingPoints; i++ {
		lr.Xn[i] = make([]float64, lr.VectorSize)
	}

	lr.Yn = make([]float64, lr.TrainingPoints)
	lr.Wn = make([]float64, lr.VectorSize)

	x0 := float64(1)
	for i := 0; i < lr.TrainingPoints; i++ {
		lr.Xn[i][0] = x0
		for j := 1; j < len(lr.Xn[i]); j++ {
			lr.Xn[i][j] = lr.Interval.RandFloat()
		}

		lr.Yn[i] = evaluate(lr.TargetFunction, lr.Xn[i])
	}
}

// InitializeFromData reads a 2 dimentional array with the following format:
// x1 x2 y
// x1 x2 y
// x1 x2 y
// And sets Xn and Yn accordingly
func (lr *LogisticRegression) InitializeFromData(data [][]float64) error {

	n := 0
	lr.Yn = make([]float64, len(data))
	lr.Xn = make([][]float64, len(data))

	for i, sample := range data {

		lr.Xn[i] = make([]float64, len(sample))
		lr.Xn[i] = []float64{1}
		lr.Xn[i] = append(lr.Xn[i], sample[:len(sample)-1]...)

		lr.Yn[i] = sample[len(sample)-1]
		n++
	}

	lr.TrainingPoints = n
	lr.VectorSize = len(lr.Xn[0])
	lr.Wn = make([]float64, lr.VectorSize)

	return nil
}

// Learn will use a stockastic gradient descent (SGD) algorithm
// and update Wn vector acordingly.
func (lr *LogisticRegression) Learn() {

	lr.Epochs = 0
	indexes := buildIndexArray(lr.TrainingPoints)
	for {
		shuffleArray(&indexes)
		wOld := make([]float64, len(lr.Wn))
		copy(wOld, lr.Wn)
		for i := range indexes {
			wi := lr.Xn[i][1:]
			yi := lr.Yn[i]
			gt := lr.Gradient(wi, yi)
			lr.UpdateWeights(gt)
		}
		lr.Epochs++
		if lr.Converged(wOld) {
			break
		}
	}
}

// Gradient returns the gradient vector with respect to:
// the current sample wi
// the current target value:yi
// the current weights: Wn
func (lr *LogisticRegression) Gradient(wi []float64, yi float64) []float64 {
	v := make([]float64, len(wi)+1)
	v[0] = yi
	for i, x := range wi {
		v[i+1] = yi * x
	}
	var a []float64
	a = append(a, 1)
	a = append(a, wi...)

	b := make([]float64, len(lr.Wn))
	copy(b, lr.Wn)
	d := float64(1) + math.Exp(float64(yi)*dot(a, b))

	//vG = [-1.0 * x / d for x in vector]
	vg := make([]float64, len(v))
	for i := range v {
		vg[i] = float64(-1) * v[i] / d
	}
	return vg
}

// UpdateWeights updates the weights given the current weights 'Wn',
// the gradient vector 'gt' using of the learning rate 'Eta'.
func (lr *LogisticRegression) UpdateWeights(gt []float64) {

	if len(gt) != len(lr.Wn) {
		// todo(santiaago): should return error instead.
		fmt.Println("Panic: length of Wn and gt should be equal")
		panic(gt)
	}

	newW := make([]float64, len(lr.Wn))
	for i := range lr.Wn {
		newW[i] = (lr.Wn[i] - lr.Eta*gt[i])
	}
	lr.Wn = newW
}

// Converged returns a boolean answer telling whether the old weight vector
// and the new vector have converted based on the epsilon value.
func (lr *LogisticRegression) Converged(wOld []float64) bool {
	diff := make([]float64, len(wOld))
	for i := range wOld {
		diff[i] = lr.Wn[i] - wOld[i]
	}
	return norm(diff) < lr.Epsilon
}

// norm performs the norm operation of the vector 'v' passed as argument.
// todo(santiaago): move this to math.go or vector.go
func norm(v []float64) float64 {
	return math.Sqrt(dot(v, v))
}

// dot performs the dot product of vectors 'a' and 'b'.
// todo(santiaago): move this to math.go or vector.go
func dot(a, b []float64) float64 {
	if len(a) != len(b) {
		fmt.Println("Panic: lenght of a, and b should be equal")
		panic(a)
	}
	var ret float64
	for i := range a {
		ret += a[i] * b[i]
	}
	return ret
}

// buildIndexArray builds an array of incremental integers
// from 0 to n -1
func buildIndexArray(n int) []int {
	indexes := make([]int, n)
	for i := range indexes {
		indexes[i] = i
	}
	return indexes
}

// shuffleArray shuffles an array of integers.
func shuffleArray(a *[]int) {
	slice := *a
	for i := range slice {
		j := rand.Intn(i + 1)
		slice[i], slice[j] = slice[j], slice[i]
	}
}

// evaluate returns +1 or -1 based on the point passed as argument
// and the function 'f'. if it stands on one side of the function it is +1 else -1
func evaluate(f linear.Function, x []float64) float64 {
	last := len(x) - 1
	if x[last] < f(x[1:last]) {
		return -1
	}
	return 1
}

// Ein returns the in sample error of the current model.
// todo(santiaago): compute Ein
func (lr *LogisticRegression) Ein() float64 {
	return 1
}

// Eout is the out of sample error of the logistic regression.
// It uses the cross entropy error given a generated data set and the weight vector Wn
func (lr *LogisticRegression) Eout() float64 {
	outOfSample := 1000
	cee := float64(0)

	x0 := float64(1)
	for i := 0; i < outOfSample; i++ {
		var oY float64
		oX := make([]float64, lr.VectorSize)
		oX[0] = x0
		for j := 1; j < len(oX); j++ {
			oX[j] = lr.Interval.RandFloat()
		}
		oY = evaluate(lr.TargetFunction, oX)
		cee += lr.CrossEntropyError(oX, oY)
	}
	return cee / float64(outOfSample)
}

// CrossEntropyError computes the cross entropy error given a sample X and its target,
// with respect to weight vector Wn based on formula:
// log(1 + exp(-y*sample*w))
func (lr *LogisticRegression) CrossEntropyError(sample []float64, Y float64) float64 {
	return math.Log(float64(1) + math.Exp(-Y*dot(sample, lr.Wn)))
}
