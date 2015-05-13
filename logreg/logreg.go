// Package logreg provide a set of logistic regression types and functions.
package logreg

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/santiaago/ml"
	"github.com/santiaago/ml/linear"
)

// LogisticRegression holds all the information needed to run the LogisticRegression algorithm.
//
type LogisticRegression struct {
	TrainingPoints       int             // number of training points.
	Dimension            int             // dimension.
	Interval             linear.Interval // interval in which the points, outputs and function are defined.
	RandomTargetFunction bool            // flag to know if target function is generated at random or defined by user.
	Eta                  float64         // learning rate.
	Epsilon              float64         // small value used to determined when is it converging.
	Equation             linear.Equation // random equation that defines the random linear function: targetFunction.
	TargetFunction       linear.Function // random linear function.
	TransformFunction    TransformFunc   // transformation function.
	HasTransform         bool            // determines if logistic regression uses a transformation funtion, in which case 'TransformationFunction' should be defined.
	IsRegularized        bool            // flag to determine if model is regularized or not.
	Xn                   [][]float64     // data set of points for training (if defined at random, they are uniformly chosen in Interval).
	Yn                   []float64       // output, evaluation of each Xi based on the linear function.
	Wn                   []float64       // weight vector.
	WReg                 []float64       // weight vector with regularization.
	K                    int             // used for setting the value of lambda.
	Lambda               float64         // used for regularization lambda = 10^-k
	VectorSize           int             // size of vectors Xi and Wi.
	Epochs               int             // number of epochs.
	MaxEpochs            int             // upper bound for the logistic regression model with it is not able to converge.
	ComputedEin          bool            // flag that tells if Ein has already been computed.
	ein                  float64         // last computed in sample error.
	ComputedEcv          bool            // flag that tells if Ecv has already been computed.
	ecv                  float64         // last computed cross validation error.
}

// NewLogisticRegression creates a logistic regression object.
// TrainingPoints = 100
// Interval [-1 : 1]
// Learning rate: 0.01
// Epsilon: 0.01
//
func NewLogisticRegression() *LogisticRegression {
	lr := LogisticRegression{
		TrainingPoints:       100,
		Interval:             linear.Interval{-1, 1},
		RandomTargetFunction: true,
		Eta:                  0.01,
		Epsilon:              0.01,
		VectorSize:           3,
		MaxEpochs:            1000,
		K:                    -3,
	}
	return &lr
}

// Initialize will set up the LogisticRegression structure with the following:
// * the random linear function
// * vector Xn with X0 = 1 and X1 and X2 random points in the defined input space.
// * vector Yn the output of the random linear function on each point Xi. either -1 or +1 based on the linear function.
// * vector Wn is set zeros.
//
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
//
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

// InitializeFromFile reads a file with the following format:
// x1 x2 y
// x1 x2 y
// x1 x2 y
// And sets Xn and Yn accordingly
// todo(santiaago): make function accept any number of points and 'y'.
//
func (lr *LogisticRegression) InitializeFromFile(filename string) error {

	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	n := 0
	for scanner.Scan() {
		split := strings.Split(scanner.Text(), " ")
		var line []string
		for _, s := range split {
			cell := strings.Replace(s, " ", "", -1)
			if len(cell) > 0 {
				line = append(line, cell)
			}
		}

		var x1, x2, y float64

		if x1, err = strconv.ParseFloat(line[0], 64); err != nil {
			return err
		}

		if x2, err = strconv.ParseFloat(line[1], 64); err != nil {
			return err
		}

		if y, err = strconv.ParseFloat(line[2], 64); err != nil {
			return err
		}

		newX := []float64{1, x1, x2}
		lr.Xn = append(lr.Xn, newX)
		lr.Yn = append(lr.Yn, y)

		n++
	}

	lr.TrainingPoints = n
	lr.VectorSize = len(lr.Xn[0])
	lr.Wn = make([]float64, lr.VectorSize)

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
		return err
	}
	return nil
}

// Learn will use a stockastic gradient descent (SGD) algorithm
// and update Wn vector acordingly.
//
func (lr *LogisticRegression) Learn() error {
	lr.Epochs = 0
	indexes := buildIndexArray(lr.TrainingPoints)
	for {
		shuffleArray(&indexes)
		wOld := make([]float64, len(lr.Wn))
		copy(wOld, lr.Wn)
		for i := range indexes {
			wi := lr.Xn[i][1:]
			yi := lr.Yn[i]
			gt, err := lr.Gradient(wi, yi)
			if err != nil {
				log.Printf("failed when calling Gradient with error: %v, exiting learning algorithm.\n", err)
				return err
			}
			if err := lr.UpdateWeights(gt); err != nil {
				log.Printf("failed when calling UpdateWeights with error: %v, exiting learning algorithm.\n", err)
				return err
			}
		}
		lr.Epochs++
		if lr.Converged(wOld) {
			break
		}
		if lr.Epochs > lr.MaxEpochs {
			break
		}
	}
	return nil
}

// Gradient returns the gradient vector with respect to:
// the current sample wi
// the current target value:yi
// the current weights: Wn
//
func (lr *LogisticRegression) Gradient(wi []float64, yi float64) ([]float64, error) {
	v := make([]float64, len(wi)+1)
	v[0] = yi
	for i, x := range wi {
		v[i+1] = yi * x
	}
	a := make([]float64, len(wi)+1)
	a[0] = 1
	for i := range wi {
		a[i+1] = wi[i]
	}
	b := make([]float64, len(lr.Wn))
	copy(b, lr.Wn)
	dot, err := ml.Vector(a).Dot(b)
	if err != nil {
		return nil, err
	}
	d := float64(1) + math.Exp(float64(yi)*dot)

	//vG = [-1.0 * x / d for x in vector]
	vg := make([]float64, len(v))
	for i := range v {
		vg[i] = float64(-1) * v[i] / d
	}
	return vg, nil
}

// UpdateWeights updates the weights given the current weights 'Wn',
// the gradient vector 'gt' using of the learning rate 'Eta'.
//
func (lr *LogisticRegression) UpdateWeights(gt []float64) error {

	if len(gt) != len(lr.Wn) {
		return fmt.Errorf("length of Wn and gt should be equal")
	}

	newW := make([]float64, len(lr.Wn))
	for i := range lr.Wn {
		newW[i] = (lr.Wn[i] - lr.Eta*gt[i])
	}
	lr.Wn = newW
	return nil
}

// LearnRegularized will use a stockastic gradient descent (SGD) algorithm
// with regularization, and update WReg vector accordingly.
//
func (lr *LogisticRegression) LearnRegularized() error {
	lr.WReg = make([]float64, lr.VectorSize)
	lr.Lambda = math.Pow(10, float64(lr.K))
	lr.Epochs = 0
	indexes := buildIndexArray(lr.TrainingPoints)

	for {
		shuffleArray(&indexes)
		wOld := make([]float64, len(lr.WReg))
		copy(wOld, lr.WReg)

		for i := range indexes {
			wi := lr.Xn[i][1:]
			yi := lr.Yn[i]
			gt, err := lr.GradientRegularized(wi, yi)
			if err != nil {
				log.Printf("failed when calling GradientRegularized with error: %v, exiting learning algorithm.\n", err)
				return err
			}
			if err := lr.UpdateRegularizedWeights(gt); err != nil {
				log.Printf("failed when calling UpdateRegularizedWeights with error: %v, exiting learning algorithm.\n", err)
				return err
			}
		}
		lr.Epochs++
		if lr.ConvergedRegularized(wOld) {
			break
		}
		if lr.Epochs > lr.MaxEpochs {
			break
		}
	}
	return nil
}

// GradientRegularized returns the regularized gradient vector with respect to:
// the current sample wi
// the current target value:yi
// the current weights: WReg
//
func (lr *LogisticRegression) GradientRegularized(wi []float64, yi float64) ([]float64, error) {
	v := make([]float64, len(wi)+1)
	v[0] = yi
	for i, x := range wi {
		v[i+1] = yi * x
	}
	a := make([]float64, len(wi)+1)
	a[0] = 1
	for i := range wi {
		a[i+1] = wi[i]
	}
	b := make([]float64, len(lr.WReg))
	copy(b, lr.WReg)
	dot, err := ml.Vector(a).Dot(b)
	if err != nil {
		return nil, err
	}
	d := float64(1) + math.Exp(float64(yi)*dot)

	//vG = [-1.0 * x / d for x in vector] + lambda/N*Vector(wi)^2
	wi2, err := ml.Vector(wi).Dot(wi)
	if err != nil {
		log.Println("skiping regularizer step due to %v", err)
		wi2 = 1
	}
	reg := (lr.Lambda / float64(len(lr.WReg))) * wi2
	vg := make([]float64, len(v))
	for i := range v {
		vg[i] = (float64(-1) * v[i] / d) + reg
	}
	return vg, nil
}

// UpdateRegularizedWeights updates the weights given the current weights 'WReg',
// the gradient vector 'gt' using of the learning rate 'Eta'.
//
func (lr *LogisticRegression) UpdateRegularizedWeights(gt []float64) error {

	if len(gt) != len(lr.WReg) {
		return fmt.Errorf("length of WReg and gt should be equal")
	}

	newW := make([]float64, len(lr.WReg))
	for i := range lr.WReg {
		newW[i] = (lr.WReg[i] - lr.Eta*gt[i])
	}
	lr.WReg = newW
	return nil
}

// TransformFunc type is used to define transformation functions.
//
type TransformFunc func([]float64) ([]float64, error)

// ApplyTransformation sets Transform flag to true
// and transforms the Xn vector into Xtrans = TransformationFunction(Xn).
// It Sets Wn size to the size of Xtrans.
//
func (lr *LogisticRegression) ApplyTransformation() error {
	lr.HasTransform = true

	for i := 0; i < lr.TrainingPoints; i++ {
		if Xtrans, err := lr.TransformFunction(lr.Xn[i]); err == nil {
			lr.Xn[i] = Xtrans
		} else {
			return err
		}

	}
	lr.VectorSize = len(lr.Xn[0])
	lr.Wn = make([]float64, lr.VectorSize)
	return nil
}

// Predict returns the result of the dot product between the x vector passed as param
// and the logistic regression vector of weights.
//
func (lr *LogisticRegression) Predict(x []float64) (float64, error) {
	if len(x) != len(lr.Wn) {
		return 0, fmt.Errorf("logreg.Predict, size of x and Wn vector are different")
	}
	return ml.Vector(x).Dot(lr.Wn)
}

// Predictions returns the prediction of each row of the 'data' passed in.
// It make a prediction by calling lr.Predict on each row of the data.
// If it fails to make a prediction it arbitrarly sets the result to 0
//
func (lr *LogisticRegression) Predictions(data [][]float64) ([]float64, error) {

	var err error
	var predictions []float64
	for i := 0; i < len(data); i++ {

		x := []float64{}
		// append x0
		x = append(x, 1)

		x = append(x, data[i]...)

		if lr.HasTransform {
			if x, err = lr.TransformFunction(x); err != nil {
				return nil, err
			}
		}

		gi, err := lr.Predict(x)
		if err != nil {
			return nil, err
		}

		if ml.Sign(gi) == float64(1) {
			predictions = append(predictions, 1)
		} else {
			predictions = append(predictions, 0)
		}
	}
	return predictions, nil
}

// Converged returns a boolean answer telling whether the old weight vector
// and the new vector have converted based on the epsilon value.
//
func (lr *LogisticRegression) Converged(wOld []float64) bool {
	diff := make([]float64, len(wOld))
	for i := range wOld {
		diff[i] = lr.Wn[i] - wOld[i]
	}
	norm, err := ml.Vector(diff).Norm()
	if err != nil {
		log.Println("forcing convergence as we fail to compute norm.")
		return true
	}
	return norm < lr.Epsilon
}

// ConvergedRegularized returns a boolean answer telling whether the old weight vector
// and the new vector have converted based on the epsilon value.
//
func (lr *LogisticRegression) ConvergedRegularized(wOld []float64) bool {
	diff := make([]float64, len(wOld))
	for i := range wOld {
		diff[i] = lr.WReg[i] - wOld[i]
	}
	norm, err := ml.Vector(diff).Norm()
	if err != nil {
		log.Println("forcing convergence as we fail to compute norm.")
		return true
	}
	return norm < lr.Epsilon
}

// evaluate returns +1 or -1 based on the point passed as argument
// and the function 'f'. if it stands on one side of the function it is +1 else -1
//
func evaluate(f linear.Function, x []float64) float64 {
	last := len(x) - 1
	if x[last] < f(x[1:last]) {
		return -1
	}
	return 1
}

// Ein returns the in sample error of the current model.
//
func (lr *LogisticRegression) Ein() float64 {
	if lr.ComputedEin {
		return lr.ein
	}

	// XnWn
	gInSample := make([]float64, len(lr.Xn))
	for i := 0; i < len(lr.Xn); i++ {
		gi, err := lr.Predict(lr.Xn[i])
		if err != nil {
			continue
		}
		gInSample[i] = ml.Sign(gi)
	}

	nEin := 0
	for i := 0; i < len(gInSample); i++ {
		if gInSample[i] != lr.Yn[i] {
			nEin++
		}
	}
	ein := float64(nEin) / float64(len(gInSample))
	lr.ComputedEin = true
	lr.ein = ein

	return ein

}

// Ecv returns the leave one out cross validation
// in sample error of the current logistic regression model.
//
func (lr *LogisticRegression) Ecv() float64 {
	if lr.ComputedEcv {
		return lr.ecv
	}

	x := lr.Xn
	y := lr.Yn

	nEcv := 0
	for out := range lr.Xn {
		outx, outy := lr.Xn[out], lr.Yn[out]
		nlr := NewLogisticRegression()
		nlr.TrainingPoints = lr.TrainingPoints - 1
		nlr.Wn = make([]float64, lr.VectorSize)
		nlr.VectorSize = lr.VectorSize

		nlr.Xn = [][]float64{}
		nlr.Yn = []float64{}
		for i := range x {
			if i == out {
				continue
			}
			nlr.Xn = append(nlr.Xn, x[i])
			nlr.Yn = append(nlr.Yn, y[i])
		}

		if nlr.IsRegularized {
			if err := nlr.LearnRegularized(); err != nil {
				log.Println("LearnRegularized error", err)
				nEcv++
				continue
			}
			nlr.Wn = nlr.WReg
		} else {
			if err := nlr.Learn(); err != nil {
				log.Println("Learn error", err)
				nEcv++
				continue
			}
		}

		gi, err := nlr.Predict(outx)
		if err != nil {
			nEcv++
			continue
		}

		if ml.Sign(gi) != outy {
			nEcv++
		}

	}
	ecv := float64(nEcv) / float64(lr.TrainingPoints)
	lr.ComputedEcv = true
	lr.ecv = ecv
	return ecv
}

// Eout is the out of sample error of the logistic regression.
// It uses the cross entropy error given a generated data set
// and the weight vector Wn
//
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
		ccee, err := lr.CrossEntropyError(oX, oY)
		if err != nil {
			log.Println("Failed to compute CrossEntropyError, incrementing error and skiping.")
			cee++
			continue
		}
		cee += ccee
	}
	return cee / float64(outOfSample)
}

// EAugIn is the fraction of "in sample points" which got misclassified plus the term
// lambda / N * Sum(Wi^2)
// todo(santiaago): change this to use vector vector.
//
func (lr *LogisticRegression) EAugIn() float64 {

	gInSample := make([]float64, len(lr.Xn))
	for i := 0; i < len(lr.Xn); i++ {
		gi := float64(0)
		for j := 0; j < len(lr.Xn[0]); j++ {
			gi += lr.Xn[i][j] * lr.WReg[j]
		}
		gInSample[i] = ml.Sign(gi)
	}
	nEin := 0
	for i := 0; i < len(gInSample); i++ {
		if gInSample[i] != lr.Yn[i] {
			nEin++
		}
	}

	wi2, err := ml.Vector(lr.WReg).Dot(lr.WReg)
	if err != nil {
		log.Println("skiping regularizer step due to %v", err)
		wi2 = 1
	}
	reg := (lr.Lambda / float64(len(lr.WReg))) * wi2

	return float64(nEin)/float64(len(gInSample)) + reg
}

// CrossEntropyError computes the cross entropy error
// given a sample X and its target, with respect to weight
// vector Wn based on formula:
// log(1 + exp(-y*sample*w))
//
func (lr *LogisticRegression) CrossEntropyError(sample []float64, Y float64) (float64, error) {
	dot, err := ml.Vector(sample).Dot(lr.Wn)
	if err != nil {
		return 0, err
	}
	return math.Log(float64(1) + math.Exp(-Y*dot)), nil
}

// buildIndexArray builds an array of incremental integers
// from 0 to n -1
//
func buildIndexArray(n int) []int {
	indexes := make([]int, n)
	for i := range indexes {
		indexes[i] = i
	}
	return indexes
}

// shuffleArray shuffles an array of integers.
//
func shuffleArray(a *[]int) {
	slice := *a
	for i := range slice {
		j := rand.Intn(i + 1)
		slice[i], slice[j] = slice[j], slice[i]
	}
}
