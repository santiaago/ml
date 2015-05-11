// Package linreg provide a set of linear regression types and functions.
package linreg

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/santiaago/ml"
	"github.com/santiaago/ml/linear"
)

// LinearRegression holds all the information needed to run the LinearRegression algorithm.
//
type LinearRegression struct {
	TrainingPoints       int             // number of training points.
	ValidationPoints     int             // number of validation point.
	RandomTargetFunction bool            // flag to know if target function is generated at random or defined by user.
	Noise                float64         // Noise should be btwn 0(no noise) and 1(all noise). Will simulate noise by flipping the sign of the output based on the Noise.
	Interval             linear.Interval // Interval in which the points, outputs and function are defined.
	Equation             linear.Equation // random equation that defines the random linear function: targetFunction.
	TargetFunction       linear.Function // linear target function to predict.
	TransformFunction    TransformFunc   // transformation function.
	HasTransform         bool            // determines if linear regression uses a transformation funtion, in which case 'TransformationFunction' should be defined.
	Xn                   [][]float64     // data set of points for training (if defined at random, they are uniformly present in Interval).
	XVal                 [][]float64     // data set of point  for validation.
	VectorSize           int             // size of vectors Xi and Wi.
	Yn                   []float64       // output, evaluation of each Xi based on linear function.
	YVal                 []float64       // output, for validation
	Wn                   []float64       // weight vector initialized at zeros.
	WReg                 []float64       // weight vector with regularization.
	Lambda               float64         // used in weight decay.
	K                    int             // used in weight decay.
	computedEin          bool            // flag that tells if Ein has already been computed.
	ein                  float64         // last computed in sample error.
	computedEcv          bool            // flag that tells if Ecv has already been computed.
	ecv                  float64         // last computed cross validation error.

}

// NewLinearRegression creates a linear regression object.
// TrainingPoints = 10
// Interval [-1 : 1]
// RandomTargetFunction is true
// Noise = 0
// VectorSize = 3
//
func NewLinearRegression() *LinearRegression {
	lr := LinearRegression{
		TrainingPoints:       10,                        // default training points is 10
		Interval:             linear.NewInterval(-1, 1), // default interval is [-1, 1]
		RandomTargetFunction: true,                      // default RandomTargetFunction is true
		Noise:                0,                         // default noise is 0
		VectorSize:           3,                         // default vector size is 3
	}
	return &lr
}

// Initialize will set up the LinearRegression structure with the following:
// * the random linear function
// * vector Xn with X0 = 1 and X1 and X2 random point in the defined input space.
// * vector Yn the output of the random linear function on each point Xi. either -1 or +1  based on the linear function.
// * vector Wn is set to zero.
//
func (lr *LinearRegression) Initialize() {

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

		// output with potential noise in 'flip' variable
		lr.Yn[i] = evaluate(lr.TargetFunction, lr.Xn[i]) * lr.flip()
	}
}

// flip returns 1 or -1 with respect to the amount of Noise present in the linear regression.
//
func (lr *LinearRegression) flip() float64 {
	flip := float64(1)
	if lr.Noise == 0 {
		return flip
	}

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	rN := r.Intn(100)
	if rN < int(math.Ceil(lr.Noise*100)) {
		flip = float64(-1)
	}
	return flip
}

// InitializeFromFile reads a file with the following format:
// x1 x2 y
// x1 x2 y
// x1 x2 y
// And sets Xn and Yn accordingly
// todo(santiaago): make function accept any number of points and 'y'.
//
func (lr *LinearRegression) InitializeFromFile(filename string) error {

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

// InitializeFromData reads a 2 dimentional array with the following format:
// x1 x2 y
// x1 x2 y
// x1 x2 y
// And sets Xn and Yn accordingly
//
func (lr *LinearRegression) InitializeFromData(data [][]float64) error {

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

// InitializeValidationFromData reads a 2 dimentional array with the following format:
// x1 x2 y
// x1 x2 y
// x1 x2 y
// And sets XVal and YVal accordingly
//
func (lr *LinearRegression) InitializeValidationFromData(data [][]float64) error {

	lr.YVal = make([]float64, len(data))
	lr.XVal = make([][]float64, len(data))
	n := 0
	for i, sample := range data {

		lr.XVal[i] = make([]float64, len(sample))
		lr.XVal[i] = []float64{1, sample[0], sample[1]}

		lr.YVal[i] = sample[2]
		n++

	}
	lr.ValidationPoints = n

	return nil
}

// ApplyTransformation sets Transform flag to true
// and transforms the Xn vector into Xtrans = TransformationFunction(Xn).
// It Sets Wn size to the size of Xtrans.
//
func (lr *LinearRegression) ApplyTransformation() {
	lr.HasTransform = true

	for i := 0; i < lr.TrainingPoints; i++ {
		Xtrans := lr.TransformFunction(lr.Xn[i])
		lr.Xn[i] = Xtrans
	}
	lr.VectorSize = len(lr.Xn[0])
	lr.Wn = make([]float64, lr.VectorSize)
}

// ApplyTransformationOnValidation transforms the XVal vector into
// XValtrans = TransformationFunction(XVal)
//
func (lr *LinearRegression) ApplyTransformationOnValidation() {
	for i := 0; i < lr.ValidationPoints; i++ {
		Xtrans := lr.TransformFunction(lr.XVal[i])
		lr.XVal[i] = Xtrans
	}
}

// Learn will compute the pseudo inverse X dager and set W vector accordingly
// XDagger = (X'X)^-1 X'
//
func (lr *LinearRegression) Learn() error {

	var err error

	// compute X' <=> X transpose
	var mXn ml.Matrix = lr.Xn
	var mXT ml.Matrix

	if mXT, err = mXn.Transpose(); err != nil {
		return err
	}

	// compute the product of X' and X
	var mXP ml.Matrix
	if mXP, err = mXT.Product(mXn); err != nil {
		return err
	}

	// inverse XProduct
	var mXInv ml.Matrix
	if mXInv, err = mXP.Inverse(); err != nil {
		return err
	}

	// compute product: (X'X)^-1 X'
	var XDagger ml.Matrix
	if XDagger, err = mXInv.Product(mXT); err != nil {
		return err
	}

	lr.setWeight(XDagger)

	return nil
}

// setWeight updates the weights Wn of the given linear regression.
// The weighs are updated by computing Wn = dagger * Yn
// todo(santiaago): change this to Wn = d[i]*Yn
//
func (lr *LinearRegression) setWeight(d ml.Matrix) {

	lr.Wn = make([]float64, lr.VectorSize)
	for i := 0; i < len(d); i++ {
		for j := 0; j < len(d[0]); j++ {
			lr.Wn[i] += d[i][j] * lr.Yn[j]
		}
	}
}

// setWeightReg updates the weights WReg of the given linear regression.
// The weights are updated by computing WReg = dagger * Yn
// todo(santiaago): change this to WReg = d[i]*Yn
//
func (lr *LinearRegression) setWeightReg(d ml.Matrix) {

	lr.WReg = make([]float64, lr.VectorSize)

	for i := 0; i < len(d); i++ {
		for j := 0; j < len(d[0]); j++ {
			lr.WReg[i] += d[i][j] * lr.Yn[j]
		}
	}
}

// Ein returns the in sample error of the current linear regression model.
// It is the fraction of in sample points which got misclassified.
// todo(santiaago): change this to gi = d[i]*Yn
//
func (lr *LinearRegression) Ein() float64 {
	if lr.computedEin {
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
	lr.computedEin = true
	lr.ein = ein

	return ein
}

// Ecv returns the leave one out cross validation
// in sample error of the current linear regression model.
//
func (lr *LinearRegression) Ecv() float64 {
	if lr.computedEcv {
		return lr.ecv
	}
	x := lr.Xn
	y := lr.Yn

	nEcv := 0
	for out := range lr.Xn {
		outx, outy := lr.Xn[out], lr.Yn[out]
		nlr := NewLinearRegression()
		nlr.TrainingPoints = lr.TrainingPoints - 1
		nlr.VectorSize = lr.VectorSize

		nlr.Xn = append(x[:out], x[out+1:]...)
		nlr.Yn = append(y[:out], y[out+1:]...)
		if err := nlr.Learn(); err != nil {
			nEcv++
			continue
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
	lr.computedEcv = true
	lr.ecv = ecv
	return ecv
}

// EAugIn is the fraction of "in sample points" which got misclassified plus the term
// lambda / N * Sum(Wi^2)
// todo(santiaago): change this to use vector vector.
// todo(santiaago): add term lambda / N * Sum(Wi^2)
//
func (lr *LinearRegression) EAugIn() float64 {

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

	return float64(nEin) / float64(len(gInSample))
}

// EValIn returns the in sample error of the Validation points.
// It is the fraction of misclassified points present in the Validation set XVal.
//
func (lr *LinearRegression) EValIn() float64 {

	gInSample := make([]float64, len(lr.XVal))
	for i := 0; i < len(lr.XVal); i++ {
		gi, err := lr.Predict(lr.XVal[i])
		if err != nil {
			continue
		}
		gInSample[i] = ml.Sign(gi)
	}
	nEin := 0
	for i := 0; i < len(gInSample); i++ {
		if gInSample[i] != lr.YVal[i] {
			nEin++
		}
	}

	return float64(nEin) / float64(len(gInSample))
}

// Eout returns the out of sample error.
// It is the fraction of out of sample points which got misclassified.
// It generates 1000 out of sample points and classifies them.
//
func (lr *LinearRegression) Eout() float64 {
	outOfSample := 1000
	numError := 0

	for i := 0; i < outOfSample; i++ {
		oX := make([]float64, lr.VectorSize)
		oX[0] = 1
		for j := 1; j < len(oX); j++ {
			oX[j] = lr.Interval.RandFloat()
		}

		// output with potential noise in 'flip' variable
		var oY float64
		oY = evaluate(lr.TargetFunction, oX) * lr.flip()

		var gi float64
		gi, err := lr.Predict(oX)
		if err != nil {
			numError++
			continue
		}

		if ml.Sign(gi) != oY {
			numError++
		}
	}
	return float64(numError) / float64(outOfSample)
}

// EoutFromFile returns error in the out of sample data provided in the file.
//  It only supports linear regressions with transformed data.
// todo:(santiaago) make this more generic.
//
func (lr *LinearRegression) EoutFromFile(filename string) (float64, error) {

	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	numError := 0
	n := 0
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		split := strings.Split(scanner.Text(), " ")
		var line []string
		for _, s := range split {
			cell := strings.Replace(s, " ", "", -1)
			if len(cell) > 0 {
				line = append(line, cell)
			}
		}

		var oX1, oX2, oY float64

		if oX1, err = strconv.ParseFloat(line[0], 64); err != nil {
			return 0, err
		}

		if oX2, err = strconv.ParseFloat(line[1], 64); err != nil {
			return 0, err
		}

		if oY, err = strconv.ParseFloat(line[2], 64); err != nil {
			return 0, err
		}

		oX := lr.TransformFunction([]float64{1, oX1, oX2})

		gi, err := lr.Predict(oX)
		if err != nil {
			numError++
			n++
			continue
		}
		if ml.Sign(gi) != oY {
			numError++
		}
		n++
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	return float64(numError) / float64(n), nil
}

// EAugOutFromFile returns the augmented error from an out of sample file
//
func (lr *LinearRegression) EAugOutFromFile(filename string) (float64, error) {

	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	numError := 0
	n := 0
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		split := strings.Split(scanner.Text(), " ")
		var line []string
		for _, s := range split {
			cell := strings.Replace(s, " ", "", -1)
			if len(cell) > 0 {
				line = append(line, cell)
			}
		}
		var oY float64
		var oX1, oX2 float64

		if oX1, err = strconv.ParseFloat(line[0], 64); err != nil {
			return 0, err
		}

		if oX2, err = strconv.ParseFloat(line[1], 64); err != nil {
			return 0, err
		}

		oX := lr.TransformFunction([]float64{1, oX1, oX2})

		if oY, err = strconv.ParseFloat(line[2], 64); err != nil {
			return 0, err
		}

		gi := float64(0)
		for j := 0; j < len(oX); j++ {
			gi += oX[j] * lr.WReg[j]
		}
		if ml.Sign(gi) != oY {
			numError++
		}
		n++
	}

	return float64(numError) / float64(n), nil
}

// LearnWeightDecay computes the following formula and update WReg.
// (Z'Z+λI)^−1 * Z'
// WReg = (Z'Z + λI)^−1 Z'y
//
func (lr *LinearRegression) LearnWeightDecay() error {
	lr.Lambda = math.Pow(10, float64(lr.K))

	var err error

	// compute X' <=> X transpose
	var mX ml.Matrix = lr.Xn
	var mXT ml.Matrix

	if mXT, err = mX.Transpose(); err != nil {
		return err
	}

	// compute lambda*Identity
	var ID ml.Matrix
	if ID, err = ml.Identity(len(lr.Xn[0])); err != nil {
		return err
	}

	var mLID ml.Matrix

	if mLID, err = ID.Scalar(lr.Lambda); err != nil {
		return err
	}

	// compute Z'Z
	var mXP ml.Matrix
	if mXP, err = mXT.Product(mX); err != nil {
		return err
	}

	// compute Z'Z + lambda*I
	var mS ml.Matrix
	if mS, err = mLID.Add(mXP); err != nil {
		return err
	}

	// inverse
	var mInv ml.Matrix

	if mInv, err = mS.Inverse(); err != nil {
		return err
	}

	// compute product: inverseMatrix Z'
	var XDagger ml.Matrix
	if XDagger, err = mInv.Product(mXT); err != nil {
		return err
	}

	// set WReg
	lr.setWeightReg(XDagger)
	return nil
}

// CompareInSample returns the number of points that are different between
// the current hypothesis function learned by the linear regression with respect to 'f'
//
func (lr *LinearRegression) CompareInSample(f linear.Function) float64 {

	gInSample := make([]float64, len(lr.Xn))
	fInSample := make([]float64, len(lr.Xn))

	for i := 0; i < len(lr.Xn); i++ {
		gi, err := lr.Predict(lr.Xn[i])
		if err != nil {
			// force difference because of error
			gInSample[i] = 0
			fInSample[i] = f(lr.Xn[i][1:])
			continue
		}

		gInSample[i] = ml.Sign(gi)
		fInSample[i] = f(lr.Xn[i][1:])
	}

	// measure difference:

	diff := 0
	for i := 0; i < len(lr.Xn); i++ {
		if gInSample[i] != fInSample[i] {
			diff++
		}
	}
	return float64(diff) / float64(len(lr.Xn))
}

// CompareOutOfSample returns the number of points that are different between the
// current hypothesis function learned by the linear regression with respect to
// 'f', the linear function passed as paral. The comparison is made on out of sample points
// generated randomly in the defined interval.
//
func (lr *LinearRegression) CompareOutOfSample(f linear.Function) float64 {

	outOfSample := 1000
	diff := 0

	for i := 0; i < outOfSample; i++ {
		//var oY int
		oX := make([]float64, lr.VectorSize)
		oX[0] = float64(1)
		for j := 1; j < len(oX); j++ {
			oX[j] = lr.Interval.RandFloat()
		}

		gi, err := lr.Predict(oX)
		if err != nil {
			diff++
			continue
		}
		if ml.Sign(gi) != f(oX[1:]) {
			diff++
		}
	}

	return float64(diff) / float64(outOfSample)
}

// TransformFunc type is used to define transformation functions.
//
type TransformFunc func([]float64) []float64

// TransformDataSet modifies Xn with the transformed function 'f' and updates the
// size of vector Wn.
//
func (lr *LinearRegression) TransformDataSet(f TransformFunc, newSize int) {
	for i := 0; i < len(lr.Xn); i++ {
		oldXn := lr.Xn[i]
		newXn := f(oldXn)
		lr.Xn[i] = make([]float64, newSize)
		for j := 0; j < len(newXn); j++ {
			lr.Xn[i][j] = newXn[j]
		}
	}
	lr.Wn = make([]float64, newSize)
}

// Predict returns the result of the dot product between the x vector passed as param
// and the linear regression vector of weights.
//
func (lr *LinearRegression) Predict(x []float64) (float64, error) {
	if len(x) != len(lr.Wn) {
		return 0, fmt.Errorf("Linreg.Predict, size of x and Wn vector are different")
	}
	var p float64
	for j := 0; j < len(x); j++ {
		p += x[j] * lr.Wn[j]
	}
	return p, nil
}

// Predictions returns the prediction of each row of the 'data' passed in.
// It make a prediction by calling lr.Predict on each row of the data.
// If it fails to make a prediction it arbitrarly sets the result to 0
//
func (lr *LinearRegression) Predictions(data [][]float64) ([]float64, error) {

	var predictions []float64
	for i := 0; i < len(data); i++ {

		x := []float64{}
		// append x0
		x = append(x, 1)

		x = append(x, data[i]...)

		if lr.HasTransform {
			x = lr.TransformFunction(x)
		}

		gi, err := lr.Predict(x)
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

// evaluate will map function f in point p with respect to the current y point.
// if it stands on one side it is +1 else -1
// vector x is defined as x0, x1 .. , xn, y
// So linear.Function should take sub vector [x1, ... ,xn]
// todo: might change name to mapPoint
//
func evaluate(f linear.Function, x []float64) float64 {
	last := len(x) - 1
	if x[last] < f(x[1:last]) {
		return -1
	}
	return 1
}

// String returns the string representation of the current
// random function and the current data hold by vectors Xn, Yn and Wn.
//
func (lr *LinearRegression) String() string {
	var ret string
	ret = lr.Equation.String()
	for i := 0; i < lr.TrainingPoints; i++ {
		ret += fmt.Sprintf("X: %v", lr.Xn[i])
		ret += fmt.Sprintf("\t Y: %v\n", lr.Yn[i])
	}
	ret += fmt.Sprintln()
	ret += fmt.Sprintf("W: %v\n", lr.Wn)
	return ret
}

// LinearRegressionError returns the error defined by:
// Ed[Ein(wlin)] = sigma^2 (1 - (d + 1)/ N)
//
func LinearRegressionError(n int, sigma float64, d int) float64 {
	return sigma * sigma * (1 - (float64(d+1))/float64(n))
}
