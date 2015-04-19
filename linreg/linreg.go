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
type LinearRegression struct {
	Name                 string          // the name of this linear regression, empty by default.
	TrainingPoints       int             // number of training points.
	ValidationPoints     int             // number of validation point.
	RandomTargetFunction bool            // flag to know if target function is generated at random or defined by user.
	TwoParams            bool            // flag to know if target function takes two parameters.
	Noise                float64         // Noise should be btwn 0(no noise) and 1(all noise). Will simulate noise by flipping the sign of the output based on the Noise.
	Interval             linear.Interval // Interval in which the points, outputs and function are defined.
	Equation             linear.Equation // random equation that defines the random linear function: targetFunction.
	TargetFunction       linear.Function // linear target function to predict.
	TransformFunction    TransformFunc   // transformation function.
	Transform            bool            // determines if linear regression uses a transformation funtion, in which case 'TransformationFunction' should be defined.
	Xn                   [][]float64     // data set of points for training (if defined at random, they are uniformly present in Interval).
	XVal                 [][]float64     // data set of point  for validation.
	VectorSize           int             // size of vectors Xi and Wi.
	Yn                   []float64       // output, evaluation of each Xi based on linear function.
	YVal                 []float64       // output, for validation
	Wn                   []float64       // weight vector initialized at zeros.
	WReg                 []float64       // weight vector with regularization.
	Lambda               float64         // used in weight decay.
	K                    int             // used in weight decay.
}

// NewLinearRegression creates a linear regression object.
// TrainingPoints = 10
// Interval [-1 : 1]
// RandomTargetFunction is true
// Noise = 0
// VectorSize = 3
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
			fmt.Printf("x1 unable to parse line %d in file %s\n", n, filename)
			return err
		}

		if x2, err = strconv.ParseFloat(line[1], 64); err != nil {
			fmt.Printf("x2 unable to parse line %d in file %s\n", n, filename)
			return err
		}

		if y, err = strconv.ParseFloat(line[2], 64); err != nil {
			fmt.Printf("y unable to parse line %d in file %s\n", n, filename)
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
func (lr *LinearRegression) ApplyTransformation() {
	lr.Transform = true

	for i := 0; i < lr.TrainingPoints; i++ {
		Xtrans := lr.TransformFunction(lr.Xn[i])
		lr.Xn[i] = Xtrans
	}
	lr.VectorSize = len(lr.Xn[0])
	lr.Wn = make([]float64, lr.VectorSize)
}

// ApplyTransformationOnValidation transforms the XVal vector into
// XValtrans = TransformationFunction(XVal)
func (lr *LinearRegression) ApplyTransformationOnValidation() {
	for i := 0; i < lr.ValidationPoints; i++ {
		Xtrans := lr.TransformFunction(lr.XVal[i])
		lr.XVal[i] = Xtrans
	}
}

// Learn will compute the pseudo inverse X dager and set W vector accordingly
// XDagger = (X'X)^-1 X'
func (lr *LinearRegression) Learn() error {

	var err error

	// compute X' <=> X transpose
	var mXn ml.Matrix = lr.Xn
	var mXT ml.Matrix

	if mXT, err = mXn.Transpose(); err != nil {
		log.Println("unable to make transpose, %v", err)
		return err
	}

	// compute the product of X' and X
	var mXP ml.Matrix
	if mXP, err = mXT.Product(mXn); err != nil {
		log.Println("unable to make matrix product, %v", err)
		return err
	}

	// inverse XProduct
	var mXInv ml.Matrix
	if mXInv, err = mXP.Inverse(); err != nil {
		log.Println("unable to compute matrix inverse, %v", err)
		return err
	}

	// compute product: (X'X)^-1 X'
	var XDagger ml.Matrix
	if XDagger, err = mXInv.Product(mXT); err != nil {
		log.Println("unable to compute XDagger, %v", err)
		return err
	}

	lr.setWeight(XDagger)

	return nil
}

func (linreg *LinearRegression) setWeight(d ml.Matrix) {

	for i := 0; i < len(d); i++ {
		for j := 0; j < len(d[0]); j++ {
			linreg.Wn[i] += d[i][j] * linreg.Yn[j]
		}
	}
}

// set Wreg
func (linreg *LinearRegression) setWeightReg(d ml.Matrix) {

	linreg.WReg = make([]float64, linreg.VectorSize)

	for i := 0; i < len(d); i++ {
		for j := 0; j < len(d[0]); j++ {
			linreg.WReg[i] += d[i][j] * linreg.Yn[j]
		}
	}
}

// Ein is the fraction of in sample points which got misclassified.
func (linreg *LinearRegression) Ein() float64 {
	// XnWn
	gInSample := make([]float64, len(linreg.Xn))
	for i := 0; i < len(linreg.Xn); i++ {
		gi := float64(0)
		for j := 0; j < len(linreg.Xn[0]); j++ {
			gi += linreg.Xn[i][j] * linreg.Wn[j]
		}
		gInSample[i] = ml.Sign(gi)
	}
	nEin := 0
	for i := 0; i < len(gInSample); i++ {
		if gInSample[i] != linreg.Yn[i] {
			nEin++
		}
	}
	return float64(nEin) / float64(len(gInSample))

}

// EAug is the fraction of in sample points which got misclassified plus the term
// lambda / N * Sum(Wi^2)
func (linreg *LinearRegression) EAugIn() float64 {

	gInSample := make([]float64, len(linreg.Xn))
	for i := 0; i < len(linreg.Xn); i++ {
		gi := float64(0)
		for j := 0; j < len(linreg.Xn[0]); j++ {
			gi += linreg.Xn[i][j] * linreg.WReg[j]
		}
		gInSample[i] = ml.Sign(gi)
	}
	nEin := 0
	for i := 0; i < len(gInSample); i++ {
		if gInSample[i] != linreg.Yn[i] {
			nEin++
		}
	}

	return float64(nEin) / float64(len(gInSample))
}

func (linreg *LinearRegression) EValIn() float64 {

	gInSample := make([]float64, len(linreg.XVal))
	for i := 0; i < len(linreg.XVal); i++ {
		gi := float64(0)
		for j := 0; j < len(linreg.XVal[0]); j++ {
			gi += linreg.XVal[i][j] * linreg.Wn[j]
		}
		gInSample[i] = ml.Sign(gi)
	}
	nEin := 0
	for i := 0; i < len(gInSample); i++ {
		if gInSample[i] != linreg.YVal[i] {
			nEin++
		}
	}

	return float64(nEin) / float64(len(gInSample))
}

// Eout is the fraction of out of sample points which got misclassified.
func (linreg *LinearRegression) Eout() float64 {
	outOfSample := 1000
	numError := 0

	for i := 0; i < outOfSample; i++ {
		var oY float64
		oX := make([]float64, linreg.VectorSize)
		oX[0] = float64(1)
		for j := 1; j < len(oX); j++ {
			oX[j] = linreg.Interval.RandFloat()
		}
		flip := float64(1)
		if linreg.Noise != 0 {
			r := rand.New(rand.NewSource(time.Now().UnixNano()))
			rN := r.Intn(100)
			if rN < int(math.Ceil(linreg.Noise*100)) {
				flip = float64(-1)
			}
		}
		// output with potential noise in 'flip' variable
		if !linreg.TwoParams {
			oY = evaluate(linreg.TargetFunction, oX) * flip
		} else {
			oY = evaluateTwoParams(linreg.TargetFunction, oX) * flip
		}

		gi := float64(0)
		for j := 0; j < len(oX); j++ {
			gi += oX[j] * linreg.Wn[j]
		}

		if ml.Sign(gi) != float64(oY) {
			numError++
		}
	}
	return float64(numError) / float64(outOfSample)
}

// EoutFromFile only supports linear regressions with transformed data.
// todo:(santiaago) make this more generic.
func (linreg *LinearRegression) EoutFromFile(filename string) (float64, error) {

	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	numError := 0
	numberOfLines := 0
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
		var oY int
		var oX1, oX2 float64

		if x1, err := strconv.ParseFloat(line[0], 64); err != nil {
			fmt.Printf("x1 unable to parse line %d in file %s\n", numberOfLines, filename)
			return 0, err
		} else {
			oX1 = x1
		}
		if x2, err := strconv.ParseFloat(line[1], 64); err != nil {
			fmt.Printf("x2 unable to parse line %d in file %s\n", numberOfLines, filename)
			return 0, err
		} else {
			oX2 = x2
		}

		oX := linreg.TransformFunction([]float64{float64(1), oX1, oX2})

		if y, err := strconv.ParseFloat(line[2], 64); err != nil {
			fmt.Printf("y unable to parse line %d in file %s\n", numberOfLines, filename)
			return 0, err
		} else {
			oY = int(y)
		}

		gi := float64(0)
		for j := 0; j < len(oX); j++ {
			gi += oX[j] * linreg.Wn[j]
		}
		if ml.Sign(gi) != float64(oY) {
			numError++
		}
		numberOfLines++
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	return float64(numError) / float64(numberOfLines), nil
}

// Ein is the fraction of in sample points which got misclassified.
func (linreg *LinearRegression) EAugOutFromFile(filename string) (float64, error) {

	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	numError := 0
	numberOfLines := 0
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
		var oY int
		var oX1, oX2 float64

		if x1, err := strconv.ParseFloat(line[0], 64); err != nil {
			fmt.Printf("x1 unable to parse line %d in file %s\n", numberOfLines, filename)
			return 0, err
		} else {
			oX1 = x1
		}
		if x2, err := strconv.ParseFloat(line[1], 64); err != nil {
			fmt.Printf("x2 unable to parse line %d in file %s\n", numberOfLines, filename)
			return 0, err
		} else {
			oX2 = x2
		}

		oX := linreg.TransformFunction([]float64{float64(1), oX1, oX2})

		if y, err := strconv.ParseFloat(line[2], 64); err != nil {
			fmt.Printf("y unable to parse line %d in file %s\n", numberOfLines, filename)
			return 0, err
		} else {
			oY = int(y)
		}

		gi := float64(0)
		for j := 0; j < len(oX); j++ {
			gi += oX[j] * linreg.WReg[j]
		}
		if ml.Sign(gi) != float64(oY) {
			numError++
		}
		numberOfLines++
	}

	return float64(numError) / float64(numberOfLines), nil
}

// EReg
// (Z'Z+λI)^−1 * Z'
// WReg = (Z'Z + λI)^−1 Z'y
func (linreg *LinearRegression) LearnWeightDecay() error {
	linreg.Lambda = math.Pow(10, float64(linreg.K))

	// compute X' <=> X transpose
	XTranspose := make([][]float64, len(linreg.Xn[0]))
	for i := 0; i < len(linreg.Xn[0]); i++ {
		XTranspose[i] = make([]float64, len(linreg.Xn))
	}

	for i := 0; i < len(XTranspose); i++ {
		for j := 0; j < len(XTranspose[0]); j++ {
			XTranspose[i][j] = linreg.Xn[j][i]
		}
	}

	// compute lambda*Identity
	lambdaIdentity := make([][]float64, len(linreg.Xn[0]))
	for i := 0; i < len(lambdaIdentity); i++ {
		lambdaIdentity[i] = make([]float64, len(lambdaIdentity))
		lambdaIdentity[i][i] = float64(1) * linreg.Lambda
	}

	// compute Z'Z
	XProduct := make([][]float64, len(linreg.Xn[0]))
	for i := 0; i < len(linreg.Xn[0]); i++ {
		XProduct[i] = make([]float64, len(linreg.Xn[0]))
	}
	for k := 0; k < len(linreg.Xn[0]); k++ {
		for i := 0; i < len(XTranspose); i++ {
			for j := 0; j < len(XTranspose[0]); j++ {
				XProduct[i][k] += XTranspose[i][j] * linreg.Xn[j][k]
			}
		}
	}

	// compute Z'Z + lambda*I
	sumMatrix := make([][]float64, len(lambdaIdentity))
	for i := 0; i < len(sumMatrix); i++ {
		sumMatrix[i] = make([]float64, len(sumMatrix))
		for j := 0; j < len(sumMatrix); j++ {
			sumMatrix[i][j] = XProduct[i][j] + lambdaIdentity[i][j]
		}
	}

	// inverse
	toInverse := ml.Matrix(sumMatrix)
	inverseMatrix, err := toInverse.Inverse()
	if err != nil {
		return err
	}
	// compute product: inverseMatrix Z'
	XDagger := make([][]float64, len(sumMatrix))
	for i := 0; i < len(inverseMatrix); i++ {
		XDagger[i] = make([]float64, len(XTranspose[0]))
	}

	for k := 0; k < len(XTranspose[0]); k++ {
		for i := 0; i < len(inverseMatrix); i++ {
			for j := 0; j < len(inverseMatrix[0]); j++ {
				XDagger[i][k] += inverseMatrix[i][j] * XTranspose[j][k]
			}
		}
	}
	// set WReg
	linreg.setWeightReg(ml.Matrix(XDagger))
	return nil
}

// CompareInSample will compare the current hypothesis function learn by linear regression whith respect to 'f'
func (linreg *LinearRegression) CompareInSample(f linear.Function, nParams int) float64 {

	gInSample := make([]float64, len(linreg.Xn))
	fInSample := make([]float64, len(linreg.Xn))

	for i := 0; i < len(linreg.Xn); i++ {
		gi := float64(0)
		for j := 0; j < len(linreg.Xn[0]); j++ {
			gi += linreg.Xn[i][j] * linreg.Wn[j]
		}
		gInSample[i] = ml.Sign(gi)
		if nParams == 1 || nParams == 2 {
			fInSample[i] = f(linreg.Xn[i][1:])
		} else {
			log.Println("case not supported")
		}
	}

	// measure difference:
	diff := 0
	for i := 0; i < len(linreg.Xn); i++ {
		if gInSample[i] != fInSample[i] {
			diff++
		}
	}
	return float64(diff) / float64(len(linreg.Xn))
}

// CompareOutOfSample will compare the current hypothesis function learn by linear regression whith respect to 'f' out of sample
func (linreg *LinearRegression) CompareOutOfSample(f linear.Function, nParams int) float64 {

	outOfSample := 1000
	diff := 0

	for i := 0; i < outOfSample; i++ {
		//var oY int
		oX := make([]float64, linreg.VectorSize)
		oX[0] = float64(1)
		for j := 1; j < len(oX); j++ {
			oX[j] = linreg.Interval.RandFloat()
		}

		gi := float64(0)
		for j := 0; j < len(oX); j++ {
			gi += oX[j] * linreg.Wn[j]
		}
		if nParams == 2 || nParams == 1 {
			if ml.Sign(gi) != f(oX[1:]) {
				diff++
			}
		} else {
			log.Println("case not supported")
		}
	}

	return float64(diff) / float64(outOfSample)
}

type TransformFunc func([]float64) []float64

func (linreg *LinearRegression) TransformDataSet(f TransformFunc, newSize int) {
	for i := 0; i < len(linreg.Xn); i++ {
		oldXn := linreg.Xn[i]
		newXn := f(oldXn)
		linreg.Xn[i] = make([]float64, newSize)
		for j := 0; j < len(newXn); j++ {
			linreg.Xn[i][j] = newXn[j]
		}
	}
	linreg.Wn = make([]float64, newSize)
}

// evaluate will map function f in point p with respect to the current y point.
// if it stands on one side it is +1 else -1
// vector x is defined as x0, x1 .. , xn, y
// So linear.Function should take sub vector [x1, ... ,xn]
// todo: might change name to mapPoint
func evaluate(f linear.Function, x []float64) float64 {
	last := len(x) - 1
	if x[last] < f(x[1:last]) {
		return -1
	}
	return 1
}

// evaluate will map function f in point p with respect to the current y point.
// this evaluate version takes 2 parameters instead of a single one.
// if it stands on one side it is +1 else -1
// todo: might change name to mapPointTwoParams
// todo(santiaago): remove
func evaluateTwoParams(f linear.Function, p []float64) float64 {
	if p[3] < f(p[1:3]) {
		return -1
	}
	return 1
}

// String returns the string representation of the current
// random function and the current data hold by vectors Xn, Yn and Wn.
func (linreg *LinearRegression) String() string {
	var ret string
	ret = linreg.Equation.String()
	for i := 0; i < linreg.TrainingPoints; i++ {
		ret += fmt.Sprint("X: %v", linreg.Xn[i])
		ret += fmt.Sprintln("\t Y: %v", linreg.Yn[i])
	}
	ret += fmt.Sprintln()
	ret += fmt.Sprintln("W: %v", linreg.Wn)
	return ret
}

// Ed[Ein(wlin)] = sigma^2 (1 - (d + 1)/ N)
func LinearRegressionError(n int, sigma float64, d int) float64 {
	return sigma * sigma * (1 - (float64(d+1))/float64(n))
}
