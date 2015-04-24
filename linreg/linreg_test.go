package linreg

import (
	"math"
	"testing"

	"github.com/santiaago/ml/linear"
)

func TestNewLinearRegression(t *testing.T) {
	if lr := NewLinearRegression(); lr == nil {
		t.Errorf("got nil linear regression")
	}
}

func TestInitialize(t *testing.T) {
	lr := NewLinearRegression()

	lr.Initialize()

	if len(lr.Xn) != len(lr.Yn) {
		t.Errorf("got different size of vectors Xn Yn, wants same size")
	}

	if len(lr.Xn[0]) != len(lr.Wn) {
		t.Errorf("got different size of vectors Xn Wn, wants same size")
	}

	if len(lr.Xn) != lr.TrainingPoints {
		t.Errorf("got different size of vectors Xn and training points, wants same number")
	}

	for i := 0; i < len(lr.Xn); i++ {
		for j := 0; j < len(lr.Xn[0]); j++ {
			if lr.Xn[i][j] < lr.Interval.Min ||
				lr.Xn[i][j] > lr.Interval.Max {
				t.Errorf("got value of Xn[%d][%d] = %v, want it between %v and %v", i, j, lr.Xn[i][j], lr.Interval.Min, lr.Interval.Max)
			}
		}
	}

	for i := 0; i < len(lr.Yn); i++ {
		if lr.Yn[i] != float64(-1) && lr.Yn[i] != float64(1) {
			t.Errorf("got value of Yn[%v] = %v, want it equal to -1 or 1", i, lr.Yn[i])
		}
	}

}

func TestFlip(t *testing.T) {
	lr := NewLinearRegression()
	lr.Noise = 0
	for i := 0; i < 100; i++ {
		if v := lr.flip(); v != float64(1) {
			t.Errorf("got flip value = -1 wants 1")
		}
	}
	lr.Noise = 1
	for i := 0; i < 100; i++ {
		if v := lr.flip(); v != float64(-1) {
			t.Errorf("got flip value = 1 wants -1")
		}
	}

	lr.Noise = 0.5
	for i := 0; i < 100; i++ {
		if v := lr.flip(); v != float64(-1) && v != float64(1) {
			t.Errorf("got flip value = %v wants value equal to 1 or -1", v)
		}
	}
}

func TestInitializeFromFile(t *testing.T) {
	lr := NewLinearRegression()
	if err := lr.InitializeFromFile("TestInitializeFromFile.data"); err != nil {
		t.Errorf("%v", err)
	}
	if len(lr.Xn) != 6 || len(lr.Yn) != 6 {
		t.Errorf("got difference in size of Xn or Yn and data")
	}
	if len(lr.Xn) != lr.TrainingPoints {
		t.Errorf("got difference in size of Xn or TrainingPoints and data")
	}
	if len(lr.Xn[0]) != len(lr.Wn) {
		t.Errorf("got different size of vectors Xn Wn, wants same size")
	}
	if len(lr.Xn[0]) != lr.VectorSize || 3 != lr.VectorSize {
		t.Errorf("got difference in size of Xn[0] or data[0] with VectorSize")
	}
}

func TestInitializeFromData(t *testing.T) {
	data := [][]float64{
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
	}
	lr := NewLinearRegression()
	if err := lr.InitializeFromData(data); err != nil {
		t.Errorf("%v", err)
	}
	if len(lr.Xn) != len(data) || len(lr.Yn) != len(data) {
		t.Errorf("got difference in size of Xn or Yn and data")
	}

	if len(lr.Xn) != lr.TrainingPoints {
		t.Errorf("got difference in size of Xn or TrainingPoints and data")
	}

	if len(lr.Xn[0]) != len(lr.Wn) {
		t.Errorf("got different size of vectors Xn Wn, wants same size")
	}

	if len(lr.Xn[0]) != lr.VectorSize || len(data[0]) != lr.VectorSize {
		t.Errorf("got difference in size of Xn[0] or data[0] with VectorSize")
	}
}

func TestInitializeValidationFromData(t *testing.T) {
	//todo(santiaago): test this
}

func TestApplyTransformation(t *testing.T) {

	tf := func(a []float64) []float64 {
		for i := 1; i < len(a); i++ {
			a[i] = -a[i]
		}
		return a
	}

	data := [][]float64{
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
	}

	lr := NewLinearRegression()
	lr.InitializeFromData(data)
	lr.TransformFunction = tf
	lr.ApplyTransformation()

	for i := 0; i < lr.TrainingPoints; i++ {
		for j := 1; j < len(lr.Xn[i]); j++ {
			if lr.Xn[i][j] != -1 {
				t.Errorf("got %v wants -1", lr.Xn[i][j])
			}
		}
		if lr.Yn[i] != 1 {
			t.Errorf("got Yn[%v] = %v wants %v", i, lr.Yn[i], 1)
		}
	}

}

func TestApplyTransformationOnValidation(t *testing.T) {
	// todo(santiaago): test this
}

func TestLearn(t *testing.T) {
	lr := NewLinearRegression()
	data := [][]float64{
		{0.1, 1, 1},
		{0.2, 1, 1},
		{0.3, 1, 1},
		{1, 0.5, -1},
		{1, 0.6, -1},
		{1, 0.7, -1},
	}

	lr.InitializeFromData(data)
	lr.Learn()
	expectedWn := []float64{0.393, -1.967, 0.983}
	if !equal(expectedWn, lr.Wn) {
		t.Errorf("Weight vector is not correct: got %v, want %v", lr.Wn, expectedWn)
	}
}

func TestSetWeight(t *testing.T) {

	lr := NewLinearRegression()
	lr.VectorSize = 5
	lr.Yn = []float64{-1, -1, -1}
	lr.TrainingPoints = 5
	d := [][]float64{
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
	}

	lr.setWeight(d)

	expectedWn := []float64{-3, -3, -3, -3, -3}
	if !equal(expectedWn, lr.Wn) {
		t.Errorf("Weight vector is not correct: got %v, want %v", lr.Wn, expectedWn)
	}
}

func TestSetWeightReg(t *testing.T) {

	lr := NewLinearRegression()
	lr.VectorSize = 5
	lr.Yn = []float64{-1, -1, -1}
	lr.TrainingPoints = 5
	d := [][]float64{
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
	}

	lr.setWeightReg(d)

	expectedWReg := []float64{-3, -3, -3, -3, -3}
	if !equal(expectedWReg, lr.WReg) {
		t.Errorf("Weight vector is not correct: got %v, want %v", lr.WReg, expectedWReg)
	}
}

func TestEin(t *testing.T) {
	lr := NewLinearRegression()
	lr.Xn = [][]float64{
		{1, 0.1, 0.1},
		{1, 0.2, 0.2},
		{1, 0.3, 0.3},
		{1, 0.4, 0.4},
		{1, 0.5, 0.5},
		{1, 0.6, 0.6},
	}

	tests := []struct {
		Y           []float64
		Wn          []float64
		expectedEin float64
	}{
		{
			[]float64{1, 1, 1, -1, -1, -1},
			[]float64{-1, 0, 0},
			0.5,
		},
		{
			[]float64{-1, -1, -1, -1, -1, -1},
			[]float64{-1, 0, 0},
			0,
		},
		{
			[]float64{-1, -1, -1, -1, -1, -1},
			[]float64{1, 0, 0},
			1.0,
		},
	}

	for _, tt := range tests {
		lr.Yn = tt.Y
		lr.Wn = tt.Wn
		got := lr.Ein()
		want := tt.expectedEin
		if got != want {
			t.Errorf("Ein is not correct, got %v, want %v", got, want)
		}
	}
}

func TestEAugIn(t *testing.T) {
	lr := NewLinearRegression()
	lr.Xn = [][]float64{
		{1, 0.1, 0.1},
		{1, 0.2, 0.2},
		{1, 0.3, 0.3},
		{1, 0.4, 0.4},
		{1, 0.5, 0.5},
		{1, 0.6, 0.6},
	}

	tests := []struct {
		Y              []float64
		WReg           []float64
		expectedEAugIn float64
	}{
		{
			[]float64{1, 1, 1, -1, -1, -1},
			[]float64{-1, 0, 0},
			0.5,
		},
		{
			[]float64{-1, -1, -1, -1, -1, -1},
			[]float64{-1, 0, 0},
			0,
		},
		{
			[]float64{-1, -1, -1, -1, -1, -1},
			[]float64{1, 0, 0},
			1.0,
		},
	}

	for _, tt := range tests {
		lr.Yn = tt.Y
		lr.WReg = tt.WReg
		got := lr.EAugIn()
		want := tt.expectedEAugIn
		if got != want {
			t.Errorf("Ein is not correct, got %v, want %v", got, want)
		}
	}
}

func TestEValIn(t *testing.T) {
	lr := NewLinearRegression()
	lr.XVal = [][]float64{
		{1, 0.1, 0.1},
		{1, 0.2, 0.2},
		{1, 0.3, 0.3},
		{1, 0.4, 0.4},
		{1, 0.5, 0.5},
		{1, 0.6, 0.6},
	}

	tests := []struct {
		YVal           []float64
		Wn             []float64
		expectedEValIn float64
	}{
		{
			[]float64{1, 1, 1, -1, -1, -1},
			[]float64{-1, 0, 0},
			0.5,
		},
		{
			[]float64{-1, -1, -1, -1, -1, -1},
			[]float64{-1, 0, 0},
			0,
		},
		{
			[]float64{-1, -1, -1, -1, -1, -1},
			[]float64{1, 0, 0},
			1.0,
		},
	}

	for _, tt := range tests {
		lr.YVal = tt.YVal
		lr.Wn = tt.Wn
		got := lr.EValIn()
		want := tt.expectedEValIn
		if got != want {
			t.Errorf("Ein is not correct, got %v, want %v", got, want)
		}
	}
}

func TestEoutManual(t *testing.T) {
	lr := NewLinearRegression()

	lr.TargetFunction = func(a []float64) float64 {
		return 1
	}

	tests := []struct {
		Wn           []float64
		expectedEout float64
	}{
		{
			[]float64{-1, 0, 0},
			0,
		},
		{
			[]float64{1, 0, 0},
			1,
		},
	}

	for _, tt := range tests {
		lr.Wn = tt.Wn
		got := lr.Eout()
		want := tt.expectedEout
		if got != want {
			t.Errorf("Eout is not correct, got %v, want %v", got, want)
		}
	}
}

func TestEoutAndEinAreCloseWithEnoughTraining(t *testing.T) {
	lr := NewLinearRegression()
	lr.TrainingPoints = 10000
	lr.Initialize()
	lr.Learn()

	ein := lr.Ein()
	eout := lr.Eout()

	if math.Abs(eout-ein) < epsilon*0.1 {
		t.Errorf("got %v < %v want %v > %v ", eout, ein, eout, ein)
	}
}

func TestEoutFromFile(t *testing.T) {
	// todo(santiaago)
}

func TestEAugOutFromFile(t *testing.T) {
	// todo(santiaago)
}

func TestLearnWeightDecay(t *testing.T) {
	lr := NewLinearRegression()
	lr.K = 1
	lr.Xn = [][]float64{
		{1, 1, 1},
		{1, 2, 1},
		{1, 3, 1},
		{1, 4, 1},
		{1, 5, 1},
		{1, 6, 1},
	}

	lr.Yn = []float64{1, 1, 1, 1, 1, 1}

	lr.LearnWeightDecay()
	expectedWn := []float64{0.123, 0.156, 0.123}
	if !equal(expectedWn, lr.WReg) {
		t.Errorf("Weight vector is not correct: got %v, want %v", lr.WReg, expectedWn)
	}
}

func TestCompareInSample(t *testing.T) {

	tests := []struct {
		data          [][]float64
		W             []float64
		f             linear.Function
		expectedDelta float64
	}{
		{
			data: [][]float64{
				{1, 1, 1},
				{1, 2, 1},
				{1, 3, 1},
				{1, 4, 1},
				{1, 5, 1},
				{1, 6, 1},
			},
			W: []float64{1, 0, 0},
			f: func(x []float64) float64 {
				return -1
			},
			expectedDelta: 1.0,
		},
		{
			data: [][]float64{
				{1, 1, 1},
				{1, 2, 1},
				{1, 3, 1},
				{1, 4, 1},
				{1, 5, 1},
				{1, 6, 1},
			},
			W: []float64{1, 0, 0},
			f: func(x []float64) float64 {
				return 1
			},
			expectedDelta: 0,
		},
		{
			data: [][]float64{
				{-1, 1, 1},
				{-1, 2, 1},
				{-1, 3, 1},
				{1, 4, 1},
				{1, 5, 1},
				{1, 6, 1},
			},
			W: []float64{1, 0, 0},
			f: func(x []float64) float64 {
				return -1
			},
			expectedDelta: 0.5,
		},
	}

	for i, tt := range tests {
		lr := NewLinearRegression()
		lr.Xn = tt.data
		lr.Wn = tt.W

		got := lr.CompareInSample(tt.f)
		if got != tt.expectedDelta {
			t.Errorf("test %v: wrong delta btwn functions, got %v, want %v", i, got, tt.expectedDelta)
		}
	}
}

func TestCompareOutOfSample(t *testing.T) {

	tests := []struct {
		W             []float64
		f             linear.Function
		expectedDelta float64
	}{
		{
			W: []float64{1, 0, 0},
			f: func(x []float64) float64 {
				return 0
			},
			expectedDelta: 1.0,
		},
		{
			W: []float64{0, 0, 0},
			f: func(x []float64) float64 {
				return -1
			},
			expectedDelta: 0,
		},
	}

	for i, tt := range tests {
		lr := NewLinearRegression()
		lr.Wn = tt.W

		got := lr.CompareOutOfSample(tt.f)
		if got != tt.expectedDelta {
			t.Errorf("test %v: wrong delta btwn functions, got %v, want %v", i, got, tt.expectedDelta)
		}
	}
}

func TestTransformDataSet(t *testing.T) {
	tests := []struct {
		data              [][]float64
		transformFunction TransformFunc
		newSize           int
		expected          [][]float64
	}{
		{
			data: [][]float64{
				{1, 1, 1},
				{1, 1, 1},
				{1, 1, 1},
			},
			transformFunction: func(x []float64) []float64 {
				var xt []float64
				xt = append(xt, x...)
				xt = append(xt, 1)
				return xt
			},
			newSize: 4,
			expected: [][]float64{
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
			},
		},
		{
			data: [][]float64{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
			},
			transformFunction: func(x []float64) []float64 {
				var xt []float64
				xt = append(xt, 1)
				for i := range x {
					xt = append(xt, x[i]*x[i])
				}
				return xt
			},
			newSize: 4,
			expected: [][]float64{
				{1, 1, 4, 9},
				{1, 16, 25, 36},
				{1, 49, 64, 81},
			},
		},
	}

	for i, tt := range tests {
		lr := NewLinearRegression()
		lr.Xn = tt.data
		lr.TransformDataSet(tt.transformFunction, tt.newSize)
		if !equal2D(lr.Xn, tt.expected) {
			t.Errorf("test %v: got %v want %v", i, lr.Xn, tt.expected)
		}
	}
}

func TestEvaluate(t *testing.T) {
	tests := []struct {
		point    []float64
		f        linear.Function
		expected float64
	}{
		{
			point: []float64{1, 1},
			f: func(x []float64) float64 {
				return 1
			},
			expected: 1,
		},
		{
			point: []float64{1, -1},
			f: func(x []float64) float64 {
				return 1
			},
			expected: -1,
		},
	}

	for i, tt := range tests {
		got := evaluate(tt.f, tt.point)
		if got != tt.expected {
			t.Errorf("test %v: got %v want %v", i, got, tt.expected)
		}
	}
}

func TestString(t *testing.T) {
	lr := NewLinearRegression()
	lr.TrainingPoints = 1
	lr.Xn = [][]float64{
		{1, 1, 1},
	}
	lr.Yn = []float64{1}
	lr.Wn = []float64{1, 1, 1}
	expected := `f(X) = 0.00X + 0.00
X: [1 1 1]	 Y: 1

W: [1 1 1]
`
	got := lr.String()
	if got != expected {
		t.Errorf("got \n'%v', want \n'%v'", got, expected)
	}
}

func TestLinearRegressionError(t *testing.T) {
	got := LinearRegressionError(1, 2, 3)
	want := -12.0
	if got != want {
		t.Errorf("got %v, want %v", got, want)
	}
}

const epsilon float64 = 0.001

func equal(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if (a[i] - b[i]) > epsilon {
			return false
		}
	}
	return true
}

func equal2D(a, b [][]float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {

			if (a[i][j] - b[i][j]) > epsilon {
				return false
			}
		}
	}
	return true
}
