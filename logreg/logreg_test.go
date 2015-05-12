package logreg

import (
	"math"
	"strings"
	"testing"

	"github.com/santiaago/ml/linear"
)

func TestNewLogisticRegression(t *testing.T) {
	if lr := NewLogisticRegression(); lr == nil {
		t.Errorf("got nil linear regression")
	}
}

func TestInitialize(t *testing.T) {
	lr := NewLogisticRegression()
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

func TestInitializeFromData(t *testing.T) {
	data := [][]float64{
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
	}

	lr := NewLogisticRegression()
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

func TestInitializeFromFile(t *testing.T) {
	lr := NewLogisticRegression()
	if err := lr.InitializeFromFile("init.data"); err != nil {
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

func TestLearn(t *testing.T) {
	lr := NewLogisticRegression()
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
	expectedWn := []float64{0.06586, -0.99194, 0.56851}
	if !equal(expectedWn, lr.Wn) {
		t.Errorf("Weight vector is not correct: got %v, want %v", lr.Wn, expectedWn)
	}
}

func TestLearnRegularized(t *testing.T) {
	lr := NewLogisticRegression()
	data := [][]float64{
		{0.1, 1, 1},
		{0.2, 1, 1},
		{0.3, 1, 1},
		{1, 0.5, -1},
		{1, 0.6, -1},
		{1, 0.7, -1},
	}

	lr.InitializeFromData(data)
	lr.LearnRegularized()
	expectedWReg := []float64{0.06538, -0.99309, 0.567615}
	if !equal(expectedWReg, lr.WReg) {
		t.Errorf("Weight vector is not correct: got %v, want %v", lr.WReg, expectedWReg)
	}
}

func TestGradient(t *testing.T) {
	tests := []struct {
		w    []float64
		y    float64
		wn   []float64
		want []float64
	}{
		{
			w:    []float64{0, 0, 0},
			y:    1,
			wn:   []float64{0, 0, 0, 0},
			want: []float64{-0.5, 0, 0, 0},
		},
		{
			w:    []float64{1, 0, 0},
			y:    1,
			wn:   []float64{0, 0, 0, 0},
			want: []float64{-0.5, -0.5, 0, 0},
		},
		{
			w:    []float64{1, 1, 0},
			y:    1,
			wn:   []float64{0, 0, 0, 0},
			want: []float64{-0.5, -0.5, -0.5, 0},
		},
		{
			w:    []float64{1, 1, 1},
			y:    1,
			wn:   []float64{0, 0, 0, 0},
			want: []float64{-0.5, -0.5, -0.5, -0.5},
		},
	}
	for i, tt := range tests {
		lr := NewLogisticRegression()
		lr.Wn = tt.wn
		got, err := lr.Gradient(tt.w, tt.y)
		if err != nil {
			t.Errorf("test %v: got error %v", i, err)
		}
		if !equal(got, tt.want) {
			t.Errorf("test %v: got Gradient = %v, want %v", i, got, tt.want)
		}
	}
}

func TestGradientRegularized(t *testing.T) {
	tests := []struct {
		w    []float64
		y    float64
		wn   []float64
		want []float64
	}{
		{
			w:    []float64{0, 0, 0},
			y:    1,
			wn:   []float64{0, 0, 0, 0},
			want: []float64{-0.5, 0, 0, 0},
		},
		{
			w:    []float64{1, 0, 0},
			y:    1,
			wn:   []float64{0, 0, 0, 0},
			want: []float64{-0.5, -0.5, 0, 0},
		},
		{
			w:    []float64{1, 1, 0},
			y:    1,
			wn:   []float64{0, 0, 0, 0},
			want: []float64{-0.5, -0.5, -0.5, 0},
		},
		{
			w:    []float64{1, 1, 1},
			y:    1,
			wn:   []float64{0, 0, 0, 0},
			want: []float64{-0.5, -0.5, -0.5, -0.5},
		},
	}
	for i, tt := range tests {
		lr := NewLogisticRegression()
		lr.WReg = tt.wn
		got, err := lr.GradientRegularized(tt.w, tt.y)
		if err != nil {
			t.Errorf("test %v: got error %v", i, err)
		}
		if !equal(got, tt.want) {
			t.Errorf("test %v: got Gradient = %v, want %v", i, got, tt.want)
		}
	}
}

func TestUpdateWeights(t *testing.T) {
	tests := []struct {
		eta            float64
		w              []float64
		gradientVector []float64
		want           []float64
	}{
		{
			eta:            0.1,
			w:              []float64{1, 1, 1},
			gradientVector: []float64{1, 1, 1},
			want:           []float64{0.9, 0.9, 0.9},
		},
		{
			eta:            0.5,
			w:              []float64{1, 1, 1},
			gradientVector: []float64{1, 1, 1},
			want:           []float64{0.5, 0.5, 0.5},
		},
		{
			eta:            0,
			w:              []float64{1, 1, 1},
			gradientVector: []float64{1, 1, 1},
			want:           []float64{1, 1, 1},
		},
		{
			eta:            0.1,
			w:              []float64{0, 0, 0},
			gradientVector: []float64{1, 1, 1},
			want:           []float64{0.1, 0.1, 0.1},
		},
		{
			eta:            0.1,
			w:              []float64{1, 1, 1},
			gradientVector: []float64{0, 0, 0},
			want:           []float64{1, 1, 1},
		},
	}

	for i, tt := range tests {
		lr := NewLogisticRegression()
		lr.Eta = tt.eta
		lr.Wn = tt.w
		err := lr.UpdateWeights(tt.gradientVector)
		if err != nil {
			t.Errorf("test %v: got error %v", i, err)
		}
		got := lr.Wn
		if !equal(got, tt.want) {
			t.Errorf("test %v: got Wn:%v, want %v", i, got, tt.want)
		}
	}
}

func TestUpdateRegularizedWeights(t *testing.T) {
	tests := []struct {
		eta            float64
		w              []float64
		gradientVector []float64
		want           []float64
	}{
		{
			eta:            0.1,
			w:              []float64{1, 1, 1},
			gradientVector: []float64{1, 1, 1},
			want:           []float64{0.9, 0.9, 0.9},
		},
		{
			eta:            0.5,
			w:              []float64{1, 1, 1},
			gradientVector: []float64{1, 1, 1},
			want:           []float64{0.5, 0.5, 0.5},
		},
		{
			eta:            0,
			w:              []float64{1, 1, 1},
			gradientVector: []float64{1, 1, 1},
			want:           []float64{1, 1, 1},
		},
		{
			eta:            0.1,
			w:              []float64{0, 0, 0},
			gradientVector: []float64{1, 1, 1},
			want:           []float64{0.1, 0.1, 0.1},
		},
		{
			eta:            0.1,
			w:              []float64{1, 1, 1},
			gradientVector: []float64{0, 0, 0},
			want:           []float64{1, 1, 1},
		},
	}

	for i, tt := range tests {
		lr := NewLogisticRegression()
		lr.Eta = tt.eta
		lr.WReg = tt.w
		err := lr.UpdateRegularizedWeights(tt.gradientVector)
		if err != nil {
			t.Errorf("test %v: got error %v", i, err)
		}
		got := lr.WReg
		if !equal(got, tt.want) {
			t.Errorf("test %v: got Wn:%v, want %v", i, got, tt.want)
		}
	}
}

func TestConverged(t *testing.T) {
	tests := []struct {
		epsilon float64
		wn      []float64
		w       []float64
		want    bool
	}{
		{
			epsilon: 0.001,
			wn:      []float64{0.1, 0.1, 0.1},
			w:       []float64{0.1, 0.1, 0.1},
			want:    true,
		},
		{
			epsilon: 0.001,
			wn:      []float64{0.1, 0.1, 0.1},
			w:       []float64{0.1, 0.1, 0.2},
			want:    false,
		},
		{
			epsilon: 0.2,
			wn:      []float64{0.1, 0.1, 0.1},
			w:       []float64{0.09, 0.1, 0.2},
			want:    true,
		},
	}

	for i, tt := range tests {
		lr := NewLogisticRegression()
		lr.Wn = tt.wn
		lr.Epsilon = tt.epsilon
		got := lr.Converged(tt.w)
		if got != tt.want {
			t.Errorf("test %v: got converged = %v, wants %v", i, got, tt.want)
		}
	}
}

func TestEin(t *testing.T) {
	lr := NewLogisticRegression()
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

func TestEcv(t *testing.T) {
	lr := NewLogisticRegression()
	lr.TrainingPoints = 6
	lr.Xn = [][]float64{
		{1, -0.71, 0.331},
		{1, 0.27, -0.95},
		{1, -0.37, 0.12},
		{1, -0.49, -0.52},
		{1, 0.53, -0.11},
		{1, 0.62, 0.9},
	}

	tests := []struct {
		Y           []float64
		expectedEcv float64
	}{
		{
			Y:           []float64{1, 1, 1, 1, 1, 1},
			expectedEcv: 0,
		},
		{
			Y:           []float64{-1, -1, -1, -1, -1, -1},
			expectedEcv: 0,
		},
		{
			Y:           []float64{-1, 1, -1, 1, -1, 1},
			expectedEcv: 0.5,
		},
	}

	for i, tt := range tests {
		lr.Yn = tt.Y
		got := lr.Ecv()
		want := tt.expectedEcv
		if math.Abs(got-want) > epsilon {
			t.Errorf("test %v: got Ecv = %v, want %v", i, got, want)
		}
	}
}

func TestApplyTransformation(t *testing.T) {

	tf := func(a []float64) ([]float64, error) {
		for i := 1; i < len(a); i++ {
			a[i] = -a[i]
		}
		return a, nil
	}

	data := [][]float64{
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
	}

	lr := NewLogisticRegression()
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

func TestPredict(t *testing.T) {
	tests := []struct {
		w        []float64
		x        []float64
		expected float64
		err      string
	}{
		{
			w:        []float64{1, 0, 0},
			x:        []float64{1, 1, 1},
			expected: 1,
		},
		{
			w:        []float64{1, 1, 1},
			x:        []float64{1, 1, 1},
			expected: 3,
		},
		{
			w:        []float64{0, 0, 0},
			x:        []float64{1, 1, 1},
			expected: 0,
		},
		{
			w:        []float64{1, 2, 3},
			x:        []float64{4, 5, 6},
			expected: 32,
		},
		{
			w:        []float64{1, 2, 3},
			x:        []float64{4, 5},
			expected: 32,
			err:      "size",
		},
	}

	for i, tt := range tests {
		lr := NewLogisticRegression()
		lr.Wn = tt.w
		if got, err := lr.Predict(tt.x); err != nil {
			if !strings.Contains(errstring(err), tt.err) {
				t.Errorf("test %v: got error %v, want %v", i, err, tt.err)
			}
		} else if got != tt.expected {
			t.Errorf("test %v: got %v ,want %v", i, got, tt.expected)
		}
	}
}

func TestPredictions(t *testing.T) {

	tests := []struct {
		w        []float64
		data     [][]float64
		expected []float64
		err      string
	}{
		{
			w: []float64{1, 0, 0, 0},
			data: [][]float64{
				{1, 1, 1},
				{1, 1, 1},
				{1, 1, 1},
				{1, 1, 1},
				{1, 1, 1},
			},
			expected: []float64{1, 1, 1, 1, 1},
		},
		{
			w: []float64{1, 1, 1, 1},
			data: [][]float64{
				{1, 1, 1},
				{1, 1, 1},
				{1, 1, 1},
				{1, 1, 1},
				{1, 1, 1},
			},
			expected: []float64{4, 4, 4, 4, 4},
		},
		{
			w: []float64{0, 0, 0, 0},
			data: [][]float64{
				{1, 1, 1},
				{1, 1, 1},
				{1, 1, 1},
				{1, 1, 1},
				{1, 1, 1},
			},
			expected: []float64{0, 0, 0, 0, 0},
		},
		{
			w: []float64{1, 2, 3, 4},
			data: [][]float64{
				{4, 5, 6},
				{4, 5, 6},
				{4, 5, 6},
				{4, 5, 6},
				{4, 5, 6},
			},
			expected: []float64{32, 32, 32, 32, 32},
		},
		{
			w: []float64{1, 2, 3, 4},
			data: [][]float64{
				{4, 5, 6},
				{4, 5},
				{4, 5},
				{4, 5},
				{4, 5, 6},
			},
			err: "vector are different",
		},
	}

	for i, tt := range tests {
		lr := NewLogisticRegression()
		lr.Wn = tt.w
		if got, err := lr.Predictions(tt.data); err != nil {
			if !strings.Contains(errstring(err), tt.err) {
				t.Errorf("test %v: got error %v, want %v", i, err, tt.err)
			}
		} else if !equal(got, tt.expected) {
			t.Errorf("test %v: got %v ,want %v", i, got, tt.expected)
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

func TestCrossEntropyError(t *testing.T) {
	tests := []struct {
		sample []float64
		y      float64
		wn     []float64
		want   float64
	}{
		{
			sample: []float64{1, 1, 1},
			y:      float64(-1),
			wn:     []float64{1, 1, 1},
			want:   3.048,
		},
		{
			sample: []float64{1, 1, 1},
			y:      float64(1),
			wn:     []float64{1, 1, 1},
			want:   0.048,
		},
		{
			sample: []float64{1, 1, 1},
			y:      float64(0),
			wn:     []float64{1, 1, 1},
			want:   0.693,
		},
		{
			sample: []float64{1, 1, 1},
			y:      float64(1),
			wn:     []float64{0, 0, 0},
			want:   0.693,
		},
	}

	for i, tt := range tests {
		lr := NewLogisticRegression()
		lr.Wn = tt.wn
		got, err := lr.CrossEntropyError(tt.sample, tt.y)
		if err != nil {
			t.Errorf("test %v: got error %v", i, err)
		}
		if math.Abs(got-tt.want) > epsilon {
			t.Errorf("test %v: got %v, want %v", i, got, tt.want)
		}
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

// errstring returns the string representation of an error.
func errstring(err error) string {
	if err != nil {
		return err.Error()
	}
	return ""
}
