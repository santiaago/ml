package svm

import (
	"strings"
	"testing"
)

func TestNewSVM(t *testing.T) {
	m := NewSVM()
	if m.K != 1 {
		t.Errorf("got %v want %v", m.K, 1)
	}
	if m.Lambda != 0.01 {
		t.Errorf("got %v want %v", m.Lambda, 0.01)
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
	svm := NewSVM()
	if err := svm.InitializeFromData(data); err != nil {
		t.Errorf("%v", err)
	}
	if len(svm.Xn) != len(data) || len(svm.Yn) != len(data) {
		t.Errorf("got difference in size of Xn or Yn and data")
	}

	if len(svm.Xn) != svm.TrainingPoints {
		t.Errorf("got difference in size of Xn or TrainingPoints and data")
	}

	if len(svm.Xn[0]) != len(svm.Wn) {
		t.Errorf("got different size of vectors Xn Wn, wants same size")
	}

	if len(svm.Xn[0]) != svm.VectorSize || len(data[0]) != svm.VectorSize {
		t.Errorf("got difference in size of Xn[0] or data[0] with VectorSize")
	}

}

func TestLearn(t *testing.T) {
	svm := NewSVM()
	data := [][]float64{
		{0.1, 1, 1},
		{0.2, 1, 1},
		{0.3, 1, 1},
		{1, 0.5, -1},
		{1, 0.6, -1},
		{1, 0.7, -1},
	}

	svm.InitializeFromData(data)
	err := svm.Learn()
	if err != nil {
		t.Errorf("got %v, expected no error", err)
	}
	expectedWn := []float64{0.393, -1.967, 0.983}
	if !equal(expectedWn, svm.Wn) {
		t.Errorf("Weight vector is not correct: got %v, want %v", svm.Wn, expectedWn)
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

	svm := NewSVM()
	svm.InitializeFromData(data)
	svm.TransformFunction = tf
	svm.ApplyTransformation()

	for i := 0; i < svm.TrainingPoints; i++ {
		for j := 1; j < len(svm.Xn[i]); j++ {
			if svm.Xn[i][j] != -1 {
				t.Errorf("got %v wants -1", svm.Xn[i][j])
			}
		}
		if svm.Yn[i] != 1 {
			t.Errorf("got Yn[%v] = %v wants %v", i, svm.Yn[i], 1)
		}
	}

}

func TestEin(t *testing.T) {
	svm := NewSVM()
	svm.Xn = [][]float64{
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
		svm.Yn = tt.Y
		svm.Wn = tt.Wn
		got := svm.Ein()
		want := tt.expectedEin
		if got != want {
			t.Errorf("Ein is not correct, got %v, want %v", got, want)
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
		svm := NewSVM()
		svm.Wn = tt.w
		if got, err := svm.Predict(tt.x); err != nil {
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
			expected: []float64{32, 0, 0, 0, 32},
		},
	}

	for i, tt := range tests {
		svm := NewSVM()
		svm.Wn = tt.w
		if got, err := svm.Predictions(tt.data); err != nil {
			t.Errorf("test %v: got error %v, want %v", i, err, tt.err)
		} else if !equal(got, tt.expected) {
			t.Errorf("test %v: got %v ,want %v", i, got, tt.expected)
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
