package svm

import "testing"

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
