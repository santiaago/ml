package logreg

import "testing"

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
