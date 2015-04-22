package linreg

import "testing"

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
	// todo(santiaago): make this test.
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
