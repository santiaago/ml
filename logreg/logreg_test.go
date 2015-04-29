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
		got := lr.Gradient(tt.w, tt.y)
		if !equal(got, tt.want) {
			t.Errorf("test %i: got Gradient = %v, want %v", i, got, tt.want)
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
		lr.UpdateWeights(tt.gradientVector)
		got := lr.Wn
		if !equal(got, tt.want) {
			t.Errorf("test %v: got Wn:%v, want %v", i, got, tt.want)
		}
	}
}

func TestConverged(t *testing.T) {
	tests := []struct {
		wn   []float64
		w    []float64
		want bool
	}{
		{
			wn:   []float64{0.1, 0.1, 0.1},
			w:    []float64{0.1, 0.1, 0.1},
			want: true,
		},
	}

	for i, tt := range tests {
		lr := NewLogisticRegression()
		lr.Wn = tt.wn
		got := lr.Converged(tt.w)
		if got != tt.want {
			t.Errorf("test %v: got converged = %v, wants %v", i, got, tt.want)
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
