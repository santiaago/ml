package ml

import (
	"fmt"
	"strings"
	"testing"
)

const epsilon float64 = 0.001

func TestIdentity(t *testing.T) {

	tests := []struct {
		n        int
		expected Matrix
		err      string
	}{
		{
			n: 3,
			expected: Matrix{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
		},
		{
			n: 2,
			expected: Matrix{
				{1, 0},
				{0, 1},
			},
		},
		{
			n: 1,
			expected: Matrix{
				{1},
			},
		},
		{
			n:   0,
			err: "greater than",
		},
		{
			n:   -1,
			err: "greater than",
		},
	}

	for i, tt := range tests {
		if got, err := Identity(tt.n); err != nil {

			if !strings.Contains(errstring(err), tt.err) && len(tt.err) == 0 {
				t.Errorf("test %d: got error %v, want %v", i, err, tt.err)
			} else if len(tt.err) == 0 {
				t.Errorf("test %d: Identity got error %v", i, err)
			}
		} else if !equal(got, tt.expected) {
			t.Errorf("test %d: Transpose got %v want %v", i, got, tt.expected)
		}
	}

}

func TestScalar(t *testing.T) {

	tests := []struct {
		l        float64
		m        Matrix
		expected Matrix
		err      string
	}{
		{
			l: 3,
			m: Matrix{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
			expected: Matrix{
				{3, 0, 0},
				{0, 3, 0},
				{0, 0, 3},
			},
		},
		{
			l: 2,
			m: Matrix{
				{1, 1},
				{1, 1},
			},
			expected: Matrix{
				{2, 2},
				{2, 2},
			},
		},
		{
			l:        10,
			m:        Matrix{{1}},
			expected: Matrix{{10}},
		},
	}

	for i, tt := range tests {
		if got, err := tt.m.Scalar(tt.l); err != nil {

			if !strings.Contains(errstring(err), tt.err) && len(tt.err) == 0 {
				t.Errorf("test %d: got error %v, want %v", i, err, tt.err)
			} else if len(tt.err) == 0 {
				t.Errorf("test %d: Scalar got error %v", i, err)
			}
		} else if !equal(got, tt.expected) {
			t.Errorf("test %d: Scalar got %v want %v", i, got, tt.expected)
		}
	}

}

func TestAdd(t *testing.T) {

	tests := []struct {
		a        Matrix
		b        Matrix
		expected Matrix
		err      string
	}{
		{
			a: Matrix{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
			b: Matrix{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
			expected: Matrix{
				{2, 0, 0},
				{0, 2, 0},
				{0, 0, 2},
			},
		},
		{
			a: Matrix{
				{1, 1},
				{1, 1},
			},
			b: Matrix{
				{1, 2},
				{3, 4},
			},
			expected: Matrix{
				{2, 3},
				{4, 5},
			},
		},
		{
			a:        Matrix{{1}},
			b:        Matrix{{1}},
			expected: Matrix{{2}},
		},
		{
			a: Matrix{
				{1, 1},
				{1, 1},
			},
			b: Matrix{
				{3, 4},
			},
			err: "dimensions",
		},
	}

	for i, tt := range tests {
		if got, err := tt.a.Add(tt.b); err != nil {

			if !strings.Contains(errstring(err), tt.err) && len(tt.err) == 0 {
				t.Errorf("test %d: got error %v, want %v", i, err, tt.err)
			} else if len(tt.err) == 0 {
				t.Errorf("test %d: Scalar got error %v", i, err)
			}
		} else if !equal(got, tt.expected) {
			t.Errorf("test %d: Scalar got %v want %v", i, got, tt.expected)
		}
	}

}

func TestTranspose(t *testing.T) {
	tests := []struct {
		m        Matrix
		expected Matrix
		err      string
	}{
		{
			m: Matrix{
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
			},
			expected: Matrix{
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
			},
		},
		{
			m: Matrix{
				{0, 0, 0},
				{0, 0, 0},
			},
			expected: Matrix{
				{0, 0},
				{0, 0},
				{0, 0},
			},
		},
		{
			m: Matrix{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
			expected: Matrix{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
		},
		{
			m: Matrix{
				{4, 2},
				{3, 2},
			},
			expected: Matrix{
				{4, 3},
				{2, 2},
			},
		},
		{
			m: Matrix{
				{4, 7},
				{2, 6},
				{1, 3},
				{0, 1},
			},
			expected: Matrix{
				{4, 2, 1, 0},
				{7, 6, 3, 1},
			},
		},
		{
			m: Matrix{
				{1, 2, 3},
				{0, 1, 4},
				{5, 6, 0},
			},
			expected: Matrix{
				{1, 0, 5},
				{2, 1, 6},
				{3, 4, 0},
			},
		},
	}

	for i, tt := range tests {
		if got, err := tt.m.Transpose(); err != nil {

			if !strings.Contains(errstring(err), tt.err) && len(tt.err) == 0 {
				t.Errorf("test %d: got error %v, want %v", i, err, tt.err)
			} else if len(tt.err) == 0 {
				t.Errorf("test %d: Transpose got error %v", i, err)
			}
		} else if !equal(got, tt.expected) {
			t.Errorf("test %d: Transpose got %v want %v", i, got, tt.expected)
		}
	}

}

func TestProduct(t *testing.T) {
	tests := []struct {
		a        Matrix
		b        Matrix
		expected Matrix
		err      string
	}{
		{
			a: Matrix{
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
			},
			b: Matrix{
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
			},
			expected: Matrix{
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
			},
		},
		{
			a: Matrix{
				{0, 0, 0},
				{0, 0, 0},
			},
			b: Matrix{
				{0, 0},
				{0, 0},
				{0, 0},
			},
			expected: Matrix{
				{0, 0},
				{0, 0},
			},
		},
		{
			a: Matrix{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
			b: Matrix{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
			expected: Matrix{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
		},
		{
			a: Matrix{
				{4, 2},
				{3, 2},
			},
			b: Matrix{
				{4, 3},
				{2, 2},
			},
			expected: Matrix{
				{20, 16},
				{16, 13},
			},
		},
		{
			a: Matrix{
				{4, 7},
				{2, 6},
				{1, 3},
				{0, 1},
			},
			b: Matrix{
				{4, 2, 1, 0},
				{7, 6, 3, 1},
			},
			expected: Matrix{
				{65, 50, 25, 7},
				{50, 40, 20, 6},
				{25, 20, 10, 3},
				{7, 6, 3, 1},
			},
		},
		{
			a: Matrix{
				{1, 1, 1},
				{1, 1, 1},
			},
			b: Matrix{
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
			},
			err: "Product",
		},
	}

	for i, tt := range tests {
		if got, err := tt.a.Product(tt.b); err != nil {

			if !strings.Contains(errstring(err), tt.err) && len(tt.err) == 0 {
				t.Errorf("test %d: got error %v, want %v", i, err, tt.err)
			} else if len(tt.err) == 0 {
				t.Errorf("test %d: Product got error %v", i, err)
			}
		} else if !equal(got, tt.expected) {
			t.Errorf("test %d: Transpose got %v want %v", i, got, tt.expected)
		}
	}

}

func TestInverse(t *testing.T) {

	tests := []struct {
		m        Matrix
		expected Matrix
		err      string
	}{
		{
			m: Matrix{
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
			},
			err: "singular",
		},
		{
			m: Matrix{
				{0, 0, 0},
				{0, 0, 0},
			},
			err: "square",
		},
		{
			m: Matrix{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
			expected: Matrix{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
		},

		{
			m: Matrix{
				{4, 3},
				{3, 2},
			},
			expected: Matrix{
				{-2, 3},
				{3, -4},
			},
		},
		{
			m: Matrix{
				{4, 7},
				{2, 6},
			},
			expected: Matrix{
				{0.6, -0.7},
				{-0.2, 0.4},
			},
		},
		{
			m: Matrix{
				{1, 2, 3},
				{0, 1, 4},
				{5, 6, 0},
			},
			expected: Matrix{
				{-24, 18, 5},
				{20, -15, -4},
				{-5, 4, 1},
			},
		},
	}

	for i, tt := range tests {
		if got, err := tt.m.Inverse(); err != nil {

			if !strings.Contains(errstring(err), tt.err) && len(tt.err) == 0 {
				t.Errorf("test %d: got error %v, want %v", i, err, tt.err)
			} else if len(tt.err) == 0 {
				t.Errorf("test %d: Inverse got error %v", i, err)
			}
		} else if !equal(got, tt.expected) {
			t.Errorf("test %d: Inverse got %v want %v", i, got, tt.expected)
		}
	}
}

func TestMatrixString(t *testing.T) {

	m := Matrix{
		{0, 0, 0},
		{0, 0, 0},
		{0, 0, 0},
	}
	expected := fmt.Sprintf("0.00\t0.00\t0.00\t\n0.00\t0.00\t0.00\t\n0.00\t0.00\t0.00\t\n")

	got := m.String()
	if got != expected {
		t.Errorf("String got \n%v\nwant \n%v", got, expected)
	}
}

func equal(a, b Matrix) bool {
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

// errstring returns the string representation of an error.
func errstring(err error) string {
	if err != nil {
		return err.Error()
	}
	return ""
}
