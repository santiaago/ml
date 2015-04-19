package ml

import (
	"fmt"
	"strings"
	"testing"
)

var (
	epsilon float64 = 0.001
)

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
				t.Errorf("test %d: got error %v, want ", i, err, tt.err)
			} else if len(tt.err) == 0 {
				t.Errorf("test %d: Transpose got error %v", i, err)
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
				t.Errorf("test %d: got error %v, want ", i, err, tt.err)
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
