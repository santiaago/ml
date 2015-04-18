package ml

import (
	"fmt"
	"strings"
	"testing"
)

var (
	epsilon float64 = 0.001
)

func TestInverse(t *testing.T) {

	tests := []struct {
		m        matrix
		expected matrix
		err      string
	}{
		{
			m: matrix{
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
			},
			err: "singular",
		},
		{
			m: matrix{
				{0, 0, 0},
				{0, 0, 0},
			},
			err: "square",
		},
		{
			m: matrix{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
			expected: matrix{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
		},

		{
			m: matrix{
				{4, 3},
				{3, 2},
			},
			expected: matrix{
				{-2, 3},
				{3, -4},
			},
		},
		{
			m: matrix{
				{4, 7},
				{2, 6},
			},
			expected: matrix{
				{0.6, -0.7},
				{-0.2, 0.4},
			},
		},
		{
			m: matrix{
				{1, 2, 3},
				{0, 1, 4},
				{5, 6, 0},
			},
			expected: matrix{
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

	m := matrix{
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

func equal(a, b matrix) bool {
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
