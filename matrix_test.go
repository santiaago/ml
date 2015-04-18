package ml

import "testing"

func testInverse(t *testing.T) {

	tests := []struct {
		m        matrix
		expected matrix
	}{
		{
			matrix{
				{1, 0, 0},
				{0, 1, 0},
				{0, 1, 0},
			},
			matrix{
				{1, 0, 0},
				{0, 1, 0},
				{0, 1, 0},
			},
		},
		{
			matrix{
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
			},
			matrix{
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
			},
		},
		{
			matrix{
				{1, 2, 3},
				{0, 1, 4},
				{5, 6, 0},
			},
			matrix{
				{-24, 18, 5},
				{20, -15, -4},
				{-5, 4, 1},
			},
		},
	}

	for _, tt := range tests {
		if got, err := tt.m.Inverse(); err != nil {
			t.Errorf("Inverse got error %v", err)
		} else if !equal(got, tt.expected) {
			t.Errorf("Inverse got %v want %v", got, tt.expected)
		}
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
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}
