package data

import "testing"

func TestFilter(t *testing.T) {
	tests := []struct {
		data     [][]float64
		keep     []int
		expected [][]float64
	}{
		{
			data: [][]float64{
				{1, 2, 3},
				{1, 2, 3},
				{1, 2, 3},
			},
			keep: []int{0},
			expected: [][]float64{
				{1},
				{1},
				{1},
			},
		},
		{
			data: [][]float64{
				{1, 2, 3},
				{1, 2, 3},
				{1, 2, 3},
			},
			keep: []int{0, 2},
			expected: [][]float64{
				{1, 3},
				{1, 3},
				{1, 3},
			},
		},
		{
			data: [][]float64{
				{1, 2, 3},
				{1, 2, 3},
				{1, 2, 3},
			},
			keep: []int{},
			expected: [][]float64{
				{},
				{},
				{},
			},
		},
	}

	for i, tt := range tests {
		container := Container{tt.data, []int{}, 0}
		got := container.Filter(tt.keep)
		if !equal2D(got, tt.expected) {
			t.Errorf("test %v:got %v, want %v", i, got, tt.expected)
		}
	}
}

func TestFilterWithPredict(t *testing.T) {
	tests := []struct {
		data     [][]float64
		predict  int
		keep     []int
		expected [][]float64
	}{
		{
			data: [][]float64{
				{1, 2, 3, 1},
				{1, 2, 3, 1},
				{1, 2, 3, 1},
			},
			predict: 3,
			keep:    []int{0},
			expected: [][]float64{
				{1, 1},
				{1, 1},
				{1, 1},
			},
		},
		{
			data: [][]float64{
				{1, 2, 3, 1},
				{1, 2, 3, 1},
				{1, 2, 3, 1},
			},
			predict: 3,
			keep:    []int{0, 2},
			expected: [][]float64{
				{1, 3, 1},
				{1, 3, 1},
				{1, 3, 1},
			},
		},
		{
			data: [][]float64{
				{1, 2, 3, 1},
				{1, 2, 3, 1},
				{1, 2, 3, 1},
			},
			predict: 3,
			keep:    []int{},
			expected: [][]float64{
				{1},
				{1},
				{1},
			},
		},
	}

	for i, tt := range tests {
		container := Container{tt.data, []int{}, tt.predict}
		got := container.FilterWithPredict(tt.keep)
		if !equal2D(got, tt.expected) {
			t.Errorf("test %v:got %v, want %v", i, got, tt.expected)
		}
	}
}

const epsilon float64 = 0.001

func equal2D(a, b [][]float64) bool {
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
