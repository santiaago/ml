package ml

import "fmt"

// Vector is a type to make vector operations.
type Vector []float64

// Dot performs the dot product of vectors 'v' and 'u'.
func (v Vector) Dot(u Vector) (float64, error) {
	if len(v) != len(u) {
		return 0, fmt.Errorf("both vectors should have same size")
	}
	var res float64
	for i := range v {
		res += v[i] * u[i]
	}
	return res, nil
}
