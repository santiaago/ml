package ml

import (
	"fmt"
	"math"
)

const epsilon float64 = 0.001

// Vector is a type to make vector operations.
//
type Vector []float64

// Dot performs the dot product of vectors 'v' and 'u'.
//
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

// Norm performs the norm operation of the vector 'v' passed as argument.
//
func (v Vector) Norm() (float64, error) {

	v2, err := v.Dot(v)
	if err != nil {
		return 0, err
	}
	return math.Sqrt(v2), nil
}

// Scale performs a multiplication between a factor f and vector v.
//
func (v Vector) Scale(f float64) (u Vector) {

	for i := range v {
		u = append(u, v[i]*f)
	}
	return
}

// Add performs a element by element adition of two vectors.
//
func (v Vector) Add(u Vector) (Vector, error) {
	if len(v) != len(u) {
		return nil, fmt.Errorf("vectors should have same size")
	}
	var w Vector
	for i := range v {
		w = append(w, v[i]+u[i])
	}
	return w, nil
}

// Equal performs a element by element comparison to check if
// two vectors are equal
func (v Vector) equal(u Vector) bool {
	if len(v) != len(u) {
		return false
	}
	for i := range v {
		if math.Abs(v[i]-u[i]) > epsilon {
			return false
		}
	}
	return true
}
