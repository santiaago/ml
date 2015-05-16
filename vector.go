package ml

import (
	"fmt"
	"math"
)

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
