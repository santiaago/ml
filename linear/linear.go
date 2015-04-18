// Package linear provides a set of linear methods.
package linear

import (
	"fmt"
	"math/rand"
	"time"
)

// An Interval is defined by a min and a max value.
type Interval struct {
	Min float64
	Max float64
}

// NewInterval returns a new Interval between min and max.
// If min is bigger than max it returns Interval{0,0}
func NewInterval(min, max float64) Interval {
	if max < min {
		return Interval{}
	}
	return Interval{min, max}
}

// RandFloat returns a random float number with respect to the interval.
func (i *Interval) RandFloat() float64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	size := float64(i.Max - i.Min)
	return r.Float64()*size + float64(i.Min)
}

// An Equation type holds the two variables slope and intercept that define a linear function.
type Equation struct {
	A float64 // slope
	B float64 // intercept
}

// RandEquation returns an equation object with with a random slope and a random intercept.
// The values depend on the interval passed as param.
func RandEquation(i Interval) Equation {
	return Equation{i.RandFloat(), i.RandFloat()}
}

// A Function is a linear function that takes a float64 and returns a float64
// f(x) = y
type Function func(x float64) float64

// Function returns a linear function with respect of the defined equation.
// f(x) = ax + b
// With a and b defined by Equation
func (eq *Equation) Function() Function {
	return func(x float64) float64 {
		return x*eq.A + eq.B
	}
}

// String returns the string representation of the Equation type
// f(X) = aX + b
func (eq *Equation) String() string {
	return fmt.Sprintf("f(X) = %4.2fX + %4.2f\n", eq.A, eq.B)
}
