package linear

import (
	"fmt"
	"testing"
)

func TestNewInterval(t *testing.T) {
	i := NewInterval(0, 1)
	if i.Min != 0 || i.Max != 1 {
		t.Errorf("NewInterval want min = %v, max = %v, got min = %v, max = %v, got ", 0, 1, i.Min, i.Max)
	}

	i = NewInterval(1, 0)
	if i.Min != 0 || i.Max != 0 {
		t.Errorf("NewInterval want min = %v, max = %v, got min = %v, max = %v, got ", 0, 0, i.Min, i.Max)
	}
}

func TestRandFloat(t *testing.T) {
	i := NewInterval(-1, 1)
	for j := 0; j < 100; j++ {
		if r := i.RandFloat(); r < -1 || r >= 1 {
			t.Errorf("RandFloat want results between %v and %v, got %v ", i.Min, i.Max, r)
		}
	}
}

func TestRandEquation(t *testing.T) {
	i := NewInterval(-1, 1)
	if eq := RandEquation(i); eq.A < i.Min || eq.A > i.Max || eq.B < i.Min || eq.B > i.Max {
		t.Errorf("RandFloat want results vars between %v and %v, got %+v", i.Min, i.Max, eq)
	}
}

func TestEquationFunction(t *testing.T) {

	tests := []struct {
		eq       Equation
		params   []float64
		expected float64
	}{
		{Equation{1, 1}, []float64{1}, float64(2)},
		{Equation{1, 0}, []float64{1}, float64(1)},
		{Equation{0, 1}, []float64{1}, float64(1)},
		{Equation{0, 0}, []float64{1}, float64(0)},
	}

	for _, tt := range tests {
		f := tt.eq.Function()
		got := f(tt.params)
		if got != tt.expected {
			t.Errorf("Equation.Function want f(%v) = %v, got f(%v) = %v", tt.params, tt.expected, tt.params, got)
		}
	}
}

func TestEquationFunctionString(t *testing.T) {
	eq := Equation{1, 1}
	expected := fmt.Sprintf("f(X) = %4.2fX + %4.2f\n", float64(1), float64(1))
	got := eq.String()
	if expected != got {
		t.Errorf("Equation.String want %v, got %v", expected, got)
	}
}
