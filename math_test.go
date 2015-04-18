package ml

import "testing"

func TestSign(t *testing.T) {
	tests := []struct {
		param    float64
		expected float64
	}{
		{float64(1), 1},
		{float64(-1), -1},
		{float64(0), -1},
		{float64(0.1), 1},
		{float64(-0.1), -1},
	}
	for _, tt := range tests {
		got := Sign(tt.param)
		if got != tt.expected {
			t.Errorf("Sign(%v) = %v, expected %v", tt.param, got, tt.expected)
		}
	}
}
