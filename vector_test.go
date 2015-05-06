package ml

import (
	"math"
	"strings"
	"testing"
)

func TestVectorDot(t *testing.T) {
	tests := []struct {
		v    Vector
		u    Vector
		want float64
		err  string
	}{
		{
			v:    []float64{1, 0, 0},
			u:    []float64{0, 1, 0},
			want: 0,
		},
		{
			v:    []float64{1, 1, 1},
			u:    []float64{1, 1, 1},
			want: 3,
		},
		{
			v:    []float64{0.1, 0.2, 0.3},
			u:    []float64{0.4, 0.5, 0.6},
			want: 0.32,
		},
		{
			v:    []float64{1, 0, 0},
			u:    []float64{0, 1},
			want: float64(0),
			err:  "size",
		},
	}

	for i, tt := range tests {
		if got, err := tt.v.Dot(tt.u); err != nil {
			if !strings.Contains(errstring(err), tt.err) && len(tt.err) == 0 {
				t.Errorf("test %d: got error %v, want %v", i, err, tt.err)
			} else if len(tt.err) == 0 {
				t.Errorf("Error computing Dot product")
			}
		} else {
			if tt.want != got {
				t.Errorf("test %v: got %v, want %v", i, got, tt.want)
			}
		}
	}
}

func TestVectorNorm(t *testing.T) {
	tests := []struct {
		v    Vector
		want float64
		err  string
	}{
		{
			v:    []float64{1, 0, 0},
			want: 1,
		},
		{
			v:    []float64{1, 1, 1},
			want: 1.7320,
		},
		{
			v:    []float64{0.1, 0.2, 0.3},
			want: 0.3741,
		},
		{
			v:    []float64{0, 0, 0},
			want: 0,
		},
	}

	for i, tt := range tests {
		if got, err := tt.v.Norm(); err != nil {
			if !strings.Contains(errstring(err), tt.err) && len(tt.err) == 0 {
				t.Errorf("test %d: got error %v, want %v", i, err, tt.err)
			} else if len(tt.err) == 0 {
				t.Errorf("Error computing the Norm")
			}
		} else {
			if math.Abs(tt.want-got) > epsilon {
				t.Errorf("test %v: got %v, want %v", i, got, tt.want)
			}
		}
	}
}
