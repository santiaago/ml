package ml

import "testing"

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
			want: float64(0),
		},
	}

	for i, tt := range tests {
		if got, err := tt.v.Dot(tt.u); err != nil {
			t.Errorf("Error computing Dot product")
		} else {
			if tt.want != got {
				t.Errorf("test %v: got %v, want %v", i, got, tt.want)
			}
		}
	}
}
