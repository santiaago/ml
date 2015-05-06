package ml

import "testing"

func TestVectorDot(t *testing.T) {
	var v Vector = []float64{1, 0, 0}
	var u Vector = []float64{0, 1, 0}
	want := float64(0)
	if got, err := v.Dot(u); err != nil {
		t.Errorf("Error computing Dot product")
	} else {
		if want != got {
			t.Errorf("got %v, want %v", got, want)
		}
	}
}
