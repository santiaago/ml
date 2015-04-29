package logreg

import "testing"

func TestNewLogisticRegression(t *testing.T) {
	if lr := NewLogisticRegression(); lr == nil {
		t.Errorf("got nil linear regression")
	}
}
