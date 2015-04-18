// Package ml provides a set of functions for machine learning.
package ml

// Sign returns float64 1 if number is > than 0 and -1 otherwise
func Sign(n float64) float64 {
	if n > float64(0) {
		return float64(1)
	}
	return float64(-1)
}
