package ml

import (
	"errors"
	"fmt"
	"math"
)

// Matrix is a type to make matrix operations.
type Matrix [][]float64

func (pm *Matrix) String() string {
	m := *pm
	var ret string
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[i]); j++ {
			ret += fmt.Sprintf("%4.2f\t", m[i][j])
		}
		ret += fmt.Sprintln()
	}
	return ret
}

// Transpose returns the transposed matrix.
func (pm *Matrix) Transpose() (Matrix, error) {
	m := *pm

	t := make([][]float64, len(m[0]))
	for i := 0; i < len(m[0]); i++ {
		t[i] = make([]float64, len(m))
	}

	for i := 0; i < len(t); i++ {
		for j := 0; j < len(t[0]); j++ {
			t[i][j] = m[j][i]
		}
	}
	return t, nil
}

// Scalar returns the scalar multiplication of the Matrix to l.
func (pm *Matrix) Scalar(l float64) (Matrix, error) {
	m := *pm

	r := make([][]float64, len(m))
	for i := 0; i < len(m[0]); i++ {
		r[i] = make([]float64, len(m[0]))
	}

	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[0]); j++ {
			r[i][j] = m[i][j] * l
		}
	}
	return r, nil
}

// Add returns the matrix addition between a and b.
func (pm *Matrix) Add(b Matrix) (Matrix, error) {
	a := *pm
	if len(a) != len(b) || len(a[0]) != len(b[0]) {
		return nil, fmt.Errorf("Add: both matrices must have same dimensions")
	}

	r := make([][]float64, len(a))
	for i := 0; i < len(a[0]); i++ {
		r[i] = make([]float64, len(a[0]))
	}

	for i := 0; i < len(a); i++ {
		for j := 0; j < len(a[0]); j++ {
			r[i][j] = a[i][j] + b[i][j]
		}
	}
	return r, nil
}

// Product returns the matrix product between a and b.
func (pm *Matrix) Product(b Matrix) (Matrix, error) {
	a := *pm

	if len(a[0]) != len(b) {
		return nil, fmt.Errorf("Product: number of cols in 'a' must be equal to the number of rows in 'b'")
	}
	p := make([][]float64, len(a))
	for i := 0; i < len(a); i++ {
		p[i] = make([]float64, len(b[0]))
	}

	for k := 0; k < len(b[0]); k++ {
		for i := 0; i < len(a); i++ {
			for j := 0; j < len(a[0]); j++ {
				p[i][k] += a[i][j] * b[j][k]
			}
		}
	}
	return p, nil
}

// Inverse returns the inverse matrix of the current Matrix if exists.
func (pm *Matrix) Inverse() (Matrix, error) {
	m := *pm
	n := len(m)
	if n != len(m[0]) {
		return nil, errors.New("Panic: matrix should be square")
	}
	x := make([][]float64, n) // inverse matrix to return
	for i := 0; i < n; i++ {
		x[i] = make([]float64, n)
	}
	LU, p, err := lupDecomposition(m)
	if err != nil {
		return nil, err
	}
	// Solve AX = e for each column ei of the identity matrix using LUP decomposition
	for i := 0; i < n; i++ {
		e := make([]float64, n)
		e[i] = 1
		solve := lupSolve(LU, p, e)
		for j := 0; j < len(solve); j++ {
			x[j][i] = solve[j]
		}
	}
	return x, nil
}

func lupSolve(LU Matrix, pi []int, b []float64) []float64 {
	n := len(LU)
	x := make([]float64, n)
	y := make([]float64, n)
	var suml, sumu, lij float64

	// solve for y using formward substitution
	for i := 0; i < n; i++ {
		suml = float64(0)
		for j := 0; j <= i-1; j++ {
			if i == j {
				lij = 1
			} else {
				lij = LU[i][j]
			}
			suml = suml + (lij * y[j])
		}
		y[i] = b[pi[i]] - suml
	}
	//Solve for x by using back substitution
	for i := n - 1; i >= 0; i-- {
		sumu = 0
		for j := i + 1; j < n; j++ {
			sumu = sumu + (LU[i][j] * x[j])
		}
		x[i] = (y[i] - sumu) / LU[i][i]
	}
	return x
}

// Perform LUP decomposition on a matrix A.
// Return L and U as a single matrix(double[][]) and P as an array of ints.
// We implement the code to compute LU "in place" in the matrix A.
// In order to make some of the calculations more straight forward and to
// match Cormen's et al. pseudocode the matrix A should have its first row and first columns
// to be all 0.
func lupDecomposition(A Matrix) (Matrix, []int, error) {

	n := len(A)
	// pi is the permutation matrix.
	// We implement it as an array whose value indicates which column the 1 would appear.
	//We use it to avoid dividing by zero or small numbers.
	pi := make([]int, n)
	var p float64
	var kp, pik, pikp int
	var aki, akpi float64

	for j := 0; j < n; j++ {
		pi[j] = j
	}

	for k := 0; k < n; k++ {
		p = 0
		for i := k; i < n; i++ {
			if math.Abs(A[i][k]) > p {
				p = math.Abs(A[i][k])
				kp = i
			}
		}
		if p == 0 {
			return nil, nil, errors.New("Panic: singular matrix")
		}

		pik = pi[k]
		pikp = pi[kp]
		pi[k] = pikp
		pi[kp] = pik

		for i := 0; i < n; i++ {
			aki = A[k][i]
			akpi = A[kp][i]
			A[k][i] = akpi
			A[kp][i] = aki
		}

		for i := k + 1; i < n; i++ {
			A[i][k] = A[i][k] / A[k][k]
			for j := k + 1; j < n; j++ {
				A[i][j] = A[i][j] - (A[i][k] * A[k][j])
			}
		}
	}
	return A, pi, nil
}

// Identity returns the identity Matrix with dimention n
func Identity(n int) (Matrix, error) {
	if n <= 0 {
		return nil, fmt.Errorf("Identity: dimention 'n' must be greater than '0'")
	}
	ID := make([][]float64, n)
	for i := 0; i < n; i++ {
		ID[i] = make([]float64, n)
		ID[i][i] = 1
	}
	return ID, nil
}
