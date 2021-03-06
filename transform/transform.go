package transform

import (
	"fmt"
	"math"
)

func Funcs2D() []func([]float64) ([]float64, error) {
	return []func([]float64) ([]float64, error){
		transform2D1,
		transform2D2,
		transform2D3,
		transform2D4,
		transform2D5,
		transform2D6,
		transform2D7,
	}
}

//    (1, x1, x2, |x1+ x2|)
func transform2D1(a []float64) ([]float64, error) {
	if len(a) != 3 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2 := a[1], a[2]
	b := make([]float64, 4)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = math.Abs(x1 + x2)
	return b, nil
}

//    (1, x1, x2, |x1 - x2|)
func transform2D2(a []float64) ([]float64, error) {
	if len(a) != 3 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2 := a[1], a[2]
	b := make([]float64, 4)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = math.Abs(x1 - x2)
	return b, nil
}

//    (1, x1, x2, x1 * x2)
func transform2D3(a []float64) ([]float64, error) {
	if len(a) != 3 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2 := a[1], a[2]
	b := make([]float64, 4)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x1 * x2
	return b, nil
}

//    (1, x1, x2, x1 * x2, , |x1+ x2|)
func transform2D4(a []float64) ([]float64, error) {
	if len(a) != 3 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2 := a[1], a[2]
	b := make([]float64, 5)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x1 * x2
	b[4] = math.Abs(x1 + x2)
	return b, nil
}

//    (1, x1, x2, x1 * x2, , |x1 - x2|)
func transform2D5(a []float64) ([]float64, error) {
	if len(a) != 3 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2 := a[1], a[2]
	b := make([]float64, 5)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x1 * x2
	b[4] = math.Abs(x1 - x2)
	return b, nil
}

//    (1, x1, x2, x1^2 + x2^2)
func transform2D6(a []float64) ([]float64, error) {
	if len(a) != 3 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2 := a[1], a[2]
	b := make([]float64, 4)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x1*x1 + x2*x2
	return b, nil
}

//    (1, x1, x2, x1 * x2, x1^2 + x2^2)
func transform2D7(a []float64) ([]float64, error) {
	if len(a) != 3 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2 := a[1], a[2]
	b := make([]float64, 5)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x1 * x2
	b[4] = x1*x1 + x2*x2
	return b, nil
}

func Funcs3D() []func([]float64) ([]float64, error) {
	return []func([]float64) ([]float64, error){
		transform3D1,
		transform3D2,
		transform3D3,
		transform3D4,
		transform3D5,
		transform3D6,
		transform3D7,
	}
}

//    (1, x1, x2, x3, |x1 + x2 + x3|)
func transform3D1(a []float64) ([]float64, error) {
	if len(a) != 4 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3 := a[1], a[2], a[3]
	b := make([]float64, 5)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = math.Abs(x1 + x2 + x3)
	return b, nil
}

//    (1, x1, x2, x3, |x1 - x2 - x3|)
func transform3D2(a []float64) ([]float64, error) {
	if len(a) != 4 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3 := a[1], a[2], a[3]
	b := make([]float64, 5)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = math.Abs(x1 - x2 - x3)
	return b, nil
}

//    (1, x1, x2, x3, x1 * x2 * x3)
func transform3D3(a []float64) ([]float64, error) {
	if len(a) != 4 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3 := a[1], a[2], a[3]
	b := make([]float64, 5)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x1 * x2 * x3
	return b, nil
}

//    (1, x1, x2, x3, x1 * x2 * x3, , |x1 + x2 + x3|)
func transform3D4(a []float64) ([]float64, error) {
	if len(a) != 4 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3 := a[1], a[2], a[3]
	b := make([]float64, 6)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x1 * x2 * x3
	b[5] = math.Abs(x1 + x2 + x3)
	return b, nil
}

//    (1, x1, x2, x3, x1 * x2 * x3, , |x1 - x2 - x3|)
func transform3D5(a []float64) ([]float64, error) {
	if len(a) != 4 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3 := a[1], a[2], a[3]
	b := make([]float64, 6)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x1 * x2 * x3
	b[5] = math.Abs(x1 - x2 - x3)
	return b, nil
}

//    (1, x1, x2, x3, x1^2 + x2^2 + x3^2)
func transform3D6(a []float64) ([]float64, error) {
	if len(a) != 4 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3 := a[1], a[2], a[3]
	b := make([]float64, 5)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x1*x1 + x2*x2 + x3*x3
	return b, nil
}

//    (1, x1, x2, x3, x1 * x2 * x3, x1^2 + x2^2 + x3^2)
func transform3D7(a []float64) ([]float64, error) {
	if len(a) != 4 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3 := a[1], a[2], a[3]
	b := make([]float64, 6)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x1 * x2 * x3
	b[5] = x1*x1 + x2*x2 + x3*x3
	return b, nil
}

func Funcs4D() []func([]float64) ([]float64, error) {
	return []func([]float64) ([]float64, error){
		transform4D1,
		transform4D2,
		transform4D3,
		transform4D4,
		transform4D5,
		transform4D6,
		transform4D7,
		transform4D8,
	}
}

//    (1, x1, x2, x3, |x1 + x2 + x3 + x4|)
func transform4D1(a []float64) ([]float64, error) {
	if len(a) != 5 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3, x4 := a[1], a[2], a[3], a[4]

	b := make([]float64, 6)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x4
	b[5] = math.Abs(x1 + x2 + x3 + x4)
	return b, nil
}

//    (1, x1, x2, x3, |x1 - x2 - x3 - x4|)
func transform4D2(a []float64) ([]float64, error) {
	if len(a) != 5 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3, x4 := a[1], a[2], a[3], a[4]

	b := make([]float64, 6)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x4

	b[5] = math.Abs(x1 - x2 - x3 - x4)
	return b, nil
}

//    (1, x1, x2, x3, x1 * x2 * x3 * x4)
func transform4D3(a []float64) ([]float64, error) {
	if len(a) != 5 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3, x4 := a[1], a[2], a[3], a[4]

	b := make([]float64, 6)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x4

	b[5] = x1 * x2 * x3 * x4
	return b, nil
}

//    (1, x1, x2, x3, x1 * x2 * x3 * x4, , |x1 + x2 + x3 + x4|)
func transform4D4(a []float64) ([]float64, error) {
	if len(a) != 5 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3, x4 := a[1], a[2], a[3], a[4]

	b := make([]float64, 7)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x4
	b[5] = x1 * x2 * x3 * x4

	b[6] = math.Abs(x1 + x2 + x3 + x4)
	return b, nil
}

//    (1, x1, x2, x3, x1 * x2 * x3, , |x1 - x2 - x3 - x4|)
func transform4D5(a []float64) ([]float64, error) {
	if len(a) != 5 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3, x4 := a[1], a[2], a[3], a[4]

	b := make([]float64, 7)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x4

	b[5] = x1 * x2 * x3 * x4
	b[6] = math.Abs(x1 - x2 - x3 - x4)
	return b, nil
}

//    (1, x1, x2, x3, x1^2 + x2^2 + x3^2 + x4^2)
func transform4D6(a []float64) ([]float64, error) {
	if len(a) != 5 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3, x4 := a[1], a[2], a[3], a[4]

	b := make([]float64, 6)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x4

	b[5] = x1*x1 + x2*x2 + x3*x3 + x4*x4
	return b, nil
}

//    (1, x1, x2, x3, x1 * x2 * x3 * x4, x1^2 + x2^2 + x3^2) +x4^2
func transform4D7(a []float64) ([]float64, error) {
	if len(a) != 5 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3, x4 := a[1], a[2], a[3], a[4]

	b := make([]float64, 7)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x4

	b[5] = x1 * x2 * x3 * x4
	b[6] = x1*x1 + x2*x2 + x3*x3 + x4*x4
	return b, nil
}

//    (1, x1, x2, x3, x1 * x2 * x3 * x4, , |x1 + x2 + x3 + x4|)
func transform4D8(a []float64) ([]float64, error) {
	if len(a) != 5 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3, x4 := a[1], a[2], a[3], a[4]

	b := make([]float64, 15)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x4
	b[5] = x1 * x2
	b[6] = x3 * x4
	b[7] = x1 * x3
	b[8] = x2 * x4
	b[9] = x1 * x2 * x3 * x4

	b[10] = math.Abs(x1 + x2)
	b[11] = math.Abs(x3 + x4)
	b[12] = math.Abs(x1 + x3)
	b[13] = math.Abs(x2 + x4)
	b[14] = math.Abs(x1 + x2 + x3 + x4)
	return b, nil
}

func Funcs5D() []func([]float64) ([]float64, error) {
	return []func([]float64) ([]float64, error){
		transform5D1,
		transform5D2,
		transform5D3,
		transform5D4,
		transform5D5,
		transform5D6,
		transform5D7,
		transform5D8,
	}
}

//    (1, x1, x2, x3, |x1 + x2 + x3 + x4|)
func transform5D1(a []float64) ([]float64, error) {
	if len(a) != 6 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3, x4, x5 := a[1], a[2], a[3], a[4], a[5]

	b := make([]float64, 7)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x4
	b[5] = x5

	b[6] = math.Abs(x1 + x2 + x3 + x4 + x5)
	return b, nil
}

//    (1, x1, x2, x3, |x1 - x2 - x3 - x4|)
func transform5D2(a []float64) ([]float64, error) {
	if len(a) != 6 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3, x4, x5 := a[1], a[2], a[3], a[4], a[5]

	b := make([]float64, 7)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x4
	b[5] = x5

	b[6] = math.Abs(x1 - x2 - x3 - x4 - x5)
	return b, nil
}

//    (1, x1, x2, x3, x1 * x2 * x3 * x4)
func transform5D3(a []float64) ([]float64, error) {
	if len(a) != 6 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3, x4, x5 := a[1], a[2], a[3], a[4], a[5]

	b := make([]float64, 7)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x4
	b[5] = x5

	b[6] = x1 * x2 * x3 * x4 * x5
	return b, nil
}

//    (1, x1, x2, x3, x1 * x2 * x3 * x4, , |x1 + x2 + x3 + x4|)
func transform5D4(a []float64) ([]float64, error) {
	if len(a) != 6 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3, x4, x5 := a[1], a[2], a[3], a[4], a[5]

	b := make([]float64, 8)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x4
	b[5] = x5

	b[6] = x1 * x2 * x3 * x4 * x5

	b[7] = math.Abs(x1 + x2 + x3 + x4 + x5)
	return b, nil
}

//    (1, x1, x2, x3, x1 * x2 * x3, , |x1 - x2 - x3 - x4|)
func transform5D5(a []float64) ([]float64, error) {
	if len(a) != 6 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3, x4, x5 := a[1], a[2], a[3], a[4], a[5]

	b := make([]float64, 8)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x4
	b[5] = x5

	b[6] = x1 * x2 * x3 * x4 * x5
	b[7] = math.Abs(x1 - x2 - x3 - x4 - x5)
	return b, nil
}

//    (1, x1, x2, x3, x1^2 + x2^2 + x3^2 + x4^2)
func transform5D6(a []float64) ([]float64, error) {
	if len(a) != 6 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3, x4, x5 := a[1], a[2], a[3], a[4], a[5]

	b := make([]float64, 7)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x4
	b[5] = x5

	b[6] = x1*x1 + x2*x2 + x3*x3 + x4*x4 + x5*x5
	return b, nil
}

//    (1, x1, x2, x3, x1 * x2 * x3 * x4, x1^2 + x2^2 + x3^2) +x4^2
func transform5D7(a []float64) ([]float64, error) {
	if len(a) != 6 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3, x4, x5 := a[1], a[2], a[3], a[4], a[5]

	b := make([]float64, 8)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x4
	b[5] = x5

	b[6] = x1 * x2 * x3 * x4 * x5
	b[7] = x1*x1 + x2*x2 + x3*x3 + x4*x4*x5*x5
	return b, nil
}

//    (1, x1, x2, x3, x1 * x2 * x3 * x4, , |x1 + x2 + x3 + x4|)
func transform5D8(a []float64) ([]float64, error) {
	if len(a) != 6 {
		return []float64{}, fmt.Errorf("error computing transform")
	}
	x1, x2, x3, x4, x5 := a[1], a[2], a[3], a[4], a[5]

	b := make([]float64, 27)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x4
	b[5] = x5

	b[5] = x1 * x2
	b[6] = x1 * x3
	b[7] = x1 * x4
	b[8] = x1 * x5
	b[9] = x2 * x3
	b[10] = x2 * x4
	b[11] = x2 * x5
	b[12] = x3 * x4
	b[13] = x3 * x5
	b[14] = x4 * x5

	b[15] = x1 * x2 * x3 * x4 * x5

	b[16] = math.Abs(x1 + x2)
	b[17] = math.Abs(x1 + x3)
	b[18] = math.Abs(x1 + x4)
	b[19] = math.Abs(x1 + x5)
	b[20] = math.Abs(x2 + x3)
	b[21] = math.Abs(x2 + x4)
	b[22] = math.Abs(x2 + x5)
	b[23] = math.Abs(x3 + x4)
	b[24] = math.Abs(x3 + x5)
	b[25] = math.Abs(x4 + x5)
	b[26] = math.Abs(x1 + x2 + x3 + x4 + x5)
	return b, nil
}
