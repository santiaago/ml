package ml

import (
	"sort"
	"testing"
)

type myModel struct {
	score float64
}

func (m myModel) Ein() float64 {
	return m.score
}
func (m myModel) Learn() error {
	return nil
}

func TestModelContainer(t *testing.T) {
	m1 := NewModelContainer(&myModel{3}, "mymodel", []int{1})
	m2 := NewModelContainer(&myModel{2}, "mymodel", []int{1})
	m3 := NewModelContainer(&myModel{1}, "mymodel", []int{1})
	var models ModelContainers
	models = append(models, m1)
	models = append(models, m2)
	models = append(models, m3)
	sort.Sort(models)

	Eins := []float64{1, 2, 3}

	for i, want := range Eins {
		got := models[i].Model.Ein()
		if got != want {
			t.Errorf("got %v want %v", got, want)
		}
	}
}

func TestSortModelContainersByEin(t *testing.T) {
	m1 := NewModelContainer(&myModel{3}, "mymodel", []int{1})
	m2 := NewModelContainer(&myModel{2}, "mymodel", []int{1})
	m3 := NewModelContainer(&myModel{1}, "mymodel", []int{1})
	var models ModelContainers
	models = append(models, m1)
	models = append(models, m2)
	models = append(models, m3)
	sort.Sort(ByEin(models))

	Eins := []float64{1, 2, 3}

	for i, want := range Eins {
		got := models[i].Model.Ein()
		if got != want {
			t.Errorf("got %v want %v", got, want)
		}
	}
}
