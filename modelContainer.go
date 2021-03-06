package ml

import (
	"fmt"
	"sort"

	"github.com/santiaago/ml/data"
)

type Model interface {
	Learn() error
	Ein() float64
	Ecv() float64
}

type ModelContainer struct {
	Model              Model  // the model to use.
	Name               string // the name of the model.
	Features           []int  // the features to filter the data.
	TransformDimension int    // the dimensionality of the transform function if any.
	TransformID        int    // the ID of the transform function used.
}

func NewModelContainer(m Model, n string, features []int) *ModelContainer {
	return &ModelContainer{m, n, features, 0, 0}
}

type ModelContainers []*ModelContainer

func (slice ModelContainers) Len() int {
	return len(slice)
}

func (slice ModelContainers) Less(i, j int) bool {
	return (*slice[i]).Model.Ein() < (*slice[j]).Model.Ein()
}

func (slice ModelContainers) Swap(i, j int) {
	slice[i], slice[j] = slice[j], slice[i]
}

type ByEin ModelContainers

func (a ByEin) Len() int           { return len(a) }
func (a ByEin) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByEin) Less(i, j int) bool { return a[i].Model.Ein() < a[j].Model.Ein() }

type ByEcv ModelContainers

func (a ByEcv) Len() int           { return len(a) }
func (a ByEcv) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByEcv) Less(i, j int) bool { return a[i].Model.Ecv() < a[j].Model.Ecv() }

func (models ModelContainers) TopEin(n int) {
	sort.Sort(ByEin(models))
	for i := 0; i < n && i < len(models); i++ {
		if models[i] == nil {
			continue
		}
		m := models[i].Model
		fmt.Printf("EIn = %f \t%s\n", m.Ein(), models[i].Name)
	}
}

func (models ModelContainers) TopEcv(n int) {
	sort.Sort(ByEcv(models))
	for i := 0; i < n && i < len(models); i++ {
		if models[i] == nil {
			continue
		}
		m := models[i].Model
		fmt.Printf("Ecv = %f \t%s\n", m.Ecv(), models[i].Name)
	}
}

// ModelsFromFuncs returns an array of modelContainer types merged from
// the result of each function present in the 'funcs' array.
// Each of those functions takes as param a data Container and
// returns a modelContainer type.
func ModelsFromFuncs(dc data.Container, funcs []func(data.Container) (*ModelContainer, error)) (models ModelContainers) {

	for _, f := range funcs {
		if m, err := f(dc); err == nil {
			models = append(models, m)
		}
	}
	return
}
