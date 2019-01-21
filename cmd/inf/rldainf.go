package main

import (
	"flag"
	"fmt"
	"math/rand"

	"os"

	"bitbucket.org/sitfoxfly/ranklda/model"
)

func save(fn string, z [][]int, scores []float64, perplexity float64) {
	f, err := os.Create(fn)
	if err != nil {
		fmt.Println("Cannot save file!")
		os.Exit(1)
	}
	defer f.Close()

	fmt.Fprintf(f, "%d\n", len(z))
	for _, zi := range z {
		for j, zij := range zi {
			if j == 0 {
				fmt.Fprintf(f, "%d", zij)
			} else {
				fmt.Fprintf(f, " %d", zij)
			}
		}
		fmt.Fprintf(f, "\n")
	}

	for i, score := range scores {
		if i == 0 {
			fmt.Fprintf(f, "%f", score)
		} else {
			fmt.Fprintf(f, " %f", score)
		}
	}
	fmt.Fprintf(f, "\n%f\n", perplexity)
}

func main() {
	var seed int64
	settings := &model.InferSettings{}

	flag.Int64Var(&seed, "s", 1, "random seed")

	flag.Float64Var(&settings.InitT, "t", 1.0, "initial temperature")
	flag.IntVar(&settings.NumSAIter, "ti", 1000, "number of iterations for SA optimization")
	flag.Float64Var(&settings.CoolingRate, "tg", 1.0, "global cooling rate")
	var modelDataFn string
	flag.StringVar(&modelDataFn, "data", "", "model data file")
	flag.Parse()

	if flag.NArg() != 3 {
		fmt.Println("USAGE: rldainf <model> <data> <output>")
		flag.PrintDefaults()
		os.Exit(1)
	}

	modelfn := flag.Arg(0)
	datafn := flag.Arg(1)
	outfn := flag.Arg(2)

	rand.Seed(seed)

	var m *model.Model
	if modelDataFn == "" {
		m = model.ReadModel(modelfn)
	} else {
		m = model.ReadModelWithData(modelfn, modelDataFn)
	}
	data := model.Reduce(model.ReadData(datafn), m)

	z := m.Infer(data, settings)
	scores := m.Score(z)
	perplexity := m.Perplexity2(data.W)

	save(outfn, z, scores, perplexity)
}
