package main

import (
	"flag"
	"log"
	"math/rand"
	"os"

	"bitbucket.org/sitfoxfly/ranklda/lda"
	"bitbucket.org/sitfoxfly/ranklda/model"
)

func ensureDir(dir string) {
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		os.MkdirAll(dir, os.ModePerm)
	}
}

func ensureFile(fn string) {
	if fn != "" {
		if f, err := os.Create(fn); err == nil {
			f.Close()
		} else {
			log.Panic("ERROR: unable to create model file", fn)
		}
	}
}

func ensureCondition(condition bool) {
	if !condition {
		flag.PrintDefaults()
		os.Exit(1)
	}
}

func main() {
	init := &model.InitSet{}
	settings := &model.OptSettings{}
	flag.IntVar(&init.K, "k", 5, "number of topics")
	flag.Float64Var(&init.Sigma, "g", 1.0, "Gaussian regularization")
	flag.Int64Var(&init.Seed, "s", 1, "random seed")
	flag.Float64Var(&init.Alpha, "a", 1e-6, "phi pseudocounts")
	flag.Float64Var(&init.Beta, "b", 0.1, "symmetrical beta prior")

	flag.BoolVar(&settings.BetaOpt, "o", false, "optimize betas")
	flag.IntVar(&settings.NumIter, "i", 15, "number of iterations")
	flag.IntVar(&settings.BurnInIter, "bi", 0, "burn-in iterations")
	flag.Float64Var(&settings.InitT, "t", 1.0, "initial temperature")
	flag.IntVar(&settings.NumSAIter, "ti", 50, "number of iterations for SA optimization")
	flag.Float64Var(&settings.GlobalCRate, "tg", 1, "global cooling rate")
	flag.Float64Var(&settings.LocalCRate, "tl", 1, "local cooling rate")
	flag.Float64Var(&settings.ComparisonDropRate, "dr", 0.0, "comparison drop rate")

	var seedfn string
	var datafn string
	var modeldir string
	var modelfn string
	var ldaOutFn string

	flag.StringVar(&seedfn, "assign", "", "Zs seed initializer")
	flag.StringVar(&datafn, "data", "", "data file")
	flag.StringVar(&modeldir, "model-dir", "", "model directory")
	flag.StringVar(&modelfn, "model", "", "final model")
	flag.StringVar(&ldaOutFn, "lda-output", "", "vanilla LDA model")
	flag.Parse()

	ensureCondition(datafn != "")
	ensureCondition(modelfn != "")

	rand.Seed(init.Seed)

	if modelfn != "" {
		ensureFile(modelfn)
	}

	if modeldir != "" {
		ensureDir(modeldir)
	}

	data := model.ReadData(datafn)

	var m *model.Model
	if seedfn == "" {
		m = model.RandomModel(data, init)
	} else {
		assignments := lda.ReadLDA(seedfn, data.N)
		m = model.AssignedModel(data, init, assignments)
	}

	m.Optimize(settings, modeldir)

	if modelfn != "" {
		m.Save(modelfn)
	}

	if ldaOutFn != "" {
		m.SaveLDA(ldaOutFn)
	}

}
