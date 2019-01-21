package model

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path"
	"strconv"
	"strings"

	"bitbucket.org/sitfoxfly/ranklda/ints"
	"bitbucket.org/sitfoxfly/ranklda/umath"
	"github.com/gonum/floats"
)

// Model is a structure to represet RankLDA model
type Model struct {
	data   *Data
	k      int
	alpha  float64
	beta   []float64
	logPhi [][]float64
	z      [][]int
	nu     []float64
	sigma  float64
}

// InferSettings - inference settings
type InferSettings struct {
	NumSAIter   int
	InitT       float64
	CoolingRate float64
}

// OptSettings - optimization settings
type OptSettings struct {
	NumIter            int
	NumSAIter          int
	BetaOpt            bool
	BurnInIter         int
	InitT              float64
	LocalCRate         float64
	GlobalCRate        float64
	ComparisonDropRate float64
}

// InitSet initialized for the random model
type InitSet struct {
	Seed  int64
	K     int
	Sigma float64
	Alpha float64
	Beta  float64
}

func ReadModelWithData(fn1, fn2 string) *Model {
	m := ReadModel(fn1)
	m.data = ReadData(fn2)
	return m
}

func lreadInts(s string) []int {
	sc := bufio.NewScanner(strings.NewReader(s))
	sc.Split(bufio.ScanWords)
	result := make([]int, 0, 10)
	for sc.Scan() {
		if w, err := strconv.Atoi(sc.Text()); err == nil {
			result = append(result, w)
		} else {
			panic(err)
		}
	}
	return result
}

// ReadModel reads RankLDA model from the file
func ReadModel(fn string) *Model {
	f, err := os.Open(fn)
	if err != nil {
		log.Fatal("ERROR: unable to open file", err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)
	var vocabSize int
	model := &Model{}
	scanner.Scan()
	fmt.Sscanf(scanner.Text(), "%d %d", &model.k, &vocabSize)
	model.beta = make([]float64, model.k)
	{
		scanner.Scan()
		lineReader := strings.NewReader(scanner.Text())
		for i := 0; i < model.k; i++ {
			fmt.Fscanf(lineReader, "%f", &model.beta[i])
		}
	}
	scanner.Scan()
	fmt.Sscanf(scanner.Text(), "%f", &model.alpha)
	model.logPhi = make([][]float64, model.k)
	for i := 0; i < model.k; i++ {
		scanner.Scan()
		lineReader := strings.NewReader(scanner.Text())
		model.logPhi[i] = make([]float64, vocabSize)
		for j := 0; j < vocabSize; j++ {
			fmt.Fscanf(lineReader, "%f", &model.logPhi[i][j])
		}
	}
	{
		scanner.Scan()
		lineReader := strings.NewReader(scanner.Text())
		model.nu = make([]float64, model.k)
		for i := 0; i < model.k; i++ {
			fmt.Fscanf(lineReader, "%f", &model.nu[i])
		}
	}
	var n int
	scanner.Scan()
	fmt.Sscanf(scanner.Text(), "%d", &n)
	model.z = make([][]int, n)
	for i := 0; i < n; i++ {
		scanner.Scan()
		model.z[i] = lreadInts(scanner.Text())
	}
	return model
}

// Infer infers topic assigment for the unseen data
func (model *Model) Infer(data *Data, s *InferSettings) [][]int {
	n := len(data.W)
	z := make([][]int, 0, n)
	trainee := model.trainable()
	for _, doc := range data.W {
		zi := trainee.InferDoc(doc, s)
		z = append(z, zi)
	}
	return z
}

func (model *Model) scoreDoc(w []int, z []int) float64 {
	n := len(w)
	zInd := ints.Count(z, model.k)
	betaSum := floats.Sum(model.beta)
	res := 0.0

	res += umath.Lgamma(betaSum) - umath.Lgamma(float64(n)+betaSum)
	for i := 0; i < model.k; i++ {
		res += umath.Lgamma(model.beta[i]+float64(zInd[i])) - umath.Lgamma(model.beta[i])
	}

	for i := 0; i < n; i++ {
		res += model.logPhi[z[i]][w[i]]
	}

	return res
}

func (model *Model) scoreDoc2(w []int, z []int) float64 {
	n := len(w)
	//zInd := ints.Count(z, mod.k)
	//betaSum := floats.Sum(mod.bs)
	res := 0.0

	//res += umath.Lgamma(betaSum) - umath.Lgamma(float64(n)+betaSum)
	//for i := 0; i < mod.k; i++ {
	//	res += umath.Lgamma(mod.bs[i]+float64(zInd[i])) - umath.Lgamma(mod.bs[i])
	//}

	for i := 0; i < n; i++ {
		res += model.logPhi[z[i]][w[i]]
	}

	return res
}

func (model *Model) inferDoc(doc []int, s *InferSettings) []int {
	n := len(doc)
	z := make([]int, n)
	for i := 0; i < n; i++ {
		z[i] = rand.Intn(model.k)
	}
	zInd := ints.Count(z, model.k)
	T := s.InitT
	//fmt.Printf("OBJ = %g\n", mod.scoreDoc(doc, z))
	for iter := 0; iter < s.NumSAIter; iter++ {
		for i := 0; i < n; i++ {
			curZ := z[i]
			newZ := rand.Intn(model.k)
			w := doc[i]
			diff := math.Log(model.beta[curZ]+float64(zInd[curZ]-1)) - math.Log(model.beta[newZ]+float64(zInd[newZ])) + model.logPhi[curZ][w] - model.logPhi[newZ][w]
			prob := rand.Float64()
			if diff <= 0.0 || prob < math.Exp(-diff/T) {
				z[i] = newZ
				zInd[curZ]--
				zInd[newZ]++
			}
		}
		T *= s.CoolingRate
	}
	//fmt.Printf("OBJ = %g\n", mod.scoreDoc(doc, z))
	//fmt.Println("=== DONE ===")
	return z
}

// Perplexity computes point estimate perplexity on a set of documents
func (model *Model) Perplexity(docs [][]int, z [][]int) float64 {
	logProb := 0.0
	normalizer := 0
	trainee := model.trainable()
	trainee.optimizePhi()
	for i, doc := range docs {
		logProb += trainee.scoreDoc(doc, z[i])
		normalizer += len(doc)
	}
	return logProb / float64(normalizer)
}

func (model *Model) Perplexity2(docs [][]int) float64 {
	s := 10
	logProb := 0.0
	normalizer := 0
	trainee := model.trainable()
	trainee.optimizePhi()
	for _, doc := range docs {
		z := make([]int, len(doc))
		cumLogProb := 0.0
		for j := 0; j < s; j++ {
			for k := 0; k < len(doc); k++ {
				z[k] = rand.Intn(model.k)
			}
			cumLogProb += trainee.scoreDoc2(doc, z)
		}
		logProb += cumLogProb / float64(s)
		normalizer += len(doc)
	}
	return logProb / float64(normalizer)
}

// Score computes the doc scores and builds pairwise comparison list
func (model *Model) Score(z [][]int) []float64 {
	scores := make([]float64, 0)
	for _, zi := range z {
		//fmt.Println(mod.nu, ints.Dist(ints.Count(zi, mod.k)), floats.Dot(mod.nu, ints.Dist(ints.Count(zi, mod.k))))
		scores = append(scores, floats.Dot(model.nu, ints.Dist(ints.Count(zi, model.k))))
	}
	return scores
}

// RandomModel builds a randomly initialized model for the data supplied
func RandomModel(data *Data, init *InitSet) *Model {
	model := &Model{}
	model.k = init.K
	model.sigma = init.Sigma
	model.alpha = init.Alpha
	model.data = data
	model.z = make([][]int, len(data.W))
	for i, doc := range data.W {
		n := len(doc)
		model.z[i] = make([]int, n)
		for j := 0; j < n; j++ {
			model.z[i][j] = rand.Intn(model.k)
		}
	}

	model.beta = make([]float64, model.k)
	model.nu = make([]float64, model.k)
	model.logPhi = make([][]float64, model.k)
	for i := 0; i < model.k; i++ {
		model.beta[i] = init.Beta
		model.nu[i] = rand.NormFloat64() * model.sigma
		model.logPhi[i] = make([]float64, data.V)
		for j := 0; j < data.V; j++ {
			model.logPhi[i][j] = -math.Log(float64(data.V))
		}
	}
	return model
}

// AssignedModel builds a model with initialized Zs
func AssignedModel(data *Data, init *InitSet, assignments [][]ints.Pair) *Model {
	model := &Model{}
	model.k = init.K
	model.sigma = init.Sigma
	model.alpha = init.Alpha
	model.data = data
	model.z = make([][]int, len(data.W))
	for i, v := range data.W {
		docAssign := assignments[i]
		assignMap := make(map[int]int)
		for j := 0; j < len(docAssign); j++ {
			assignMap[docAssign[j].X] = docAssign[j].Y
		}
		n := len(v)
		model.z[i] = make([]int, n)
		for j := 0; j < n; j++ {
			model.z[i][j] = assignMap[v[j]]
		}
	}

	model.beta = make([]float64, model.k)
	model.nu = make([]float64, model.k)
	model.logPhi = make([][]float64, model.k)
	for i := 0; i < model.k; i++ {
		model.beta[i] = init.Beta
		model.nu[i] = rand.NormFloat64() * model.sigma
		model.logPhi[i] = make([]float64, data.V)
		for j := 0; j < data.V; j++ {
			model.logPhi[i][j] = -math.Log(float64(data.V))
		}
	}
	return model
}

// Optimize optimizes the RankLDA model with Variational Approximation Algorithm
func (model *Model) Optimize(s *OptSettings, dir string) {

	var lhLog *os.File
	if dir != "" {
		if lhLog, err := os.Open(path.Join(dir, "likelihood.txt")); err != nil {
			defer lhLog.Close()
		}
	}

	if s.BurnInIter > 0 {
		plainTrainable := model.plainTrainable()
		log.Printf("likelihood(topics) = %f\n", plainTrainable.logLikelihoodOfTopics())
		for i := 0; i < s.BurnInIter; i++ {
			plainTrainable.optimizeZ(s.NumSAIter, 1.0, s.LocalCRate, s.ComparisonDropRate)
			plainTrainable.optimizePhi()
			log.Printf("likelihood(topics) = %f\n", plainTrainable.logLikelihoodOfTopics())
		}
	}

	trainable := model.trainable()
	T := s.InitT
	for i := 0; i < s.NumIter; i++ {
		log.Printf("starting new iteration: %d (T = %g)\n", i, T)
		trainable.optimizeNu()
		trainable.optimizeZ(s.NumSAIter, T, s.LocalCRate, s.ComparisonDropRate)
		trainable.optimizePhi()
		if s.BetaOpt {
			trainable.optimizeBeta()
		}
		likelihood := trainable.logLikelihood()
		log.Printf("likelihood = %f\n", likelihood)
		if lhLog != nil {
			lhLog.WriteString(fmt.Sprintln(likelihood))
		}
		if dir != "" {
			trainable.Save(path.Join(dir, fmt.Sprintf("%02d-model.txt", i)))
		}
		T *= s.GlobalCRate
	}
	trainable.optimizeNu()
}

func (model *Model) plainTrainable() *trainableModel {
	nIndex := make([][]int, model.data.N)
	cIndex := make([][]int, model.k)
	for i := 0; i < model.k; i++ {
		cIndex[i] = make([]int, model.data.V)
	}
	zIndex := make([]int, model.k)
	for i, row := range model.z {
		nIndex[i] = make([]int, model.k)
		for j, z := range row {
			w := model.data.W[i][j]
			nIndex[i][z]++
			cIndex[z][w]++
			zIndex[z]++
		}
	}

	comparisonIndex := make([][]*coI, model.data.N)
	for i := 0; i < model.data.N; i++ {
		comparisonIndex[i] = make([]*coI, 0)
	}

	return &trainableModel{model, nIndex, cIndex, zIndex, comparisonIndex}
}

func (model *Model) trainable() *trainableModel {
	nIndex := make([][]int, model.data.N)
	cIndex := make([][]int, model.k)
	for i := 0; i < model.k; i++ {
		cIndex[i] = make([]int, model.data.V)
	}
	zIndex := make([]int, model.k)
	for i, row := range model.z {
		nIndex[i] = make([]int, model.k)
		for j, z := range row {
			w := model.data.W[i][j]
			nIndex[i][z]++
			cIndex[z][w]++
			zIndex[z]++
		}
	}

	comparisonIndex := make([][]*coI, model.data.N)
	for i := 0; i < model.data.N; i++ {
		comparisonIndex[i] = make([]*coI, 0, 10)
	}
	for _, comp := range model.data.C {
		xLength := len(model.data.W[comp.X])
		yLength := len(model.data.W[comp.Y])
		ref := &coI{comp.X, float64(xLength), -float64(yLength), umath.Anxmany(model.nu, nIndex[comp.X], nIndex[comp.Y], xLength, yLength)}
		comparisonIndex[comp.X] = append(comparisonIndex[comp.X], ref)
		comparisonIndex[comp.Y] = append(comparisonIndex[comp.Y], ref)
	}

	return &trainableModel{model, nIndex, cIndex, zIndex, comparisonIndex}
}

func (model *Model) Save(fn string) {
	f, err := os.Create(fn)
	if err != nil {
		log.Fatal("ERROR: unable to save model file:", fn)
	}
	defer f.Close()

	fmt.Fprintf(f, "%d %d\n", model.k, model.data.V)
	for _, b := range model.beta {
		fmt.Fprintf(f, "%f ", b)
	}
	fmt.Fprintln(f)
	fmt.Fprintf(f, "%f\n", model.alpha)
	for _, row := range model.logPhi {
		for _, logPhi := range row {
			fmt.Fprintf(f, "%f ", logPhi)
		}
		fmt.Fprintln(f)
	}
	for _, w := range model.nu {
		fmt.Fprintf(f, "%f ", w)
	}
	fmt.Fprintln(f)
	fmt.Fprintf(f, "%d\n", len(model.z))
	for _, row := range model.z {
		for _, z := range row {
			fmt.Fprintf(f, "%d ", z)
		}
		fmt.Fprintln(f)
	}
}

func (model *Model) SaveLDA(fn string) {
	f, err := os.Create(fn)
	if err != nil {
		log.Fatal("ERROR: unbale to save LDA model", err)
	}
	defer f.Close()

	for i := 0; i < model.k; i++ {
		for j := 0; j < model.data.V; j++ {
			fmt.Fprintf(f, "%.10f ", math.Exp(model.logPhi[i][j]))
		}
		fmt.Fprintln(f)
	}
}
