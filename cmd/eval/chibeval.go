package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/gonum/floats"
)

type lda struct {
	alpha float64
	phi   [][]float64
}

func parseFloats(line string, isLog bool) []float64 {
	sc := bufio.NewScanner(strings.NewReader(line))
	parse := make([]float64, 0, 64)
	sc.Split(bufio.ScanWords)
	for sc.Scan() {
		if val, err := strconv.ParseFloat(sc.Text(), 64); err == nil {
			if isLog {
				val = math.Exp(val)
			}
			parse = append(parse, val)
		} else {
			log.Panic("ERROR: unable to parse floats", err)
		}
	}
	if isLog {
		floats.Scale(1/floats.Sum(parse), parse)
	}
	return parse
}

func readPhi(fn string, isLog bool) [][]float64 {
	f, err := os.Open(fn)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	phi := make([][]float64, 0, 10)
	sc := bufio.NewScanner(f)
	buf := make([]byte, 0, 64*1024)
	sc.Buffer(buf, 1024*1024)
	for sc.Scan() {
		phi = append(phi, parseFloats(sc.Text(), isLog))
	}
	if sc.Err() != nil {
		log.Panic("ERROR: unable to parse data file", sc.Err().Error())
	}
	return phi
}

func parseAndFilterInts(s string, v int) []int {
	sc := bufio.NewScanner(strings.NewReader(s))
	sc.Split(bufio.ScanWords)
	result := make([]int, 0, 10)
	for sc.Scan() {
		if val, err := strconv.Atoi(sc.Text()); err == nil {
			if val < v {
				result = append(result, val)
			}
		} else {
			log.Panic("ERROR: unable to parse data file", err)
		}
	}
	return result
}

func readDocs(fn string, v int) [][]int {
	f, err := os.Open(fn)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	docs := make([][]int, 0, 10)

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		docs = append(docs, parseAndFilterInts(sc.Text(), v))
	}

	return docs
}

func sample(p []float64) int {
	x := rand.Float64()
	cda := 0.0
	for i, pi := range p {
		cda += pi
		if x < cda {
			return i
		}
	}
	return len(p) - 1
}

func sampleGibbsForward(model *lda, doc []int, z []int, zn []int) {
	n := len(doc)
	k := len(zn)
	p := make([]float64, k)
	for i := 0; i < n; i++ {
		zn[z[i]]--
		for j := 0; j < k; j++ {
			p[j] = model.phi[j][doc[i]] * (float64(zn[j]) + model.alpha)
		}
		floats.Scale(1/floats.Sum(p), p)
		newZ := sample(p)
		zn[newZ]++
		z[i] = newZ
	}
}

func sampleGibbsBackward(model *lda, doc []int, z []int, zn []int) {
	n := len(doc)
	k := len(zn)
	p := make([]float64, k)
	for i := n - 1; i > -1; i-- {
		zn[z[i]]--
		for j := 0; j < k; j++ {
			p[j] = model.phi[j][doc[i]] * (float64(zn[j]) + model.alpha)
		}
		floats.Scale(1/floats.Sum(p), p)
		newZ := sample(p)
		zn[newZ]++
		z[i] = newZ
	}
}

func evalPwz(model *lda, doc []int, z []int) float64 {
	val := 0.0
	for i, w := range doc {
		val += math.Log2(model.phi[z[i]][w])
	}
	return val
}

func lgamma(x float64) float64 {
	y, _ := math.Lgamma(x)
	return y
}

func evalPz(model *lda, doc []int, zn []int) float64 {
	k := float64(len(zn))
	val := lgamma(model.alpha*k) - lgamma(model.alpha*k+float64(len(doc)))
	for _, zni := range zn {
		val += lgamma(float64(zni)+model.alpha) - lgamma(model.alpha)
	}
	return val / math.Log(2.0)
}

func evalTz(model *lda, doc []int, za, zb []int, zn []int) float64 {
	k := len(zn)
	n := len(doc)
	curZn := make([]int, k)
	copy(curZn, zn)

	val := 0.0
	p := make([]float64, k)
	for i := n - 1; i > -1; i-- {
		curZn[za[i]]--
		for j := 0; j < k; j++ {
			p[j] = model.phi[j][doc[i]] * (float64(curZn[j]) + model.alpha)
		}
		floats.Scale(1/floats.Sum(p), p)
		val += math.Log(p[zb[i]])
		curZn[zb[i]]++
	}
	return val
}

func evalOne(model *lda, doc []int, numSamples int) float64 {
	burnIn := 1000
	k := len(model.phi)
	n := len(doc)
	z := make([]int, n)
	zn := make([]int, k)

	for i := 0; i < n; i++ {
		z[i] = rand.Intn(k)
		zn[z[i]]++
	}

	// burn-in first to get a good sample
	for a := 0; a < burnIn; a++ {
		sampleGibbsForward(model, doc, z, zn)
	}

	samples := make([][]int, 0, numSamples)
	znIndex := make([][]int, 0, numSamples)
	for i := 0; i < numSamples; i++ {
		samples = append(samples, make([]int, n))
		znIndex = append(znIndex, make([]int, k))
	}
	x := rand.Intn(numSamples)
	copy(samples[x], z)
	copy(znIndex[x], zn)
	sampleGibbsBackward(model, doc, samples[x], znIndex[x])
	for i := x + 1; i < numSamples; i++ {
		copy(samples[i], samples[i-1])
		copy(znIndex[i], znIndex[i-1])
		sampleGibbsForward(model, doc, samples[i], znIndex[i])
	}

	for i := x - 1; i > -1; i-- {
		copy(samples[i], samples[i+1])
		copy(znIndex[i], znIndex[i+1])
		sampleGibbsBackward(model, doc, samples[i], znIndex[i])
	}

	tzs := make([]float64, 0, numSamples)
	for _, sample := range samples {
		tzs = append(tzs, evalTz(model, doc, z, sample, zn))
	}

	return evalPwz(model, doc, z) + evalPz(model, doc, zn) + math.Log2(float64(numSamples)) - floats.LogSumExp(tzs)/math.Log(2.0)
}

func eval(model *lda, docs [][]int, numSamples int) float64 {
	sum := 0.0
	for i, doc := range docs {
		eval := evalOne(model, doc, numSamples)
		log.Printf("chibeval(docs[%d]) = %f\n", i, eval)
		sum += eval
	}
	return sum
}

func smooth(phis [][]float64, smoothing float64) {
	for _, phi := range phis {
		floats.AddConst(smoothing, phi)
		floats.Scale(1.0/floats.Sum(phi), phi)
	}
}

func main() {
	var seed int64
	var numSamples int
	var phiFn string
	var alpha float64
	var dataFn string
	var isLog bool
	var smoothing float64

	flag.Int64Var(&seed, "seed", 1, "random seed")
	flag.Float64Var(&alpha, "alpha", 1.0, "alpha")
	flag.StringVar(&phiFn, "phi", "", "phi definition")
	flag.StringVar(&dataFn, "data", "", "held-out data file")
	flag.IntVar(&numSamples, "samples", 100, "num of samples")
	flag.BoolVar(&isLog, "log", false, "exp transformation required")
	flag.Float64Var(&smoothing, "smoothing", 0, "smoothing param")

	flag.Parse()

	rand.Seed(seed)

	phis := readPhi(phiFn, isLog)
	smooth(phis, smoothing)
	model := &lda{alpha, phis}
	docs := readDocs(dataFn, len(model.phi[0]))
	eval := eval(model, docs, numSamples)
	log.Printf("chibeval(docs[0:%d]) = %.2f\n", len(docs), eval)
	fmt.Printf("%.2f\n", eval)
}
