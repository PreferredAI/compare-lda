package umath

import (
	"log"
	"math"
	"math/rand"
)

// Lgamma returns the natural logarithm of Gamma(x).
func Lgamma(x float64) float64 {
	result, _ := math.Lgamma(x)
	return result
}

func _Sigmoid(x float64) float64 {
	return 0.5*(x/(1.0+math.Abs(x))) + 0.5
}

func _LogSigmoid(x float64) float64 {
	return math.Log(Sigmoid(x))
}

// Sigmoid returns 1/(1+Exp(x))
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// LogSigmoid returns the natural logarithm of Sigmoid(x).
func LogSigmoid(x float64) float64 {
	if x < -10 {
		return x - math.Exp(x)
	} else if x < 10 {
		return -math.Log(1.0 + math.Exp(-x))
	} else {
		return -math.Exp(-x)
	}
}

// H returns the difference: DiGamma(x + n) - DiGamma(x)
func H(x float64, n int) float64 {
	res := 0.0
	for i := 0; i < n; i++ {
		res += 1.0 / (x + float64(i))
	}
	return res
}

// DiH returns the derivative with respect to x of H(x)
func DiH(x float64, n int) float64 {
	result := 0.0
	for i := 0; i < n; i++ {
		val := x + float64(i)
		result -= 1.0 / (val * val)
	}
	return result
}

// Anxmany alpha * (x / nx - y / ny)
func Anxmany(a []float64, x, y []int, nx, ny int) float64 {
	length := len(a)
	if length != len(x) || length != len(y) {
		log.Panic("ERROR: vector length mismatch")
	}
	result := 0.0
	for i := 0; i < length; i++ {
		result += a[i] * (float64(x[i])/float64(nx) - float64(y[i])/float64(ny))
	}
	return result
}

func SampleFromLogDist(dist []float64) int {
	p := rand.Float64()
	cdf := 0.0
	for i, logProb := range dist {
		cdf += math.Exp(logProb)
		if p < cdf {
			return i
		}
	}
	return len(dist) - 1
}
