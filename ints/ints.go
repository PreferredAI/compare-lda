package ints

import (
	"github.com/gonum/floats"
)

// Pair is a pair of int elements (x, y)
type Pair struct {
	X int
	Y int
}

// Sum returns sum of ints
func Sum(x []int) (sum int) {
	if x == nil {
		panic("invalid argument to Sum")
	}
	for _, v := range x {
		sum += v
	}
	return
}

// Max1D returns the maximum int in the array
func Max1D(x []int) (max int) {
	if x == nil || len(x) == 0 {
		panic("invalid argument to Max")
	}
	max = x[0]
	for _, v := range x[1:] {
		if max < v {
			max = v
		}
	}
	return
}

// Max2D returns the maximum int in the 2D array
func Max2D(x [][]int) (max int) {
	if x == nil || len(x) == 0 {
		panic("invalid argument to MaxMat")
	}
	max = Max1D(x[0])
	for _, row := range x[1:] {
		v := Max1D(row)
		if max < v {
			max = v
		}
	}
	return
}

// Count creates the count slice of elemtns in src
func Count(src []int, n int) []int {
	if src == nil {
		panic("invalid argument to Count")
	}
	dst := make([]int, n)
	for _, x := range src {
		dst[x]++
	}
	return dst
}

// Dist converts counts to the density values
func Dist(src []int) []float64 {
	if src == nil {
		panic("invalid argument to Dist")
	}
	res := make([]float64, 0, len(src))
	sum := 0.0
	for _, x := range src {
		fx := float64(x)
		res = append(res, fx)
		sum += fx
	}
	floats.Scale(1.0/sum, res)
	return res
}
