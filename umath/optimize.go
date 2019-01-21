package umath

import "math"

// NewtonRaphson is an instance of minimization problem with a special Hessian structure
type NewtonRaphson struct {
	Func        func(x []float64) float64
	Grad        func(grad []float64, x []float64)
	SpecialHess func(h []float64, z *float64, x []float64)
}

// FindStationaryPoint solves local minimization problem
func FindStationaryPoint(p NewtonRaphson, x0 []float64) []float64 {
	k := len(x0)
	x := make([]float64, k)
	copy(x, x0)

	z := 0.0
	h := make([]float64, k)
	g := make([]float64, k)

	prevEval := p.Func(x0)

	for iter := 0; iter < 10000; iter++ {
		p.Grad(g, x)
		p.SpecialHess(h, &z, x)

		c := 0.0
		d := 0.0
		for i := 0; i < k; i++ {
			c += g[i] / h[i]
			d += 1.0 / h[i]
		}
		c /= 1.0/z + d

		for i := 0; i < k; i++ {
			x[i] -= (g[i] - c) / h[i]
		}

		curEval := p.Func(x)
		if math.Abs(prevEval-curEval) < 1e-6 {
			break
		}
		prevEval = curEval
	}

	return x
}
