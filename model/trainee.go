package model

import (
	"log"
	"math"
	"math/rand"

	"bitbucket.org/sitfoxfly/ranklda/ints"
	"bitbucket.org/sitfoxfly/ranklda/umath"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize"
)

type coI struct {
	X    int
	etaX float64
	etaY float64
	eval float64
}

type trainableModel struct {
	*Model
	nIndex          [][]int
	cIndex          [][]int
	zIndex          []int
	comparisonIndex [][]*coI
}

func (model *trainableModel) logLikelihoodOfTopics() float64 {
	k := model.k
	betaSum := floats.Sum(model.beta)
	alphaSum := model.alpha * float64(model.data.V)
	result := 0.0
	for i := 0; i < model.data.N; i++ {
		n := len(model.data.W[i])
		result += umath.Lgamma(betaSum) - umath.Lgamma(float64(n)+betaSum)
		for j := 0; j < k; j++ {
			result += umath.Lgamma(model.beta[j]+float64(model.nIndex[i][j])) - umath.Lgamma(model.beta[j])
		}

		//for j := 0; j < m; j++ {
		//	res += mod.logPhis[mod.zs[i][j]][mod.data.Docs[i][j]]
		//}
	}

	result -= float64(model.k) * (math.Log(float64(model.data.V)) + umath.Lgamma(model.alpha) - umath.Lgamma(alphaSum))
	for k := 0; k < model.k; k++ {
		result -= umath.Lgamma(float64(model.zIndex[k]) + alphaSum)
		for w := 0; w < model.data.V; w++ {
			result += umath.Lgamma(float64(model.cIndex[k][w]) + model.alpha)
		}
	}

	return result
}

func (model *trainableModel) logLikelihood() float64 {
	k := model.k
	betaSum := floats.Sum(model.beta)
	alphaSum := model.alpha * float64(model.data.V)
	result := 0.0
	for i := 0; i < model.data.N; i++ {
		n := len(model.data.W[i])
		result += umath.Lgamma(betaSum) - umath.Lgamma(float64(n)+betaSum)
		for j := 0; j < k; j++ {
			result += umath.Lgamma(model.beta[j]+float64(model.nIndex[i][j])) - umath.Lgamma(model.beta[j])
		}

		//for j := 0; j < m; j++ {
		//	res += mod.logPhis[mod.zs[i][j]][mod.data.Docs[i][j]]
		//}
	}

	result -= float64(model.k) * (math.Log(float64(model.data.V)) + umath.Lgamma(model.alpha) - umath.Lgamma(alphaSum))
	for k := 0; k < model.k; k++ {
		result -= umath.Lgamma(float64(model.zIndex[k]) + alphaSum)
		for w := 0; w < model.data.V; w++ {
			result += umath.Lgamma(float64(model.cIndex[k][w]) + model.alpha)
		}
	}

	for _, comp := range model.data.C {
		xLength := len(model.data.W[comp.X])
		yLength := len(model.data.W[comp.Y])
		result += umath.LogSigmoid(umath.Anxmany(model.nu, model.nIndex[comp.X], model.nIndex[comp.Y], xLength, yLength))
	}

	for _, nui := range model.nu {
		result -= 0.5 * nui * nui / model.sigma
	}

	return result
}

func (model *trainableModel) InferDoc(doc []int, s *InferSettings) []int {
	n := len(doc)
	alphaSum := model.alpha * float64(model.data.V)
	z := make([]int, n)
	for i := 0; i < n; i++ {
		z[i] = rand.Intn(model.k)
	}
	nIndex := ints.Count(z, model.k)
	cIndex := make([][]int, model.k)
	for i := 0; i < model.k; i++ {
		cIndex[i] = make([]int, model.data.V)
	}
	for i, w := range doc {
		cIndex[z[i]][w]++
	}
	T := s.InitT
	for iter := 0; iter < s.NumSAIter; iter++ {
		for i := 0; i < n; i++ {
			curZ := z[i]
			newZ := rand.Intn(model.k)
			if curZ == newZ {
				continue
			}
			w := doc[i]
			diff := math.Log(model.beta[curZ]+float64(nIndex[curZ]-1)) -
				math.Log(model.beta[newZ]+float64(nIndex[newZ])) +
				math.Log(model.alpha+float64(model.cIndex[curZ][w]+cIndex[curZ][w]-1)) -
				math.Log(model.alpha+float64(model.cIndex[newZ][w]+cIndex[newZ][w])) +
				math.Log(float64(model.zIndex[newZ]+nIndex[newZ])+alphaSum) -
				math.Log(float64(model.zIndex[curZ]+nIndex[curZ]-1)+alphaSum)
			prob := rand.Float64()
			if diff <= 0.0 || prob < math.Exp(-diff/T) {
				z[i] = newZ
				nIndex[curZ]--
				nIndex[newZ]++
				cIndex[curZ][w]--
				cIndex[newZ][w]++
			}
		}
		T *= s.CoolingRate
	}
	return z
}

func (model *trainableModel) nuObjEval(nu []float64) float64 {
	result := 0.0
	for _, comp := range model.data.C {
		xLength := len(model.data.W[comp.X])
		yLength := len(model.data.W[comp.Y])
		// re-weighted
		result += umath.LogSigmoid(umath.Anxmany(nu, model.nIndex[comp.X], model.nIndex[comp.Y], xLength, yLength))
	}
	for _, nui := range nu {
		result -= 0.5 * nui * nui / model.sigma
		//result -= math.Abs(nui) / model.sigma
	}
	return -result
}

func (model *trainableModel) nuObjGrad(grad []float64, nu []float64) {
	k := len(nu)
	//for i, nui := range nu {
	for i, nui := range nu {
		grad[i] = -nui / model.sigma
		//grad[i] = -1.0 / model.sigma
	}
	for _, comp := range model.data.C {
		xLength := len(model.data.W[comp.X])
		yLength := len(model.data.W[comp.Y])
		// re-weighted
		sigmoid := umath.Sigmoid(-umath.Anxmany(nu, model.nIndex[comp.X], model.nIndex[comp.Y], xLength, yLength))

		for i := 0; i < k; i++ {
			grad[i] += (float64(model.nIndex[comp.X][i])/float64(xLength) - float64(model.nIndex[comp.Y][i])/float64(yLength)) * sigmoid
		}
	}
	floats.Scale(-1, grad)
}

func (model *trainableModel) betaObjEval(bs []float64) float64 {
	n := len(model.nIndex)
	k := len(bs)
	sumLGammaBeta := 0.0
	sumBeta := 0.0
	for _, b := range bs {
		sumLGammaBeta += umath.Lgamma(b)
		sumBeta += b
	}
	res := float64(n) * (umath.Lgamma(sumBeta) - sumLGammaBeta)
	for i := 0; i < n; i++ {
		for j := 0; j < k; j++ {
			res += umath.Lgamma(bs[j] + float64(model.nIndex[i][j]))
		}
		res -= umath.Lgamma(float64(len(model.data.W[i])) + sumBeta)
	}
	return -res /*/ float64(n)*/
}

func (model *trainableModel) betaObjGrad(grad []float64, bs []float64) {
	sumBeta := floats.Sum(bs)
	n := len(model.nIndex)
	k := len(bs)
	for i := 0; i < k; i++ {
		grad[i] = 0.0
	}

	for i := 0; i < n; i++ {
		for j := 0; j < k; j++ {
			grad[j] += umath.H(bs[j], model.nIndex[i][j]) - umath.H(sumBeta, len(model.data.W[i]))
		}
	}
	floats.Scale(-1 /*/float64(n)*/, grad)
}

func (model *trainableModel) betaObjSpecialHess(h []float64, z *float64, bs []float64) {
	sumBeta := floats.Sum(bs)
	n := len(model.nIndex)
	k := len(bs)

	*z = 0
	for i := 0; i < k; i++ {
		h[i] = 0
	}

	for i := 0; i < n; i++ {
		ni := len(model.data.W[i])
		*z -= umath.DiH(sumBeta, ni)
		for j := 0; j < k; j++ {
			h[j] += umath.DiH(bs[j], model.nIndex[i][j])
		}
	}

	floats.Scale(-1 /*/float64(n)*/, h)
	*z = -(*z) /*/ float64(n)*/
}

func (model *trainableModel) betaObjHess(hess mat64.MutableSymmetric, bs []float64) {
	sumBeta := floats.Sum(bs)
	n := len(model.nIndex)
	k := len(bs)
	for i := 0; i < k; i++ {
		for j := i; j < k; j++ {
			hess.SetSym(i, j, 0.0)
		}
	}

	for i := 0; i < n; i++ {
		ni := len(model.data.W[i])
		diHSum := umath.DiH(sumBeta, ni)
		for j := 0; j < k; j++ {
			hess.SetSym(j, j, hess.At(j, j)+umath.DiH(bs[j], model.nIndex[i][j])-diHSum)
			for m := j + 1; m < k; m++ {
				hess.SetSym(j, m, hess.At(j, m)-diHSum)
			}
		}
	}

	for i := 0; i < k; i++ {
		for j := i; j < k; j++ {
			hess.SetSym(i, j, -hess.At(i, j) /*/float64(n)*/)
		}
	}
}

func (model *trainableModel) zCurObjEval() float64 {
	n := len(model.z)
	alphaSum := model.alpha * float64(model.data.V)
	result := 0.0

	for i := 0; i < n; i++ {
		for j := 0; j < model.k; j++ {
			result += umath.Lgamma(model.beta[j] + float64(model.nIndex[i][j]))
		}
	}

	for i := 0; i < model.k; i++ {
		result -= umath.Lgamma(float64(model.zIndex[i]) + alphaSum)
		for j := 0; j < model.data.V; j++ {
			result += umath.Lgamma(float64(model.cIndex[i][j]) + model.alpha)
		}
	}

	for _, comp := range model.data.C {
		xLength := len(model.data.W[comp.X])
		yLength := len(model.data.W[comp.Y])
		result += umath.LogSigmoid(umath.Anxmany(model.nu, model.nIndex[comp.X], model.nIndex[comp.Y], xLength, yLength))
	}
	return result
}

func (model *trainableModel) zObjEval(z [][]int) float64 {
	n := len(z)
	res := 0.0
	nZ := make([][]int, n)
	for i := 0; i < n; i++ {
		zi := z[i]
		nZ[i] = ints.Count(zi, model.k)
		for j := 0; j < model.k; j++ {
			res += umath.Lgamma(model.beta[j] + float64(nZ[i][j]))
		}

		m := len(z[i])
		for j := 0; j < m; j++ {
			zij := zi[j]
			res += model.logPhi[zij][model.data.W[i][j]]
		}
	}

	for _, comp := range model.data.C {
		zX := len(model.data.W[comp.X])
		zY := len(model.data.W[comp.Y])
		res += umath.LogSigmoid(umath.Anxmany(model.nu, nZ[comp.X], nZ[comp.Y], zX, zY))
	}

	return res
}

func (model *trainableModel) optimizeZ(numIter int, initT, cRate, dropRate float64) {
	log.Printf("optimizing Obj(z) = %g\n", model.zCurObjEval())

	alphaSum := model.alpha * float64(model.data.V)

	for i := 0; i < model.data.N; i++ {
		n := len(model.data.W[i])
		T := initT
		//for k := 0; k < numIter; k++ {
		for j := 0; j < n; j++ {
			curZ := model.z[i][j]
			newZ := rand.Intn(model.k)
			if curZ == newZ {
				continue
			}
			w := model.data.W[i][j]
			eval := math.Log(model.beta[curZ]+float64(model.nIndex[i][curZ]-1)) -
				math.Log(model.beta[newZ]+float64(model.nIndex[i][newZ])) +
				math.Log(model.alpha+float64(model.cIndex[curZ][w]-1)) -
				math.Log(model.alpha+float64(model.cIndex[newZ][w])) +
				math.Log(float64(model.zIndex[newZ])+alphaSum) -
				math.Log(float64(model.zIndex[curZ]-1)+alphaSum)

			for _, entry := range model.comparisonIndex[i] {
				if rand.Float64() < dropRate {
					continue
				}
				var eta float64
				if entry.X == i {
					eta = entry.etaX
				} else {
					eta = entry.etaY
				}
				delta := float64(model.nu[newZ]-model.nu[curZ]) / eta
				eval += umath.LogSigmoid(entry.eval) - umath.LogSigmoid(entry.eval+delta)
			}

			prob := rand.Float64()
			if eval <= 0.0 || prob < math.Exp(-eval/T) {
				model.z[i][j] = newZ
				model.nIndex[i][curZ]--
				model.nIndex[i][newZ]++
				model.cIndex[curZ][w]--
				model.cIndex[newZ][w]++
				model.zIndex[curZ]--
				model.zIndex[newZ]++
				for _, entry := range model.comparisonIndex[i] {
					var eta float64
					if entry.X == i {
						eta = entry.etaX
					} else {
						eta = entry.etaY
					}
					entry.eval += float64(model.nu[newZ]-model.nu[curZ]) / float64(eta)
				}
			}
		}
		T *= cRate
		//}
	}
	log.Printf("           Obj(z) = %g\n", model.zCurObjEval())
}

func (model *trainableModel) optimizePhi() {
	log.Printf("optimizing Obj(phi) = ?\n")

	for i := 0; i < model.k; i++ {
		for j := 0; j < model.data.V; j++ {
			model.logPhi[i][j] = model.alpha
		}
	}

	for i := 0; i < model.data.N; i++ {
		n := len(model.data.W[i])
		for j := 0; j < n; j++ {
			model.logPhi[model.z[i][j]][model.data.W[i][j]] += 1.0
		}
	}

	for i := 0; i < model.k; i++ {
		z := math.Log(floats.Sum(model.logPhi[i]))
		for j := 0; j < model.data.V; j++ {
			model.logPhi[i][j] = math.Log(model.logPhi[i][j]) - z
		}
	}
	log.Printf("           Obj(phi) = done\n")
}

func (model *trainableModel) optimizeBeta() {
	x0 := make([]float64, model.k)
	for i := 0; i < model.k; i++ {
		x0[i] = 1e-6
	}
	log.Printf("opimizing Obj(beta) = %g\n", model.betaObjEval(x0))
	betaOptProblem := umath.NewtonRaphson{Func: model.betaObjEval, Grad: model.betaObjGrad, SpecialHess: model.betaObjSpecialHess}
	x := umath.FindStationaryPoint(betaOptProblem, x0)
	copy(model.beta, x)
	log.Printf("           Obj(beta) = %g\n", model.betaObjEval(model.beta))
}

func (model *trainableModel) optimizeNu() {
	nuOptProb := optimize.Problem{Func: model.nuObjEval, Grad: model.nuObjGrad}
	log.Printf("optimizing Obj(nu) = %g\n", nuOptProb.Func(model.nu))
	settings := optimize.DefaultSettings()
	result, err := optimize.Local(nuOptProb, model.nu, settings, &optimize.GradientDescent{})
	if err != nil {
		log.Println("WARNING:", err)
	}
	copy(model.nu, result.X)
	log.Printf("           Obj(nu) = %g\n", nuOptProb.Func(model.nu))
}
