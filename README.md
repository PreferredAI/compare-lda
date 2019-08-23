# CompareLDA: A Topic Model for Document Comparison

Comparative Latent Dirichlet Allocation (CompareLDA) learns predictive topic distributions that comply with the pairwise comparison observations.

### Installation

This is a GoLang project and can be assembled using standard [GoLang](https://golang.org) infrastructure:
* `cmd/eval/chibeval.go` is a Chib-style estimator of the predictive log-likelihood;
* `cmd/fit/rldafit.go` is a CompareLDA trainer;
* `cmd/inf/rldainf.go` is a CompareLDA predictor.
