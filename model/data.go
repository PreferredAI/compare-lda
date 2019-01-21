package model

import (
	"fmt"
	"log"
	"os"

	"bufio"

	"strings"

	"strconv"

	"bitbucket.org/sitfoxfly/ranklda/ints"
)

type Data struct {
	W [][]int
	C []ints.Pair
	V int
	N int
	M int
}

func parseInts(s string) []int {
	sc := bufio.NewScanner(strings.NewReader(s))
	sc.Split(bufio.ScanWords)
	result := make([]int, 0, 10)
	for sc.Scan() {
		if val, err := strconv.Atoi(sc.Text()); err == nil {
			result = append(result, val)
		} else {
			log.Fatal("Unable to parse data file", err)
		}
	}
	return result
}

func parseLine(s string) []ints.Pair {
	sc := bufio.NewScanner(strings.NewReader(s))
	sc.Split(bufio.ScanWords)
	var result []ints.Pair
	for sc.Scan() {
		p := ints.Pair{}
		fmt.Sscanf(sc.Text(), "%d:%d", &p.X, &p.Y)
		result = append(result, p)
	}
	if sc.Err() != nil {
		log.Fatal("Error", sc.Err())
	}
	return result
}

// ReadData reads the data file
func ReadData(fn string) *Data {
	f, err := os.Open(fn)
	if err != nil {
		log.Fatal("Unable to open data file", err)
	}
	defer f.Close()
	sc := bufio.NewScanner(bufio.NewReader(f))
	buf := make([]byte, 64*1024)
	sc.Buffer(buf, 1024*1024)
	sc.Scan()
	var n, m int
	fmt.Sscanf(sc.Text(), "%d %d", &n, &m)

	// reading documents

	docs := make([][]int, 0, n)
	vocabSize := 0
	for i := 0; i < n; i++ {
		sc.Scan()
		doc := parseLine(sc.Text())
		len := 0
		for _, cn := range doc {
			if vocabSize < cn.X {
				vocabSize = cn.X
			}
			len += cn.Y
		}
		w := make([]int, len)
		h := 0
		for _, cn := range doc {
			for j := 0; j < cn.Y; j++ {
				w[h] = cn.X
				h++
			}
		}
		docs = append(docs, w)
	}
	vocabSize++

	// reading comparisons

	comparisons := make([]ints.Pair, m)
	for i := 0; i < m; i++ {
		sc.Scan()
		fmt.Sscanf(sc.Text(), "%d %d", &comparisons[i].X, &comparisons[i].Y)
	}

	// assignment

	return &Data{docs, comparisons, vocabSize, len(docs), len(comparisons)}
}

func Reduce(data *Data, model *Model) *Data {
	data.V = model.data.V
	filtered := make([][]int, data.N)
	for i, ws := range data.W {
		filtered[i] = make([]int, 0, len(ws))
		for _, w := range ws {
			if w < model.data.V {
				filtered[i] = append(filtered[i], w)
			}
		}
	}
	data.W = filtered
	return data
}
