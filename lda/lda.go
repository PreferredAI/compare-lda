package lda

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"bitbucket.org/sitfoxfly/ranklda/ints"
)

func parseLine(s string) []ints.Pair {
	sc := bufio.NewScanner(strings.NewReader(s))
	sc.Split(bufio.ScanWords)
	sc.Scan()
	if n, err := strconv.Atoi(sc.Text()); err == nil {
		result := make([]ints.Pair, n)
		for i := 0; i < n; i++ {
			sc.Scan()
			p := ints.Pair{}
			fmt.Sscanf(sc.Text(), "%d:%d", &p.X, &p.Y)
			result[i] = p
		}
		return result
	}
	log.Fatal("ERROR: unable to parse LDA data line")
	return nil
}

func ReadLDA(fn string, n int) [][]ints.Pair {
	f, err := os.Open(fn)
	if err != nil {
		log.Fatal("ERROR: unable to read LDA model")
	}
	defer f.Close()
	result := make([][]ints.Pair, n)
	scanner := bufio.NewScanner(f)
	for i := 0; i < n; i++ {
		scanner.Scan()
		result[i] = parseLine(scanner.Text())
	}
	return result
}
