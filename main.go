// Copyright 2020 The Clairvoyant Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/gob"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"time"

	"github.com/mjibson/go-dsp/fft"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tc128"
	"github.com/pointlander/gradient/tf32"
)

// Candles is a stock candle
type Candles struct {
	Close     []json.Number `json:"c"`
	High      []json.Number `json:"h"`
	Low       []json.Number `json:"l"`
	Open      []json.Number `json:"o"`
	Status    string        `json:"s"`
	Timestamp []json.Number `json:"t"`
	Volume    []json.Number `json:"v"`
}

// Symbol is a stock symbol
type Symbol struct {
	Currency    string  `json:"currency"`
	Description string  `json:"description"`
	Display     string  `json:"displaySymbol"`
	FIGI        *string `json:"figi"`
	ISIN        *string `json:"isin"`
	MIC         *string `json:"mic"`
	ShareClass  *string `json:"shareClassFIGI"`
	Symbol      *string `json:"symbol"`
	Symbol2     *string `json:"symbol2"`
	Type        *string `json:"type"`
	Prices      []float32
}

// Symbols is a list of stock symbols
type Symbols []Symbol

// APIKey is the api key for finnhub
var APIKey = os.Getenv("KEY")

// SphericalSoftmax is the spherical softmax function
// https://arxiv.org/abs/1511.05042
func SphericalSoftmax(k tc128.Continuation, node int, a *tc128.V, options ...map[string]interface{}) bool {
	const E = complex(0, 0)
	c, size, width := tc128.NewV(a.S...), len(a.X), a.S[0]
	values, sums, row := make([]complex128, width), make([]complex128, a.S[1]), 0
	for i := 0; i < size; i += width {
		sum := complex(0, 0)
		for j, ax := range a.X[i : i+width] {
			values[j] = ax*ax + E
			sum += values[j]
		}
		for _, cx := range values {
			c.X = append(c.X, (cx+E)/sum)
		}
		sums[row] = sum
		row++
	}
	if k(&c) {
		return true
	}
	// (2 a (b^2 + c^2 + d^2 + 0.003))/(a^2 + b^2 + c^2 + d^2 + 0.004)^2
	for i, d := range c.D {
		ax, sum := a.X[i], sums[i/width]
		//a.D[i] += d*(2*ax*(sum-(ax*ax+E)))/(sum*sum) - d*cx*2*ax/sum
		a.D[i] += d * (2 * ax * (sum - (ax*ax + E))) / (sum * sum)
	}
	return false
}

// Original is the original code
func Original() {
	symbols, prices := []string{"AAPL", "IBM", "CTVA", "K", "CAT", "GS", "T", "WMT"}, make([][]float32, 0, 8)
	for _, symbol := range symbols {
		stock := Prices(symbol)
		fmt.Println(symbol, len(stock))
		prices = append(prices, stock)
	}

	size := len(prices[0]) + 1
	set := tf32.NewSet()
	set.Add("w1", size, size)
	set.Add("b1", size)
	set.Add("values", size, len(symbols))

	w := set.ByName["b1"]
	factor := float32(math.Sqrt(float64(w.S[0])))
	for i := 0; i < cap(w.X); i++ {
		w.X = append(w.X, Random32(-1, 1)/factor)
	}

	w = set.ByName["w1"]
	w.X = w.X[:cap(w.X)]
	for i := 0; i < size; i++ {
		index, factor := i*size, float32(math.Sqrt(float64(i+1)))
		for j := 0; j <= i; j++ {
			w.X[index+j] = Random32(-1, 1) / factor
		}
	}

	values := set.ByName["values"]
	for _, stock := range prices {
		for _, price := range stock {
			values.X = append(values.X, price)
		}
		close := values.X[len(values.X)-1]
		values.X = append(values.X, close)
	}

	deltas := make([][]float32, 0, 3)
	for _, w := range set.Weights {
		deltas = append(deltas, make([]float32, len(w.X)))
	}

	l1 := tf32.Add(tf32.Mul(set.Get("w1"), set.Get("values")), set.Get("b1"))
	cost := tf32.Avg(tf32.Quadratic(l1, set.Get("values")))

	iterations := 100
	alpha, eta := float32(.3), float32(.05)
	points := make(plotter.XYs, 0, iterations)
	start := time.Now()
	for i := 0; i < iterations; i++ {
		set.Zero()

		total := tf32.Gradient(cost).X[0]

		norm := float32(0)
		w := set.ByName["w1"]
		for i := 0; i < size; i++ {
			index := i * size
			for j := 0; j <= i; j++ {
				d := w.D[index+j]
				norm += d * d
			}
		}
		w = set.ByName["b1"]
		for _, d := range w.D {
			norm += d * d
		}
		/*w = set.ByName["values"]
		for i := size - 1; i < len(w.D); i += size {
			d := w.D[i]
			norm += d * d
		}*/
		norm = float32(math.Sqrt(float64(norm)))
		if norm > 1 {
			scaling := 1 / norm
			w = set.ByName["w1"]
			for i := 0; i < size; i++ {
				index := i * size
				for j := 0; j <= i; j++ {
					index := index + j
					d := w.D[index]
					deltas[0][index] = alpha*deltas[0][index] - eta*d*scaling
					w.X[index] += deltas[0][index]
				}
			}
			w = set.ByName["b1"]
			for i, d := range w.D {
				deltas[1][i] = alpha*deltas[1][i] - eta*d*scaling
				w.X[i] += deltas[1][i]
			}
			w = set.ByName["values"]
			for i := size - 1; i < len(w.X); i += size {
				d := w.D[i]
				deltas[2][i] = alpha*deltas[2][i] - eta*d
				w.X[i] += deltas[2][i]
			}
		} else {
			w = set.ByName["w1"]
			for i := 0; i < size; i++ {
				index := i * size
				for j := 0; j <= i; j++ {
					index := index + j
					d := w.D[index]
					deltas[0][index] = alpha*deltas[0][index] - eta*d
					w.X[index] += deltas[0][index]
				}
			}
			w = set.ByName["b1"]
			for i, d := range w.D {
				deltas[1][i] = alpha*deltas[1][i] - eta*d
				w.X[i] += deltas[1][i]
			}
			w = set.ByName["values"]
			for i := size - 1; i < len(w.X); i += size {
				d := w.D[i]
				deltas[2][i] = alpha*deltas[2][i] - eta*d
				w.X[i] += deltas[2][i]
			}
		}

		fmt.Println(i, total, time.Now().Sub(start))
		start = time.Now()
		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		if total < .001 {
			fmt.Println("stopping...")
			break
		}
	}

	w, index := set.ByName["values"], 0
	for i := size - 1; i < len(w.X); i += size {
		fmt.Println(symbols[index], w.X[i-1], w.X[i])
		index++
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}
}

var (
	// Original is the original mode
	FlagOriginal = flag.Bool("original", false, "original mode")
	// FlagFetch fetch stock data
	FlagFetch = flag.Bool("fetch", false, "fetch stock data")
)

func main() {
	flag.Parse()
	rand.Seed(1)

	if *FlagOriginal {
		Original()
		return
	}

	if *FlagFetch {
		url := fmt.Sprintf("https://finnhub.io/api/v1/stock/symbol?exchange=US&token=%s", APIKey)
		client := http.Client{}
		request, err := http.NewRequest("GET", url, nil)
		if err != nil {
			panic(err)
		}

		resp, err := client.Do(request)
		if err != nil {
			panic(err)
		}
		defer resp.Body.Close()

		var result Symbols
		json.NewDecoder(resp.Body).Decode(&result)
		fmt.Println(len(result))
		length := float64(len(result))
		fmt.Println(length*length*16/(1024*1024*1024), "GB")
		sort.Slice(result, func(i, j int) bool {
			return *result[i].Symbol < *result[j].Symbol
		})
		fmt.Println(length*1.5/(60*60*24), "days")
		for i := range result {
			time.Sleep(1500 * time.Millisecond)
			result[i].Prices = Prices(*result[i].Symbol)
			fmt.Println(len(result[i].Prices), *result[i].Symbol)
		}
		output, err := os.Create("symbols.bin")
		if err != nil {
			panic(err)
		}
		defer output.Close()
		encoder := gob.NewEncoder(output)
		err = encoder.Encode(result)
		if err != nil {
			panic(err)
		}
		return
	}

	inputs, err := os.Open("symbols.bin")
	if err != nil {
		panic(err)
	}
	defer inputs.Close()
	decoder := gob.NewDecoder(inputs)
	var symbols Symbols
	err = decoder.Decode(&symbols)
	if err != nil {
		panic(err)
	}
	fmt.Println("symbols", len(symbols))

	type Stock struct {
		Symbol  string
		Name    string
		Entropy float64
		Phase   float64
		Prices  []float32
		FFT     []complex128
	}
	stocks := make([]Stock, 0, 8)
	for _, symbol := range symbols {
		stock := symbol.Prices
		if len(stock) < 251 {
			continue
		}
		if len(stock) == 252 {
			stock = stock[:251]
		}
		//fmt.Println(*symbol.Symbol, len(stock))
		input, max := make([]float64, len(stock)), 0.0
		for i, v := range stock {
			value := float64(v)
			if value > max {
				max = value
			}
			input[i] = value
		}
		for i := range input {
			input[i] /= float64(max)
		}
		output := fft.FFTReal(input)
		for i := range output {
			output[i] /= complex(float64(len(output)), 0)
		}
		stocks = append(stocks, Stock{
			Symbol: *symbol.Symbol,
			Name:   symbol.Description,
			Prices: stock,
			FFT:    output,
		})
	}

	width, length := len(stocks[0].Prices), len(stocks)
	others := tc128.NewSet()
	others.Add("input", width, 1)
	input := others.ByName["input"]
	input.X = input.X[:cap(input.X)]

	// Create the weight data matrix
	x := tc128.NewSet()
	x.Add("points", width, length)
	point := x.ByName["points"]
	for _, v := range stocks {
		point.X = append(point.X, v.FFT...)
	}

	// The neural network is the attention model from attention is all you need
	spherical := tc128.U(SphericalSoftmax)
	l1 := spherical(tc128.Mul(x.Get("points"), others.Get("input")))
	l2 := spherical(tc128.T(tc128.Mul(l1, tc128.T(x.Get("points")))))
	cost := tc128.Entropy(l2)

	for i := 0; i < len(stocks); i++ {
		// Load the input
		copy(input.X, stocks[i].FFT)
		// Calculate the l1 output of the neural network
		cost(func(a *tc128.V) bool {
			stocks[i].Entropy = cmplx.Abs(a.X[0])
			stocks[i].Phase = cmplx.Phase(a.X[0])
			return true
		})
	}
	sort.Slice(stocks, func(i, j int) bool {
		return stocks[i].Entropy > stocks[j].Entropy
	})
	for _, stock := range stocks {
		fmt.Printf("%5s % 11.7f % 11.7f % 11.7f %s\n", stock.Symbol, stock.Entropy, stock.Phase, stock.Prices[len(stock.Prices)-1]-stock.Prices[0], stock.Name)
	}
}

// Prices returns the prices of the stock symbol
func Prices(symbol string) []float32 {
	stop := time.Now()
	start := stop.AddDate(-1, 0, 0)
	url := fmt.Sprintf("https://finnhub.io/api/v1/stock/candle?symbol=%s&resolution=D&from=%d&to=%d&token=%s",
		symbol, start.Unix(), stop.Unix(), APIKey)
	client := http.Client{}
	request, err := http.NewRequest("GET", url, nil)
	if err != nil {
		fmt.Println(err)
	}

	resp, err := client.Do(request)
	if err != nil {
		fmt.Println(err)
	}
	defer resp.Body.Close()

	var result Candles
	json.NewDecoder(resp.Body).Decode(&result)
	values := make([]float32, 0, len(result.Close))
	for _, value := range result.Close {
		v, err := value.Float64()
		if err != nil {
			panic(err)
		}
		values = append(values, float32(v))
	}
	return values
}

// Random32 return a random float32
func Random32(a, b float32) float32 {
	return (b-a)*rand.Float32() + a
}
