// Copyright 2020 The Clairvoyant Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"os"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

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

// APIKey is the api key for finnhub
var APIKey = os.Getenv("KEY")

func main() {
	rand.Seed(1)

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
