package main

import (
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"

	"github.com/MXLange/rinha-de-backend-2026-go/internal/referenceio"
	"github.com/coder/hnsw"
)

func main() {
	inputPath := envOrDefault("REFERENCES_JSON_GZ", "references.json.gz")
	outputPath := envOrDefault("REFERENCES_BIN", filepath.Join("resources-bin", "references.bin"))
	graphPath := envOrDefault("HNSW_BIN", filepath.Join("resources-bin", "hnsw.bin"))
	hnswM := envIntOrDefault("HNSW_M", 16)
	hnswEfSearch := envIntOrDefault("HNSW_EF_SEARCH", 2)

	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		log.Fatalf("create output directory: %v", err)
	}

	vectors, labels, count, err := referenceio.LoadJSONGZ(inputPath)
	if err != nil {
		log.Fatalf("load json.gz references: %v", err)
	}

	vectors, labels, groups := referenceio.ReorderByBooleanGroups(vectors, labels, count)

	if err := referenceio.WriteBinaryWithGroups(outputPath, vectors, labels, count, groups); err != nil {
		log.Fatalf("write references.bin: %v", err)
	}

	graph := hnsw.NewGraph[int]()
	graph.Distance = hnsw.CosineDistance
	graph.M = hnswM
	graph.EfSearch = hnswEfSearch
	graph.Rng = rand.New(rand.NewSource(1))

	nodes := make([]hnsw.Node[int], 0, count)
	for i := 0; i < count; i++ {
		offset := i * referenceio.VectorDimensions
		nodes = append(nodes, hnsw.MakeNode(i, normalizeL2(vectors[offset:offset+referenceio.VectorDimensions])))
	}
	graph.Add(nodes...)

	graphFile, err := os.Create(graphPath)
	if err != nil {
		log.Fatalf("create hnsw.bin: %v", err)
	}
	defer graphFile.Close()

	if err := graph.Export(graphFile); err != nil {
		log.Fatalf("write hnsw.bin: %v", err)
	}

	log.Printf("wrote %d references to %s and HNSW index to %s", count, outputPath, graphPath)
}

func envOrDefault(key, fallback string) string {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	return value
}

func envIntOrDefault(key string, fallback int) int {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return fallback
	}
	return parsed
}

func normalizeL2(vector []float32) []float32 {
	var squaredNorm float32
	for i := 0; i < len(vector); i++ {
		squaredNorm += vector[i] * vector[i]
	}

	if squaredNorm <= 1e-12 {
		out := make([]float32, len(vector))
		copy(out, vector)
		return out
	}

	invNorm := float32(1.0 / math.Sqrt(float64(squaredNorm)))
	out := make([]float32, len(vector))
	for i := 0; i < len(vector); i++ {
		out[i] = vector[i] * invNorm
	}
	return out
}
