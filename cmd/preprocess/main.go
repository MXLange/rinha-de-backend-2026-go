package main

import (
	"log"
	"os"
	"path/filepath"

	"github.com/MXLange/rinha-de-backend-2026-go/internal/referenceio"
)

func main() {
	inputPath := envOrDefault("REFERENCES_JSON_GZ", "references.json.gz")
	outputPath := envOrDefault("REFERENCES_BIN", filepath.Join("resources-bin", "references.bin"))

	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		log.Fatalf("create output directory: %v", err)
	}

	vectors, labels, count, err := referenceio.LoadJSONGZ(inputPath)
	if err != nil {
		log.Fatalf("load json.gz references: %v", err)
	}

	if err := referenceio.WriteBinary(outputPath, vectors, labels, count); err != nil {
		log.Fatalf("write references.bin: %v", err)
	}

	log.Printf("wrote %d references to %s", count, outputPath)
}

func envOrDefault(key, fallback string) string {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	return value
}
