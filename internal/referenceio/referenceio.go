package referenceio

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
)

const (
	VectorDimensions = 14
	fileMagic        = "R26REF01"
	fileVersion      = uint32(1)
)

type Record struct {
	Vector []float32 `json:"vector"`
	Label  string    `json:"label"`
}

func LoadJSONGZ(path string) ([]float32, []byte, int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("open references dataset: %w", err)
	}
	defer file.Close()

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("open gzip references dataset: %w", err)
	}
	defer gzReader.Close()

	decoder := json.NewDecoder(gzReader)

	token, err := decoder.Token()
	if err != nil {
		return nil, nil, 0, fmt.Errorf("read references opening token: %w", err)
	}
	if delim, ok := token.(json.Delim); !ok || delim != '[' {
		return nil, nil, 0, errors.New("references dataset must be a JSON array")
	}

	vectors := make([]float32, 0, 100000*VectorDimensions)
	labels := make([]byte, 0, 100000)
	index := 0

	for decoder.More() {
		var record Record
		if err := decoder.Decode(&record); err != nil {
			return nil, nil, 0, fmt.Errorf("decode reference record %d: %w", index, err)
		}
		if len(record.Vector) != VectorDimensions {
			return nil, nil, 0, fmt.Errorf("reference record %d has %d dimensions", index, len(record.Vector))
		}

		vectors = append(vectors, record.Vector...)
		if record.Label == "fraud" {
			labels = append(labels, 1)
		} else {
			labels = append(labels, 0)
		}
		index++
	}

	if _, err := decoder.Token(); err != nil {
		return nil, nil, 0, fmt.Errorf("read references closing token: %w", err)
	}

	return vectors, labels, index, nil
}

func WriteBinary(path string, vectors []float32, labels []byte, count int) error {
	if count < 0 {
		return errors.New("count must be non-negative")
	}
	if len(vectors) != count*VectorDimensions {
		return fmt.Errorf("invalid vectors length: got %d want %d", len(vectors), count*VectorDimensions)
	}
	if len(labels) != count {
		return fmt.Errorf("invalid labels length: got %d want %d", len(labels), count)
	}

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create references binary: %w", err)
	}
	defer file.Close()

	writer := io.Writer(file)
	if _, err := writer.Write([]byte(fileMagic)); err != nil {
		return fmt.Errorf("write magic: %w", err)
	}

	if err := binary.Write(writer, binary.LittleEndian, fileVersion); err != nil {
		return fmt.Errorf("write version: %w", err)
	}
	if err := binary.Write(writer, binary.LittleEndian, uint32(count)); err != nil {
		return fmt.Errorf("write count: %w", err)
	}
	if err := binary.Write(writer, binary.LittleEndian, vectors); err != nil {
		return fmt.Errorf("write vectors: %w", err)
	}
	if _, err := writer.Write(labels); err != nil {
		return fmt.Errorf("write labels: %w", err)
	}

	return nil
}

func LoadBinary(path string) ([]float32, []byte, int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("open references binary: %w", err)
	}
	defer file.Close()

	header := make([]byte, len(fileMagic))
	if _, err := io.ReadFull(file, header); err != nil {
		return nil, nil, 0, fmt.Errorf("read magic: %w", err)
	}
	if string(header) != fileMagic {
		return nil, nil, 0, fmt.Errorf("invalid references binary magic: %q", string(header))
	}

	var version uint32
	if err := binary.Read(file, binary.LittleEndian, &version); err != nil {
		return nil, nil, 0, fmt.Errorf("read version: %w", err)
	}
	if version != fileVersion {
		return nil, nil, 0, fmt.Errorf("unsupported references binary version: %d", version)
	}

	var count32 uint32
	if err := binary.Read(file, binary.LittleEndian, &count32); err != nil {
		return nil, nil, 0, fmt.Errorf("read count: %w", err)
	}
	count := int(count32)

	vectors := make([]float32, count*VectorDimensions)
	if err := binary.Read(file, binary.LittleEndian, vectors); err != nil {
		return nil, nil, 0, fmt.Errorf("read vectors: %w", err)
	}

	labels := make([]byte, count)
	if _, err := io.ReadFull(file, labels); err != nil {
		return nil, nil, 0, fmt.Errorf("read labels: %w", err)
	}

	return vectors, labels, count, nil
}
