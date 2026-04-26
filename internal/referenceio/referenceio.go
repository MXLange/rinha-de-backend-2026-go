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
	fileVersion      = uint32(2)
)

type Record struct {
	Vector []float32 `json:"vector"`
	Label  string    `json:"label"`
}

type GroupRange struct {
	Start int
	Count int
}

type GroupRanges struct {
	All GroupRange
	TT  GroupRange
	TF  GroupRange
	FT  GroupRange
	FF  GroupRange
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
	return WriteBinaryWithGroups(path, vectors, labels, count, deriveGroupRanges(vectors, count))
}

func WriteBinaryWithGroups(path string, vectors []float32, labels []byte, count int, groups GroupRanges) error {
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
	for _, group := range []GroupRange{groups.TT, groups.TF, groups.FT, groups.FF} {
		if err := binary.Write(writer, binary.LittleEndian, uint32(group.Start)); err != nil {
			return fmt.Errorf("write group start: %w", err)
		}
		if err := binary.Write(writer, binary.LittleEndian, uint32(group.Count)); err != nil {
			return fmt.Errorf("write group count: %w", err)
		}
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
	vectors, labels, count, _, err := LoadBinaryWithGroups(path)
	return vectors, labels, count, err
}

func LoadBinaryWithGroups(path string) ([]float32, []byte, int, GroupRanges, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, GroupRanges{}, fmt.Errorf("open references binary: %w", err)
	}
	defer file.Close()

	header := make([]byte, len(fileMagic))
	if _, err := io.ReadFull(file, header); err != nil {
		return nil, nil, 0, GroupRanges{}, fmt.Errorf("read magic: %w", err)
	}
	if string(header) != fileMagic {
		return nil, nil, 0, GroupRanges{}, fmt.Errorf("invalid references binary magic: %q", string(header))
	}

	var version uint32
	if err := binary.Read(file, binary.LittleEndian, &version); err != nil {
		return nil, nil, 0, GroupRanges{}, fmt.Errorf("read version: %w", err)
	}

	var count32 uint32
	if err := binary.Read(file, binary.LittleEndian, &count32); err != nil {
		return nil, nil, 0, GroupRanges{}, fmt.Errorf("read count: %w", err)
	}
	count := int(count32)
	groups := GroupRanges{
		All: GroupRange{Start: 0, Count: count},
	}

	switch version {
	case 1:
		groups = deriveGroupRangesFromVectorsReader(file, count)
	case fileVersion:
		groupTargets := []*GroupRange{&groups.TT, &groups.TF, &groups.FT, &groups.FF}
		for _, group := range groupTargets {
			var start32 uint32
			var count32 uint32
			if err := binary.Read(file, binary.LittleEndian, &start32); err != nil {
				return nil, nil, 0, GroupRanges{}, fmt.Errorf("read group start: %w", err)
			}
			if err := binary.Read(file, binary.LittleEndian, &count32); err != nil {
				return nil, nil, 0, GroupRanges{}, fmt.Errorf("read group count: %w", err)
			}
			group.Start = int(start32)
			group.Count = int(count32)
		}
	default:
		return nil, nil, 0, GroupRanges{}, fmt.Errorf("unsupported references binary version: %d", version)
	}

	vectors := make([]float32, count*VectorDimensions)
	if err := binary.Read(file, binary.LittleEndian, vectors); err != nil {
		return nil, nil, 0, GroupRanges{}, fmt.Errorf("read vectors: %w", err)
	}

	labels := make([]byte, count)
	if _, err := io.ReadFull(file, labels); err != nil {
		return nil, nil, 0, GroupRanges{}, fmt.Errorf("read labels: %w", err)
	}

	if version == 1 {
		groups = deriveGroupRanges(vectors, count)
	}

	return vectors, labels, count, groups, nil
}

func deriveGroupRanges(vectors []float32, count int) GroupRanges {
	startTT := 0
	ttCount := 0
	tfCount := 0
	ftCount := 0
	ffCount := 0

	for i := 0; i < count; i++ {
		offset := i * VectorDimensions
		isOnline := vectors[offset+9] >= 0.5
		cardPresent := vectors[offset+10] >= 0.5
		switch {
		case isOnline && cardPresent:
			ttCount++
		case isOnline && !cardPresent:
			tfCount++
		case !isOnline && cardPresent:
			ftCount++
		default:
			ffCount++
		}
	}

	startTF := startTT + ttCount
	startFT := startTF + tfCount
	startFF := startFT + ftCount

	return GroupRanges{
		All: GroupRange{Start: 0, Count: count},
		TT:  GroupRange{Start: startTT, Count: ttCount},
		TF:  GroupRange{Start: startTF, Count: tfCount},
		FT:  GroupRange{Start: startFT, Count: ftCount},
		FF:  GroupRange{Start: startFF, Count: ffCount},
	}
}

func ReorderByBooleanGroups(vectors []float32, labels []byte, count int) ([]float32, []byte, GroupRanges) {
	groupBuckets := [4][]int{}
	for i := 0; i < count; i++ {
		offset := i * VectorDimensions
		isOnline := vectors[offset+9] >= 0.5
		cardPresent := vectors[offset+10] >= 0.5
		groupBuckets[groupIndex(isOnline, cardPresent)] = append(groupBuckets[groupIndex(isOnline, cardPresent)], i)
	}

	reorderedVectors := make([]float32, 0, len(vectors))
	reorderedLabels := make([]byte, 0, len(labels))

	ranges := GroupRanges{
		All: GroupRange{Start: 0, Count: count},
	}

	appendGroup := func(target *[]int, group *GroupRange) {
		group.Start = len(reorderedLabels)
		group.Count = len(*target)
		for _, originalIndex := range *target {
			offset := originalIndex * VectorDimensions
			reorderedVectors = append(reorderedVectors, vectors[offset:offset+VectorDimensions]...)
			reorderedLabels = append(reorderedLabels, labels[originalIndex])
		}
	}

	appendGroup(&groupBuckets[0], &ranges.TT)
	appendGroup(&groupBuckets[1], &ranges.TF)
	appendGroup(&groupBuckets[2], &ranges.FT)
	appendGroup(&groupBuckets[3], &ranges.FF)

	return reorderedVectors, reorderedLabels, ranges
}

func groupIndex(isOnline, cardPresent bool) int {
	switch {
	case isOnline && cardPresent:
		return 0
	case isOnline && !cardPresent:
		return 1
	case !isOnline && cardPresent:
		return 2
	default:
		return 3
	}
}

func deriveGroupRangesFromVectorsReader(_ io.Reader, _ int) GroupRanges {
	return GroupRanges{}
}
