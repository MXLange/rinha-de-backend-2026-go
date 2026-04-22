package main

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync/atomic"
	"time"

	"github.com/gofiber/fiber/v2"
)

const (
	vectorDimensions = 14
	fraudThreshold   = 0.6
	neighborCount    = 5
)

type config struct {
	port                string
	resourcesDir        string
	referencesPath      string
	normalizationPath   string
	mccRiskPath         string
	requestReadTimeout  time.Duration
	requestWriteTimeout time.Duration
}

type server struct {
	ready atomic.Bool
	model atomic.Pointer[model]
}

type model struct {
	vectors       []float32
	labels        []bool
	count         int
	normalization normalization
	mccRisk       map[string]float32
}

type normalization struct {
	MaxAmount            float32 `json:"max_amount"`
	MaxInstallments      float32 `json:"max_installments"`
	AmountVsAvgRatio     float32 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float32 `json:"max_minutes"`
	MaxKM                float32 `json:"max_km"`
	MaxTxCount24h        float32 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float32 `json:"max_merchant_avg_amount"`
}

type referenceRecord struct {
	Vector []float32 `json:"vector"`
	Label  string    `json:"label"`
}

type fraudRequest struct {
	ID          string           `json:"id"`
	Transaction transaction      `json:"transaction"`
	Customer    customer         `json:"customer"`
	Merchant    merchant         `json:"merchant"`
	Terminal    terminal         `json:"terminal"`
	LastTx      *lastTransaction `json:"last_transaction"`
}

type transaction struct {
	Amount       float64 `json:"amount"`
	Installments int     `json:"installments"`
	RequestedAt  string  `json:"requested_at"`
}

type customer struct {
	AvgAmount      float64  `json:"avg_amount"`
	TxCount24h     int      `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

type merchant struct {
	ID        string  `json:"id"`
	MCC       string  `json:"mcc"`
	AvgAmount float64 `json:"avg_amount"`
}

type terminal struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KMFromHome  float64 `json:"km_from_home"`
}

type lastTransaction struct {
	Timestamp     string  `json:"timestamp"`
	KMFromCurrent float64 `json:"km_from_current"`
}

type fraudResponse struct {
	Approved   bool    `json:"approved"`
	FraudScore float64 `json:"fraud_score"`
}

func main() {
	cfg := loadConfig()

	srv := &server{}

	model, err := loadModel(cfg)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}

	srv.model.Store(model)
	srv.ready.Store(true)

	app := fiber.New(fiber.Config{
		ReadTimeout:           cfg.requestReadTimeout,
		WriteTimeout:          cfg.requestWriteTimeout,
		IdleTimeout:           30 * time.Second,
		DisableStartupMessage: true,
		Prefork:               false,
	})
	app.Get("/ready", srv.handleReady)
	app.Post("/fraud-score", srv.handleFraudScore)

	log.Printf("listening on :%s", cfg.port)
	if err := app.Listen(":" + cfg.port); err != nil {
		log.Fatalf("fiber server: %v", err)
	}
}

func loadConfig() config {
	resourcesDir := envOrDefault("RESOURCES_DIR", "/app/resources")
	return config{
		port:                envOrDefault("PORT", "8080"),
		resourcesDir:        resourcesDir,
		referencesPath:      filepath.Join(resourcesDir, "references.json.gz"),
		normalizationPath:   filepath.Join(resourcesDir, "normalization.json"),
		mccRiskPath:         filepath.Join(resourcesDir, "mcc_risk.json"),
		requestReadTimeout:  3 * time.Second,
		requestWriteTimeout: 3 * time.Second,
	}
}

func envOrDefault(key, fallback string) string {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	return value
}

func loadModel(cfg config) (*model, error) {
	norm, err := loadNormalization(cfg.normalizationPath)
	if err != nil {
		return nil, err
	}

	mccRisk, err := loadMCCRisk(cfg.mccRiskPath)
	if err != nil {
		return nil, err
	}

	vectors, labels, count, err := loadReferences(cfg.referencesPath)
	if err != nil {
		return nil, err
	}

	log.Printf("loaded %d labeled vectors", count)

	return &model{
		vectors:       vectors,
		labels:        labels,
		count:         count,
		normalization: norm,
		mccRisk:       mccRisk,
	}, nil
}

func loadNormalization(path string) (normalization, error) {
	var norm normalization
	file, err := os.Open(path)
	if err != nil {
		return norm, fmt.Errorf("open normalization file: %w", err)
	}
	defer file.Close()

	if err := json.NewDecoder(file).Decode(&norm); err != nil {
		return norm, fmt.Errorf("decode normalization file: %w", err)
	}
	return norm, nil
}

func loadMCCRisk(path string) (map[string]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open mcc risk file: %w", err)
	}
	defer file.Close()

	table := make(map[string]float32)
	if err := json.NewDecoder(file).Decode(&table); err != nil {
		return nil, fmt.Errorf("decode mcc risk file: %w", err)
	}
	return table, nil
}

func loadReferences(path string) ([]float32, []bool, int, error) {
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

	vectors := make([]float32, 0, 100000*vectorDimensions)
	labels := make([]bool, 0, 100000)
	index := 0

	for decoder.More() {
		var record referenceRecord
		if err := decoder.Decode(&record); err != nil {
			return nil, nil, 0, fmt.Errorf("decode reference record %d: %w", index, err)
		}
		if len(record.Vector) != vectorDimensions {
			return nil, nil, 0, fmt.Errorf("reference record %d has %d dimensions", index, len(record.Vector))
		}

		vectors = append(vectors, record.Vector...)
		labels = append(labels, record.Label == "fraud")
		index++
	}

	if _, err := decoder.Token(); err != nil {
		return nil, nil, 0, fmt.Errorf("read references closing token: %w", err)
	}

	return vectors, labels, index, nil
}

func (s *server) handleReady(c *fiber.Ctx) error {
	if !s.ready.Load() {
		return c.Status(fiber.StatusServiceUnavailable).SendString("not ready")
	}
	return c.SendStatus(fiber.StatusOK)
}

func (s *server) handleFraudScore(c *fiber.Ctx) error {
	if !s.ready.Load() {
		return c.Status(fiber.StatusServiceUnavailable).SendString("not ready")
	}

	model := s.model.Load()
	if model == nil {
		return c.Status(fiber.StatusServiceUnavailable).SendString("model unavailable")
	}

	var req fraudRequest
	if err := decodeRequestBody(c.Body(), &req); err != nil {
		return c.Status(fiber.StatusBadRequest).SendString("invalid request payload")
	}

	response, err := scoreRequest(context.Background(), model, req)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).SendString(err.Error())
	}

	return c.Status(fiber.StatusOK).JSON(response)
}

func decodeRequestBody(body []byte, out any) error {
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	return decoder.Decode(out)
}

func scoreRequest(_ context.Context, model *model, req fraudRequest) (fraudResponse, error) {
	vector, err := vectorize(req, model.normalization, model.mccRisk)
	if err != nil {
		return fraudResponse{}, err
	}

	fraudScore, ok := exactTopKFraudScore(model.vectors, model.labels, model.count, vector)
	if !ok {
		return fraudResponse{}, errors.New("no references available")
	}
	return fraudResponse{
		Approved:   fraudScore < fraudThreshold,
		FraudScore: fraudScore,
	}, nil
}

func exactTopKFraudScore(vectors []float32, labels []bool, count int, query [vectorDimensions]float32) (float64, bool) {
	if count == 0 {
		return 0, false
	}

	limit := neighborCount
	if count < limit {
		limit = count
	}

	bestDistances := [neighborCount]float32{
		float32(math.Inf(1)),
		float32(math.Inf(1)),
		float32(math.Inf(1)),
		float32(math.Inf(1)),
		float32(math.Inf(1)),
	}
	bestFrauds := [neighborCount]bool{}

	for i := 0; i < count; i++ {
		offset := i * vectorDimensions
		dist := squaredDistance(query, vectors[offset:offset+vectorDimensions])
		if dist >= bestDistances[limit-1] {
			continue
		}

		insertAt := limit - 1
		for insertAt > 0 && dist < bestDistances[insertAt-1] {
			bestDistances[insertAt] = bestDistances[insertAt-1]
			bestFrauds[insertAt] = bestFrauds[insertAt-1]
			insertAt--
		}

		bestDistances[insertAt] = dist
		bestFrauds[insertAt] = labels[i]
	}

	fraudVotes := 0
	for i := 0; i < limit; i++ {
		if bestFrauds[i] {
			fraudVotes++
		}
	}

	return float64(fraudVotes) / float64(limit), true
}

func squaredDistance(query [vectorDimensions]float32, candidate []float32) float32 {
	var sum float32
	for i := 0; i < vectorDimensions; i++ {
		diff := query[i] - candidate[i]
		sum += diff * diff
	}
	return sum
}

func vectorize(req fraudRequest, norm normalization, mccRisk map[string]float32) ([vectorDimensions]float32, error) {
	requestedAt, err := time.Parse(time.RFC3339, req.Transaction.RequestedAt)
	if err != nil {
		return [vectorDimensions]float32{}, fmt.Errorf("invalid transaction.requested_at: %w", err)
	}

	var minutesSinceLast float32 = -1
	var kmFromLast float32 = -1

	if req.LastTx != nil {
		lastTxAt, err := time.Parse(time.RFC3339, req.LastTx.Timestamp)
		if err != nil {
			return [vectorDimensions]float32{}, fmt.Errorf("invalid last_transaction.timestamp: %w", err)
		}
		diffMinutes := float32(requestedAt.Sub(lastTxAt).Minutes())
		if diffMinutes < 0 {
			diffMinutes = 0
		}
		minutesSinceLast = clamp01(diffMinutes / norm.MaxMinutes)
		kmFromLast = clamp01(float32(req.LastTx.KMFromCurrent) / norm.MaxKM)
	}

	knownMerchant := slices.Contains(req.Customer.KnownMerchants, req.Merchant.ID)
	unknownMerchant := float32(1)
	if knownMerchant {
		unknownMerchant = 0
	}

	risk, ok := mccRisk[req.Merchant.MCC]
	if !ok {
		risk = 0.5
	}

	vector := [vectorDimensions]float32{
		clamp01(float32(req.Transaction.Amount) / norm.MaxAmount),
		clamp01(float32(req.Transaction.Installments) / norm.MaxInstallments),
		amountVsAverage(float32(req.Transaction.Amount), float32(req.Customer.AvgAmount), norm.AmountVsAvgRatio),
		clamp01(float32(requestedAt.UTC().Hour()) / 23),
		clamp01(float32(toChallengeWeekday(requestedAt.UTC().Weekday())) / 6),
		minutesSinceLast,
		kmFromLast,
		clamp01(float32(req.Terminal.KMFromHome) / norm.MaxKM),
		clamp01(float32(req.Customer.TxCount24h) / norm.MaxTxCount24h),
		boolToFloat32(req.Terminal.IsOnline),
		boolToFloat32(req.Terminal.CardPresent),
		unknownMerchant,
		risk,
		clamp01(float32(req.Merchant.AvgAmount) / norm.MaxMerchantAvgAmount),
	}

	return vector, nil
}

func amountVsAverage(amount, average, ratio float32) float32 {
	if average <= 0 {
		return 1
	}
	return clamp01((amount / average) / ratio)
}

func toChallengeWeekday(day time.Weekday) int {
	switch day {
	case time.Monday:
		return 0
	case time.Tuesday:
		return 1
	case time.Wednesday:
		return 2
	case time.Thursday:
		return 3
	case time.Friday:
		return 4
	case time.Saturday:
		return 5
	default:
		return 6
	}
}

func boolToFloat32(value bool) float32 {
	if value {
		return 1
	}
	return 0
}

func clamp01(value float32) float32 {
	return float32(math.Min(1, math.Max(0, float64(value))))
}
