package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/MXLange/rinha-de-backend-2026-go/internal/referenceio"
	"github.com/coder/hnsw"
	"github.com/gofiber/fiber/v2"
)

const (
	vectorDimensions = 14
	fraudThreshold   = 0.6
	neighborCount    = 5
)

type searchBackend string

const (
	searchBackendExact searchBackend = "exact"
	searchBackendHNSW  searchBackend = "hnsw"
)

type config struct {
	port                string
	resourcesDir        string
	referencesPath      string
	hnswPath            string
	normalizationPath   string
	mccRiskPath         string
	searchBackend       searchBackend
	hnswM               int
	hnswEfSearch        int
	requestReadTimeout  time.Duration
	requestWriteTimeout time.Duration
}

type server struct {
	ready atomic.Bool
	model atomic.Pointer[model]
}

type model struct {
	searchBackend searchBackend
	vectors       []float32
	count         int
	graph         *hnsw.Graph[int]
	labels        []byte
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
		referencesPath:      filepath.Join(resourcesDir, "references.bin"),
		hnswPath:            filepath.Join(resourcesDir, "hnsw.bin"),
		normalizationPath:   filepath.Join(resourcesDir, "normalization.json"),
		mccRiskPath:         filepath.Join(resourcesDir, "mcc_risk.json"),
		searchBackend:       envSearchBackendOrDefault("SEARCH_BACKEND", searchBackendExact),
		hnswM:               envIntOrDefault("HNSW_M", 16),
		hnswEfSearch:        envIntOrDefault("HNSW_EF_SEARCH", 64),
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

func envSearchBackendOrDefault(key string, fallback searchBackend) searchBackend {
	value := searchBackend(envOrDefault(key, string(fallback)))
	switch value {
	case searchBackendExact, searchBackendHNSW:
		return value
	default:
		return fallback
	}
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

	var graph *hnsw.Graph[int]
	if cfg.searchBackend == searchBackendHNSW {
		graph, err = loadHNSW(cfg.hnswPath, cfg.hnswEfSearch)
		if err != nil {
			return nil, err
		}
		log.Printf("loaded %d labeled vectors into HNSW (M=%d, EfSearch=%d)", count, cfg.hnswM, cfg.hnswEfSearch)
	} else {
		log.Printf("loaded %d labeled vectors for exact search", count)
	}

	return &model{
		searchBackend: cfg.searchBackend,
		vectors:       vectors,
		count:         count,
		graph:         graph,
		labels:        labels,
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

func loadReferences(path string) ([]float32, []byte, int, error) {
	return referenceio.LoadBinary(path)
}

func loadHNSW(path string, efSearch int) (*hnsw.Graph[int], error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open hnsw index: %w", err)
	}
	defer file.Close()

	graph := hnsw.NewGraph[int]()
	if err := graph.Import(bufio.NewReader(file)); err != nil {
		return nil, fmt.Errorf("import hnsw index: %w", err)
	}
	graph.EfSearch = efSearch
	return graph, nil
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

	var (
		fraudScore float64
		ok         bool
	)
	switch model.searchBackend {
	case searchBackendHNSW:
		fraudScore, ok = approximateTopKFraudScore(model.graph, model.labels, vector[:])
	default:
		fraudScore, ok = exactTopKFraudScore(model.vectors, model.labels, model.count, vector)
	}
	if !ok {
		return fraudResponse{}, errors.New("no references available")
	}
	return fraudResponse{
		Approved:   fraudScore < fraudThreshold,
		FraudScore: fraudScore,
	}, nil
}

func approximateTopKFraudScore(graph *hnsw.Graph[int], labels []byte, query []float32) (float64, bool) {
	if graph == nil || graph.Len() == 0 {
		return 0, false
	}

	nodes := graph.Search(normalizeL2(query), neighborCount)
	if len(nodes) == 0 {
		return 0, false
	}

	fraudVotes := 0
	for _, node := range nodes {
		if node.Key >= 0 && node.Key < len(labels) && labels[node.Key] == 1 {
			fraudVotes++
		}
	}

	return float64(fraudVotes) / float64(len(nodes)), true
}

func exactTopKFraudScore(vectors []float32, labels []byte, count int, query [vectorDimensions]float32) (float64, bool) {
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
		dist := squaredDistanceAt(&query, vectors, offset)
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
		bestFrauds[insertAt] = labels[i] == 1
	}

	fraudVotes := 0
	for i := 0; i < limit; i++ {
		if bestFrauds[i] {
			fraudVotes++
		}
	}

	return float64(fraudVotes) / float64(limit), true
}

func squaredDistanceAt(query *[vectorDimensions]float32, vectors []float32, offset int) float32 {
	d0 := query[0] - vectors[offset]
	d1 := query[1] - vectors[offset+1]
	d2 := query[2] - vectors[offset+2]
	d3 := query[3] - vectors[offset+3]
	d4 := query[4] - vectors[offset+4]
	d5 := query[5] - vectors[offset+5]
	d6 := query[6] - vectors[offset+6]
	d7 := query[7] - vectors[offset+7]
	d8 := query[8] - vectors[offset+8]
	d9 := query[9] - vectors[offset+9]
	d10 := query[10] - vectors[offset+10]
	d11 := query[11] - vectors[offset+11]
	d12 := query[12] - vectors[offset+12]
	d13 := query[13] - vectors[offset+13]

	return d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
		d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13
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

	unknownMerchant := float32(1)
	for _, merchantID := range req.Customer.KnownMerchants {
		if merchantID == req.Merchant.ID {
			unknownMerchant = 0
			break
		}
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
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}
