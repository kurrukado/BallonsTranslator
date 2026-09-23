package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// ============================================================================
// Configuration
// ============================================================================

type Config struct {
	Port         string
	GeminiAPIKey string
	DelayBetween time.Duration
	MaxRetries   int
	BaseURL      string
}

var config = Config{
	Port:         "8080",
	GeminiAPIKey: "",
	DelayBetween: 4 * time.Second,
	MaxRetries:   3,
	BaseURL:      "https://generativelanguage.googleapis.com/v1beta/openai",
}

// ============================================================================
// Event Logger - SSE broadcast
// ============================================================================

type LogEntry struct {
	Tag       string `json:"tag"`
	Message   string `json:"message"`
	Stage     string `json:"stage"`
	Timestamp string `json:"timestamp"`
}

type EventBroadcaster struct {
	mu      sync.RWMutex
	clients map[chan LogEntry]bool
}

var broadcaster = &EventBroadcaster{clients: make(map[chan LogEntry]bool)}

func (b *EventBroadcaster) Subscribe() chan LogEntry {
	ch := make(chan LogEntry, 50)
	b.mu.Lock()
	b.clients[ch] = true
	b.mu.Unlock()
	return ch
}

func (b *EventBroadcaster) Unsubscribe(ch chan LogEntry) {
	b.mu.Lock()
	delete(b.clients, ch)
	close(ch)
	b.mu.Unlock()
}

func (b *EventBroadcaster) Send(tag, msg, stage string) {
	entry := LogEntry{Tag: tag, Message: msg, Stage: stage, Timestamp: time.Now().Format(time.RFC3339)}
	b.mu.RLock()
	for ch := range b.clients {
		select {
		case ch <- entry:
		default:
		}
	}
	b.mu.RUnlock()
	log.Printf("[%s] %s", tag, msg)
}

// ============================================================================
// Request History
// ============================================================================

type RequestRecord struct {
	ID         int    `json:"id"`
	Model      string `json:"model"`
	Status     string `json:"status"` // queued, processing, ok, error
	StatusCode int    `json:"status_code"`
	Detail     string `json:"detail"`
	Duration   string `json:"duration"`
	ReceivedAt string `json:"received_at"`
}

type RequestHistory struct {
	mu      sync.RWMutex
	records []*RequestRecord
}

var history = &RequestHistory{}

func (h *RequestHistory) Add(r *RequestRecord) {
	h.mu.Lock()
	h.records = append(h.records, r)
	if len(h.records) > 100 {
		h.records = h.records[len(h.records)-100:]
	}
	h.mu.Unlock()
}

func (h *RequestHistory) Update(id int, status string, statusCode int, detail string, duration string) {
	h.mu.Lock()
	for i := len(h.records) - 1; i >= 0; i-- {
		if h.records[i].ID == id {
			h.records[i].Status = status
			h.records[i].StatusCode = statusCode
			h.records[i].Detail = detail
			h.records[i].Duration = duration
			break
		}
	}
	h.mu.Unlock()
}

func (h *RequestHistory) GetAll() []*RequestRecord {
	h.mu.RLock()
	defer h.mu.RUnlock()
	// Return reversed (newest first)
	out := make([]*RequestRecord, len(h.records))
	for i, r := range h.records {
		out[len(h.records)-1-i] = r
	}
	return out
}

// ============================================================================
// Request Queue
// ============================================================================

type QueueItem struct {
	ID       int
	Method   string
	Path     string
	Headers  http.Header
	Body     []byte
	Model    string
	Response chan *QueueResponse
}

type QueueResponse struct {
	StatusCode int
	Headers    http.Header
	Body       []byte
	Error      error
}

type RequestQueue struct {
	mu             sync.Mutex
	queue          chan *QueueItem
	counter        int
	processed      int
	failed         int
	lastReq        time.Time
	currentRequest string
}

var reqQueue = &RequestQueue{queue: make(chan *QueueItem, 500)}

func (q *RequestQueue) Enqueue(item *QueueItem) {
	q.mu.Lock()
	q.counter++
	item.ID = q.counter
	q.mu.Unlock()

	history.Add(&RequestRecord{ID: item.ID, Model: item.Model, Status: "queued", ReceivedAt: time.Now().Format(time.RFC3339)})
	broadcaster.Send("QUEUE", fmt.Sprintf("#%d enqueued | model=%s | pending ~%d", item.ID, item.Model, len(q.queue)+1), "queue")
	q.queue <- item
}

func (q *RequestQueue) StartWorker() {
	go func() {
		for item := range q.queue {
			start := time.Now()

			q.mu.Lock()
			q.currentRequest = fmt.Sprintf("#%d", item.ID)
			elapsed := time.Since(q.lastReq)
			q.mu.Unlock()

			history.Update(item.ID, "processing", 0, "rate limit check", "")

			if elapsed < config.DelayBetween {
				wait := config.DelayBetween - elapsed
				broadcaster.Send("QUEUE", fmt.Sprintf("#%d waiting %.1fs (rate limit)", item.ID, wait.Seconds()), "wait")
				time.Sleep(wait)
			}

			broadcaster.Send("PROXY", fmt.Sprintf("#%d forwarding to Gemini | model=%s", item.ID, item.Model), "proxy")
			history.Update(item.ID, "processing", 0, "forwarding to Gemini...", "")
			resp := q.forwardRequest(item)

			q.mu.Lock()
			q.lastReq = time.Now()
			q.mu.Unlock()

			// Handle 429 retry
			if resp.StatusCode == 429 {
				broadcaster.Send("ERROR", fmt.Sprintf("#%d got 429 RATE LIMITED", item.ID), "error")
				for retry := 1; retry <= config.MaxRetries; retry++ {
					backoff := time.Duration(retry*retry) * 15 * time.Second
					broadcaster.Send("QUEUE", fmt.Sprintf("#%d retry %d/%d, backoff %v", item.ID, retry, config.MaxRetries, backoff), "wait")
					history.Update(item.ID, "processing", 429, fmt.Sprintf("retry %d/%d", retry, config.MaxRetries), "")
					time.Sleep(backoff)
					resp = q.forwardRequest(item)
					if resp.StatusCode != 429 {
						broadcaster.Send("PROXY", fmt.Sprintf("#%d retry %d succeeded", item.ID, retry), "proxy")
						break
					}
				}
			}

			dur := time.Since(start)
			q.mu.Lock()
			q.currentRequest = ""
			if resp.Error != nil || resp.StatusCode >= 400 {
				q.failed++
				detail := fmt.Sprintf("status=%d", resp.StatusCode)
				if resp.Error != nil {
					detail = resp.Error.Error()
				}
				history.Update(item.ID, "error", resp.StatusCode, detail, dur.Round(time.Millisecond).String())
				broadcaster.Send("ERROR", fmt.Sprintf("#%d failed | %s | %s", item.ID, detail, dur.Round(time.Millisecond)), "error")
			} else {
				q.processed++
				history.Update(item.ID, "ok", resp.StatusCode, fmt.Sprintf("%d bytes", len(resp.Body)), dur.Round(time.Millisecond).String())
				broadcaster.Send("PROXY", fmt.Sprintf("#%d done | %d bytes | %s", item.ID, len(resp.Body), dur.Round(time.Millisecond)), "done")
			}
			q.mu.Unlock()

			item.Response <- resp
		}
	}()
	broadcaster.Send("QUEUE", "Worker started - sequential processing", "queue")
}

func (q *RequestQueue) forwardRequest(item *QueueItem) *QueueResponse {
	path := item.Path
	if strings.HasPrefix(path, "/v1") {
		path = strings.TrimPrefix(path, "/v1")
	}
	targetURL := config.BaseURL + path

	req, err := http.NewRequest(item.Method, targetURL, bytes.NewReader(item.Body))
	if err != nil {
		return &QueueResponse{Error: err, StatusCode: 502}
	}
	for k, v := range item.Headers {
		lk := strings.ToLower(k)
		if lk == "host" || lk == "content-length" || lk == "authorization" {
			continue
		}
		req.Header[k] = v
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+config.GeminiAPIKey)

	client := &http.Client{Timeout: 180 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return &QueueResponse{Error: err, StatusCode: 502}
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return &QueueResponse{Error: err, StatusCode: 502}
	}
	return &QueueResponse{StatusCode: resp.StatusCode, Headers: resp.Header, Body: body}
}

// ============================================================================
// Sanitize unsupported OpenAI fields
// ============================================================================

var unsupportedFields = []string{
	"frequency_penalty", "presence_penalty", "logprobs", "top_logprobs",
	"logit_bias", "user", "seed", "service_tier", "store", "suffix", "best_of",
}

func sanitizeRequestBody(body []byte) ([]byte, string) {
	var data map[string]interface{}
	if err := json.Unmarshal(body, &data); err != nil {
		return body, ""
	}
	removed := []string{}
	for _, f := range unsupportedFields {
		if _, ok := data[f]; ok {
			delete(data, f)
			removed = append(removed, f)
		}
	}
	model, _ := data["model"].(string)
	if len(removed) > 0 {
		broadcaster.Send("SANITIZE", fmt.Sprintf("Stripped: %s", strings.Join(removed, ", ")), "sanitize")
		b, _ := json.Marshal(data)
		return b, model
	}
	return body, model
}

// ============================================================================
// HTTP Handlers
// ============================================================================

func proxyHandler(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	broadcaster.Send("RECV", fmt.Sprintf("%s %s | %d bytes", r.Method, r.URL.Path, len(body)), "recv")
	body, model := sanitizeRequestBody(body)

	ch := make(chan *QueueResponse, 1)
	item := &QueueItem{Method: r.Method, Path: r.URL.Path, Headers: r.Header, Body: body, Model: model, Response: ch}
	reqQueue.Enqueue(item)
	resp := <-ch

	if resp.Error != nil {
		http.Error(w, fmt.Sprintf(`{"error":{"message":"proxy: %v"}}`, resp.Error), http.StatusBadGateway)
		return
	}
	for k, v := range resp.Headers {
		for _, vv := range v {
			w.Header().Add(k, vv)
		}
	}
	setCORS(w)
	w.WriteHeader(resp.StatusCode)
	w.Write(resp.Body)
}

func modelsHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	setCORS(w)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
			// Gemini 3.x family (free tier)
			{"id": "gemini-3.8-flash", "object": "model", "owned_by": "google"},
			{"id": "gemini-3.7-flash", "object": "model", "owned_by": "google"},
			{"id": "gemini-3.6-flash", "object": "model", "owned_by": "google"},
			{"id": "gemini-3.5-flash", "object": "model", "owned_by": "google"},
			{"id": "gemini-3.5-flash-lite", "object": "model", "owned_by": "google"},
			// Legacy models
			{"id": "gemini-2.0-flash", "object": "model", "owned_by": "google"},
			{"id": "gemini-1.5-flash", "object": "model", "owned_by": "google"},
		},
	})
}

func statusHandler(w http.ResponseWriter, r *http.Request) {
	reqQueue.mu.Lock()
	s := map[string]interface{}{
		"status":          "running",
		"queue_pending":   len(reqQueue.queue),
		"total_received":  reqQueue.counter,
		"processed":       reqQueue.processed,
		"failed":          reqQueue.failed,
		"delay_seconds":   config.DelayBetween.Seconds(),
		"api_key_set":     config.GeminiAPIKey != "",
		"current_request": reqQueue.currentRequest,
	}
	reqQueue.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	setCORS(w)
	json.NewEncoder(w).Encode(s)
}

func historyHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	setCORS(w)
	json.NewEncoder(w).Encode(history.GetAll())
}

func configHandler(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	if r.Method == "POST" {
		var u map[string]interface{}
		json.NewDecoder(r.Body).Decode(&u)
		if v, ok := u["delay_seconds"].(float64); ok {
			config.DelayBetween = time.Duration(v * float64(time.Second))
			broadcaster.Send("CONFIG", fmt.Sprintf("Delay → %.1fs", v), "")
		}
		if v, ok := u["max_retries"].(float64); ok {
			config.MaxRetries = int(v)
		}
		if v, ok := u["api_key"].(string); ok && v != "" {
			config.GeminiAPIKey = v
			broadcaster.Send("CONFIG", "API key updated", "")
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "updated"})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"delay_seconds": config.DelayBetween.Seconds(),
		"max_retries":   config.MaxRetries,
		"api_key_set":   config.GeminiAPIKey != "",
	})
}

// SSE endpoint for real-time logs
func logsStreamHandler(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	setCORS(w)

	ch := broadcaster.Subscribe()
	defer broadcaster.Unsubscribe(ch)

	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case entry := <-ch:
			data, _ := json.Marshal(entry)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	}
}

func setCORS(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "*")
}

func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		setCORS(w)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		next(w, r)
	}
}

// Dashboard serves the HTML file
func dashboardHandler(w http.ResponseWriter, r *http.Request) {
	if strings.HasPrefix(r.URL.Path, "/v1") {
		proxyHandler(w, r)
		return
	}
	data, err := os.ReadFile("dashboard.html")
	if err != nil {
		http.Error(w, "Dashboard file not found", http.StatusInternalServerError)
		return
	}
	html := strings.ReplaceAll(string(data), "{{PORT}}", config.Port)
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(html))
}

// ============================================================================
// Main
// ============================================================================

func main() {
	if key := os.Getenv("GEMINI_API_KEY"); key != "" {
		config.GeminiAPIKey = key
	}
	if len(os.Args) > 1 {
		config.GeminiAPIKey = os.Args[1]
	}
	if len(os.Args) > 2 {
		config.Port = os.Args[2]
	}

	reqQueue.StartWorker()

	http.HandleFunc("/api/status", corsMiddleware(statusHandler))
	http.HandleFunc("/api/config", corsMiddleware(configHandler))
	http.HandleFunc("/api/history", corsMiddleware(historyHandler))
	http.HandleFunc("/api/logs/stream", logsStreamHandler)
	http.HandleFunc("/v1/chat/completions", corsMiddleware(proxyHandler))
	http.HandleFunc("/v1/models", corsMiddleware(modelsHandler))
	http.HandleFunc("/v1/", corsMiddleware(proxyHandler))
	http.HandleFunc("/", dashboardHandler)

	log.Println("╔═══════════════════════════════════════════════════╗")
	log.Println("║        Gemini API Proxy - Sequential Queue        ║")
	log.Println("╠═══════════════════════════════════════════════════╣")
	log.Printf("║  Dashboard:  http://localhost:%s                  ║\n", config.Port)
	log.Printf("║  Delay:      %.0fs  |  API Key: %-5v              ║\n", config.DelayBetween.Seconds(), config.GeminiAPIKey != "")
	log.Println("╚═══════════════════════════════════════════════════╝")

	if err := http.ListenAndServe(":"+config.Port, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
