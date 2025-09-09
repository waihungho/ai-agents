The "Aether" AI Agent is a sophisticated, context-aware intelligence layer designed for edge deployment, directly interfacing with Microcontroller Perception (MCP) units. Its core philosophy emphasizes real-time adaptive intelligence, proactive intervention, and a transparent, self-improving operational model. Aether's capabilities span advanced sensor data fusion, predictive analytics, ethical decision-making, secure data handling, and self-management, making it a versatile platform for intelligent environmental control, smart infrastructure, and human-centric automation.

---

## Aether AI Agent: Outline & Function Summary

### A. Core Perception & MCP Interface (Functions 1-5)

1.  **`IngestSerialSensorData(ctx context.Context)`**:
    *   **Summary**: Continuously reads, parses, and validates structured sensor data streams (e.g., JSON, Protobuf) received via a serial communication interface from connected MCPs. Handles error recovery and ensures data integrity.
    *   **Concept**: Foundation of real-time perception. Converts raw byte streams into meaningful data models for the agent.

2.  **`DynamicSensorCalibration(sensorData models.SensorData)`**:
    *   **Summary**: Applies real-time, adaptive calibration algorithms (e.g., Kalman filters for noise reduction, polynomial fitting for drift compensation) to raw sensor inputs, dynamically adjusting parameters based on environmental factors or historical performance to enhance data accuracy and reliability.
    *   **Concept**: Improves data quality beyond static factory calibration, crucial for long-term sensor deployment and precision.

3.  **`CrossModalPerceptionFusion()`**:
    *   **Summary**: Fuses data from diverse sensor modalities (e.g., ambient light, thermal, acoustic, haptic, proximity, air quality) obtained from multiple MCPs to build a holistic, enriched, and robust understanding of the environment, inferring states that individual sensors cannot.
    *   **Concept**: Mimics biological perception by combining multiple senses to form a more complete and resilient environmental model.

4.  **`GenerateHapticFeedback(pattern models.HapticPattern)`**:
    *   **Summary**: Translates complex internal agent states, abstract data, or warning signals into nuanced haptic patterns (e.g., varying intensity, frequency, duration) delivered through an MCP-controlled haptic feedback module to a user or another system for non-intrusive communication.
    *   **Concept**: Extends traditional visual/audio feedback with a subtle, physical interface for information delivery or alerts.

5.  **`PredictiveSensorHealth()`**:
    *   **Summary**: Analyzes operational parameters (e.g., power consumption anomalies, response latency drifts, unusual noise profiles, communication errors) of attached MCPs and their sensors to predict potential failures, degradation, or required maintenance before they impact performance.
    *   **Concept**: Proactive maintenance, moving beyond reactive fault detection to preemptive problem resolution, increasing system uptime and reliability.

### B. Reasoning & Decision-Making Engine (Functions 6-10)

6.  **`ContextualAnomalyDetection()`**:
    *   **Summary**: Learns baseline environmental and behavioral patterns (e.g., typical temperature ranges for time of day, usual human activity patterns), then identifies subtle, context-dependent deviations that signify unusual events, inefficiencies, or potential security risks, going beyond simple threshold breaches.
    *   **Concept**: Intelligent filtering of noise, focusing on statistically significant and contextually relevant anomalies, improving alert accuracy.

7.  **`ProactiveNudgingEngine(ctx context.Context)`**:
    *   **Summary**: Based on learned user preferences, environmental conditions, and task context, this engine generates gentle, timely suggestions or initiates automated subtle adjustments (e.g., adjust lighting for eye strain, suggest a break based on focus levels, optimize ventilation) aimed at improving user well-being, efficiency, or environmental sustainability.
    *   **Concept**: AI as a helpful assistant, subtly guiding towards optimal states rather than just reacting to explicit commands.

8.  **`IntentDeconstructionEngine(intent models.UserIntent)`**:
    *   **Summary**: Interprets high-level, natural language user goals (e.g., "I need to focus," "Optimize for energy savings," "Prepare for sleep") and translates them into a sequence of actionable agent tasks and precise MCP commands, often requiring multi-modal coordination.
    *   **Concept**: Bridges the gap between abstract human desires and concrete system actions, enabling more intuitive control.

9.  **`EthicalConstraintEvaluator(proposedAction models.ActuatorCommand)`**:
    *   **Summary**: Incorporates a configurable, lightweight ethical framework to assess potential agent actions and their consequences. It flags or modifies actions that might violate predefined ethical principles (e.g., prioritize user safety over convenience, balance comfort with energy efficiency, prevent data over-collection).
    *   **Concept**: Infuses moral reasoning into autonomous systems, preventing unintended negative consequences and building trust.

10. **`EphemeralInsightSummarizer()`**:
    *   **Summary**: Continuously processes high-volume, transient sensor data streams in real-time, applying streaming analytics to extract and summarize critical, actionable insights that are immediately relevant (e.g., "Air quality dropped by 15% in last 5 mins," "Motion detected in unoccupied zone for 30s") without requiring exhaustive historical data storage.
    *   **Concept**: Efficiently extracts value from vast data streams by focusing on immediate relevance, reducing storage overhead.

### C. Action, Control & Self-Management (Functions 11-15)

11. **`ExecuteAdaptiveActuation(command models.ActuatorCommand)`**:
    *   **Summary**: Sends precisely timed and modulated control signals (e.g., motor speeds, light intensity, valve states, haptic strength) back to MCPs, adapting outputs based on real-time feedback from sensors to achieve target states effectively and efficiently.
    *   **Concept**: Closed-loop control that dynamically responds to environmental changes, ensuring optimal performance.

12. **`DynamicResourceAllocator()`**:
    *   **Summary**: Optimally distributes computational power, network bandwidth, and energy resources among various connected MCPs and internal agent modules based on current priorities, sensor load, predicted demand, and overall system health to maximize efficiency and responsiveness.
    *   **Concept**: Intelligent resource management at the edge, crucial for energy-constrained or high-demand environments.

13. **`AutonomousSelfHealing()`**:
    *   **Summary**: Detects and initiates recovery procedures for common MCP-level faults (e.g., automatic sensor recalibration, module restart, switching to redundant sensors/communication paths, limited remote firmware updates for specific modules), aiming to maintain continuous operation.
    *   **Concept**: Self-sufficiency and resilience, minimizing human intervention for routine operational issues.

14. **`RealtimeDigitalTwinSynchronization()`**:
    *   **Summary**: Maintains and continuously updates a lightweight, virtual representation (digital twin) of the physical environment, powered by fused MCP data. This twin enables predictive simulations, supports advanced planning, and offers remote state visualization for human operators.
    *   **Concept**: Provides a comprehensive, up-to-date virtual model of reality, enhancing monitoring, diagnostics, and strategic planning.

15. **`ExplainableActionRationale(actionID string)`**:
    *   **Summary**: Generates clear, concise, and human-understandable explanations for autonomous decisions and actions taken by the agent. It links them directly to perceived environmental conditions, learned patterns, ethical constraints, or explicit rules, fostering trust and transparency.
    *   **Concept**: Addresses the "black box" problem of AI, providing accountability and allowing users to understand *why* an action was taken.

### D. Learning, Adaptation & Advanced Concepts (Functions 16-20)

16. **`ContextualBehavioralLearning()`**:
    *   **Summary**: Continuously learns and refines complex user and environmental behavioral patterns over extended periods, including routines, preferences, and environmental dynamics. This enables more accurate predictions, highly personalized responses, and anticipatory actions.
    *   **Concept**: Deep personalization and predictive intelligence, adapting to individual habits and evolving environments.

17. **`ReinforcementPolicyOptimizer()`**:
    *   **Summary**: Employs lightweight reinforcement learning (RL) techniques (e.g., Q-learning or SARSA variants applied to simplified state-action spaces) to iteratively discover and optimize control policies for complex MCP interactions, maximizing desired outcomes (e.g., energy efficiency, user comfort, stability) through trial and error.
    *   **Concept**: Enables the agent to learn optimal strategies for complex, dynamic control problems without explicit programming.

18. **`SecureFederatedModelAggregator(localModelUpdate models.LearningModel)`**:
    *   **Summary**: Participates in federated learning paradigms, securely aggregating locally trained model improvements (e.g., parameter updates) from multiple distributed Aether agents or MCPs without centralizing sensitive raw data. This enhances collective intelligence while preserving privacy and reducing bandwidth.
    *   **Concept**: Collaborative AI at the edge, leveraging insights from a network of agents while respecting data sovereignty.

19. **`GenerativeScenarioAugmentor()`**:
    *   **Summary**: Synthesizes realistic, novel sensor data scenarios based on observed patterns, identified anomalies, and known environmental factors. This augments training datasets for robust model development, allows for stress-testing the agent's adaptive capabilities, and simulates "what-if" situations.
    *   **Concept**: Self-improving AI that can generate its own training data, pushing the boundaries of its understanding and resilience.

20. **`ImmutableDLTLogger(logEntry models.DLTLogEntry)`**:
    *   **Summary**: Logs critical sensor events, agent decisions, or derived insights to a local, lightweight Distributed Ledger Technology (DLT) or a simplified append-only, cryptographically-chained log. This ensures tamper-proof auditing, enhanced data integrity, and compliance without requiring a full public blockchain.
    *   **Concept**: Provides verifiable data provenance and auditability, critical for sensitive applications and regulatory compliance at the edge.

---

## Golang Source Code for Aether AI Agent

This example provides a foundational structure. For a full production system, each function would involve more sophisticated algorithms, external library integrations (e.g., for specific ML models, full DLT clients), and robust error handling.

**`go.mod`**
```
module aether-agent

go 1.21

require (
	github.com/goburrow/modbus v0.0.0-20180415170313-054238e55e00 // Placeholder for potential Modbus RTU/TCP if serial port is advanced
	github.com/tarm/serial v0.0.0-20180822001140-592535d2149e // For serial port communication
	gopkg.in/yaml.v2 v2.4.0 // For config loading
)
```

**`main.go`**
```go
package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"aether-agent/agent"
	"aether-agent/config"
	"aether-agent/models"
	"aether-agent/utils"
)

func main() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
	log.Println("Starting Aether AI Agent...")

	// Load configuration
	cfg, err := config.LoadConfig("config.yaml") // Assume config.yaml exists
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Initialize Aether Agent
	aetherAgent, err := agent.NewAetherAgent(cfg)
	if err != nil {
		log.Fatalf("Failed to initialize Aether Agent: %v", err)
	}

	// Create a context that can be cancelled for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start agent's core processing loops
	go aetherAgent.Start(ctx)

	// --- Simulate MCP input and external events for demonstration ---
	// In a real system, IngestSerialSensorData would be actively reading from a serial port.
	// Here, we're directly pushing data into the agent for simplicity.
	go func() {
		ticker := time.NewTicker(2 * time.Second) // Simulate sensor updates every 2 seconds
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				// Simulate Temperature Sensor Data
				aetherAgent.IngestSensorData(models.SensorData{
					Timestamp: time.Now(),
					Type:      models.Temperature,
					Value:     utils.RandomFloat(20.0, 30.0), // Simulate fluctuation
					Unit:      "°C",
					Location:  "RoomA",
					Raw:       utils.RandomFloat(19.8, 30.2), // Raw value for calibration
					MCPID:     "MCP001",
				})
				// Simulate Humidity Sensor Data
				aetherAgent.IngestSensorData(models.SensorData{
					Timestamp: time.Now(),
					Type:      models.Humidity,
					Value:     utils.RandomFloat(40.0, 60.0),
					Unit:      "%",
					Location:  "RoomA",
					Raw:       utils.RandomFloat(39.5, 60.5),
					MCPID:     "MCP001",
				})
				// Simulate Light Sensor Data for CrossModalFusion
				aetherAgent.IngestSensorData(models.SensorData{
					Timestamp: time.Now(),
					Type:      models.Light,
					Value:     utils.RandomFloat(100.0, 500.0),
					Unit:      "Lux",
					Location:  "RoomA",
					MCPID:     "MCP002",
				})

				// Simulate a request for Haptic Feedback (e.g., if anomaly detected)
				if utils.RandomFloat(0, 1) > 0.8 { // 20% chance to request haptic feedback
					aetherAgent.GenerateHapticFeedback(models.HapticPattern{
						Intensity:   utils.RandomFloat(0.3, 0.9),
						DurationMs:  utils.RandomInt(200, 1000),
						PatternType: models.PatternWarning,
						MCPID:       "MCP001", // Assuming MCP can provide haptic feedback
					})
				}
			}
		}
	}()

	go func() {
		ticker := time.NewTicker(5 * time.Second) // Simulate external events every 5 seconds
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				// Simulate a user intent
				if utils.RandomFloat(0, 1) > 0.6 {
					aetherAgent.IntentDeconstructionEngine(models.UserIntent{
						Goal:     "I want to focus",
						Priority: 1,
						Context:  "Work",
						UserID:   "UserAlpha",
					})
				}

				// Simulate a request for DLT logging
				if utils.RandomFloat(0, 1) > 0.7 {
					aetherAgent.ImmutableDLTLogger(models.DLTLogEntry{
						Timestamp: time.Now(),
						EventType: "EnvironmentalAdjustment",
						Data:      map[string]interface{}{"action": "AdjustedLighting", "value": "Warm", "reason": "UserFocusIntent"},
					})
				}

				// Simulate an actuator command for ethical evaluation
				proposedCmd := models.ActuatorCommand{
					MCPID:   "MCP001",
					Command: "SetTemperature",
					Value:   utils.RandomFloat(18.0, 28.0),
				}
				aetherAgent.EthicalConstraintEvaluator(proposedCmd)
			}
		}
	}()

	// Set up OS signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	<-sigChan // Block until a signal is received
	log.Println("Shutting down Aether AI Agent...")
	cancel() // Cancel the context to signal goroutines to stop
	aetherAgent.Stop()
	log.Println("Aether AI Agent stopped.")
}

```

**`config/config.go`**
```go
package config

import (
	"gopkg.in/yaml.v2"
	"os"
	"time"
)

// Config holds the configuration for the Aether AI Agent
type Config struct {
	Agent struct {
		ID              string        `yaml:"id"`
		LogFile         string        `yaml:"log_file"`
		ProcessingInterval time.Duration `yaml:"processing_interval"`
		AnomalyThreshold float64       `yaml:"anomaly_threshold"`
	} `yaml:"agent"`
	MCP struct {
		SerialPort string        `yaml:"serial_port"`
		BaudRate   int           `yaml:"baud_rate"`
		ReadTimeout time.Duration `yaml:"read_timeout"`
	} `yaml:"mcp"`
	DLT struct {
		Enabled       bool   `yaml:"enabled"`
		StoragePath   string `yaml:"storage_path"`
		MaxLogEntries int    `yaml:"max_log_entries"`
	} `yaml:"dlt"`
	Learning struct {
		ModelPath         string        `yaml:"model_path"`
		LearningRate      float64       `yaml:"learning_rate"`
		FederatedInterval time.Duration `yaml:"federated_interval"`
	} `yaml:"learning"`
	Rules struct {
		EthicalRulesPath string `yaml:"ethical_rules_path"`
		BehaviorRulesPath string `yaml:"behavior_rules_path"`
	} `yaml:"rules"`
}

// LoadConfig reads configuration from a YAML file
func LoadConfig(filePath string) (*Config, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	var cfg Config
	err = yaml.Unmarshal(data, &cfg)
	if err != nil {
		return nil, err
	}
	return &cfg, nil
}
```

**`config/config.yaml` (example)**
```yaml
agent:
  id: AetherAlpha01
  log_file: agent.log
  processing_interval: 1s
  anomaly_threshold: 0.15 # Percentage deviation
mcp:
  serial_port: /dev/ttyUSB0 # Or COM3 on Windows
  baud_rate: 115200
  read_timeout: 500ms
dlt:
  enabled: true
  storage_path: ./dlt_logs
  max_log_entries: 1000
learning:
  model_path: ./models
  learning_rate: 0.01
  federated_interval: 5m
rules:
  ethical_rules_path: ./rules/ethical_rules.json
  behavior_rules_path: ./rules/behavior_rules.json
```

**`models/data_models.go`**
```go
package models

import (
	"time"
)

// SensorType defines the type of sensor data
type SensorType string

const (
	Temperature SensorType = "Temperature"
	Humidity    SensorType = "Humidity"
	Light       SensorType = "Light"
	Motion      SensorType = "Motion"
	AirQuality  SensorType = "AirQuality"
	Haptic      SensorType = "Haptic" // For haptic input feedback, if any
	// Add more sensor types as needed
)

// SensorData represents a single reading from a sensor
type SensorData struct {
	Timestamp time.Time  `json:"timestamp"`
	Type      SensorType `json:"type"`
	Value     float64    `json:"value"`
	Unit      string     `json:"unit"`
	Location  string     `json:"location"`
	MCPID     string     `json:"mcp_id"` // Identifier for the MCP unit
	Raw       float64    `json:"raw_value,omitempty"` // Raw value before calibration
}

// ActuatorCommand represents a command to be sent to an MCP actuator
type ActuatorCommand struct {
	Timestamp time.Time   `json:"timestamp"`
	MCPID     string      `json:"mcp_id"`
	Command   string      `json:"command"` // e.g., "SetTemp", "OpenValve", "SetLEDColor"
	Value     interface{} `json:"value"`   // e.g., 22.5, true, "#FF0000"
}

// HapticPattern defines a pattern for haptic feedback
type HapticPatternType string

const (
	PatternWarning  HapticPatternType = "Warning"
	PatternConfirm  HapticPatternType = "Confirm"
	PatternAmbient  HapticPatternType = "Ambient"
	PatternFeedback HapticPatternType = "Feedback"
)

type HapticPattern struct {
	Timestamp   time.Time         `json:"timestamp"`
	MCPID       string            `json:"mcp_id"`
	Intensity   float64           `json:"intensity"`   // 0.0 to 1.0
	DurationMs  int               `json:"duration_ms"` // Milliseconds
	PatternType HapticPatternType `json:"pattern_type"`
}

// UserIntent represents a high-level goal from a user
type UserIntent struct {
	Timestamp time.Time `json:"timestamp"`
	UserID    string    `json:"user_id"`
	Goal      string    `json:"goal"`    // e.g., "I want to focus", "Optimize energy"
	Priority  int       `json:"priority"` // 1-5
	Context   string    `json:"context"` // e.g., "Work", "Relaxing", "Away"
}

// DLTLogEntry represents an entry to be logged to the DLT
type DLTLogEntry struct {
	Timestamp time.Time              `json:"timestamp"`
	EventType string                 `json:"event_type"` // e.g., "SensorAnomaly", "AgentDecision", "CalibrationEvent"
	Data      map[string]interface{} `json:"data"`       // Arbitrary data relevant to the event
	Hash      string                 `json:"hash,omitempty"` // For internal DLT chain
	PrevHash  string                 `json:"prev_hash,omitempty"`
}

// DigitalTwinState represents the current state of the physical environment
type DigitalTwinState struct {
	LastUpdated   time.Time              `json:"last_updated"`
	Environment   map[string]interface{} `json:"environment"` // e.g., {"RoomA_Temp": 22.5, "RoomA_Light": 400}
	ActuatorStates map[string]interface{} `json:"actuator_states"` // e.g., {"HVAC_Mode": "Auto", "Blinds_Pos": 0.5}
	InferredStates map[string]interface{} `json:"inferred_states"` // e.g., {"RoomA_Occupancy": true, "User_Focus": "High"}
}

// LearningModel represents a simplified model for federated learning
// In a real system, this would be more complex (e.g., neural network weights)
type LearningModel struct {
	AgentID   string                 `json:"agent_id"`
	Version   string                 `json:"version"`
	Timestamp time.Time              `json:"timestamp"`
	Parameters map[string]interface{} `json:"parameters"` // e.g., {"temp_bias": 0.5, "light_threshold": 300}
}
```

**`utils/utils.go`**
```go
package utils

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

// RandomFloat generates a random float64 within a given range
func RandomFloat(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

// RandomInt generates a random int within a given range
func RandomInt(min, max int) int {
	return rand.Intn(max-min+1) + min
}

// CalculateSHA256Hash computes the SHA256 hash of a string
func CalculateSHA256Hash(data string) string {
	h := sha256.New()
	h.Write([]byte(data))
	return fmt.Sprintf("%x", h.Sum(nil))
}

// JSONMarshal for consistent JSON marshaling
func JSONMarshal(v interface{}) ([]byte, error) {
	return json.Marshal(v)
}
```

**`agent/agent.go`**
```go
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	"aether-agent/config"
	"aether-agent/models"
	"aether-agent/utils"
)

// AetherAgent is the core structure for the AI agent
type AetherAgent struct {
	ID                 string
	Config             *config.Config
	MCPConn            *MCPInterface // Manages serial communication with MCPs
	sensorDataChan     chan models.SensorData
	actuatorCommandChan chan models.ActuatorCommand
	hapticCommandChan  chan models.HapticPattern
	intentChan         chan models.UserIntent
	dltLogChan         chan models.DLTLogEntry
	stopChan           chan struct{}
	wg                 sync.WaitGroup

	// Internal state and learned models
	currentSensorReadings map[models.SensorType]models.SensorData
	digitalTwin           models.DigitalTwinState
	learnedPatterns       map[string]interface{} // e.g., {"temp_baseline": 25.0, "user_sleep_time": "22:00"}
	ethicalRules          []EthicalRule          // Loaded ethical rules
	dltLedger             []models.DLTLogEntry   // Simple in-memory DLT
	lastAggregatedModel   models.LearningModel   // For federated learning
	mcpHealthStatus       map[string]MCPHealthData // Status for PredictiveSensorHealth
	anomalyBaselines      map[models.SensorType]AnomalyBaseline // For ContextualAnomalyDetection
}

// AnomalyBaseline stores stats for anomaly detection
type AnomalyBaseline struct {
	Mean   float64
	StdDev float64
	Count  int
}

// MCPHealthData stores health metrics for an MCP
type MCPHealthData struct {
	LastContact    time.Time
	ErrorCount     int
	PowerDrawTrend []float64 // Simplified, could be more complex
}

// EthicalRule defines a simple rule for ethical evaluation
type EthicalRule struct {
	Name        string `json:"name"`
	Condition   string `json:"condition"`   // e.g., "temperature > 28 and user_present"
	ActionMatch string `json:"action_match"`// e.g., "SetTemperature"
	Constraint  string `json:"constraint"`  // e.g., "value < 26" or "deny"
	Priority    int    `json:"priority"`
}


// NewAetherAgent initializes a new Aether Agent
func NewAetherAgent(cfg *config.Config) (*AetherAgent, error) {
	mcpIface, err := NewMCPInterface(cfg.MCP.SerialPort, cfg.MCP.BaudRate, cfg.MCP.ReadTimeout)
	if err != nil {
		log.Printf("Could not connect to real serial port, using mock: %v", err)
		mcpIface = NewMockMCPInterface() // Fallback to mock if real connection fails
	}

	agent := &AetherAgent{
		ID:                 cfg.Agent.ID,
		Config:             cfg,
		MCPConn:            mcpIface,
		sensorDataChan:     make(chan models.SensorData, 100),
		actuatorCommandChan: make(chan models.ActuatorCommand, 10),
		hapticCommandChan:  make(chan models.HapticPattern, 10),
		intentChan:         make(chan models.UserIntent, 10),
		dltLogChan:         make(chan models.DLTLogEntry, 10),
		stopChan:           make(chan struct{}),
		currentSensorReadings: make(map[models.SensorType]models.SensorData),
		digitalTwin: models.DigitalTwinState{
			Environment: make(map[string]interface{}),
			ActuatorStates: make(map[string]interface{}),
			InferredStates: make(map[string]interface{}),
		},
		learnedPatterns: make(map[string]interface{}),
		dltLedger:       make([]models.DLTLogEntry, 0),
		mcpHealthStatus: make(map[string]MCPHealthData),
		anomalyBaselines: make(map[models.SensorType]AnomalyBaseline),
	}

	// Load ethical rules
	if err := agent.loadEthicalRules(cfg.Rules.EthicalRulesPath); err != nil {
		log.Printf("Warning: Failed to load ethical rules: %v", err)
	}

	// Initialize DLT if enabled
	if cfg.DLT.Enabled {
		if err := agent.initDLT(); err != nil {
			log.Printf("Warning: Failed to initialize DLT: %v", err)
		}
	}

	log.Printf("Aether Agent %s initialized. MCP Serial: %s", agent.ID, cfg.MCP.SerialPort)
	return agent, nil
}

// Start initiates the agent's main processing loops
func (a *AetherAgent) Start(ctx context.Context) {
	log.Println("Aether Agent core started.")

	// Start MCP communication Goroutine
	a.wg.Add(1)
	go a.IngestSerialSensorData(ctx) // This function now manages reading from serial
	
	// Start processing sensor data, commands, etc.
	a.wg.Add(1)
	go a.processInternalChannels(ctx)

	// Start periodic tasks
	a.wg.Add(1)
	go a.runPeriodicTasks(ctx)

	// Start actuation command sender
	a.wg.Add(1)
	go a.sendActuatorCommands(ctx)

	// Start haptic command sender
	a.wg.Add(1)
	go a.sendHapticCommands(ctx)
}

// Stop gracefully shuts down the agent
func (a *AetherAgent) Stop() {
	close(a.stopChan) // Signal all goroutines to stop
	a.wg.Wait()       // Wait for all goroutines to finish
	a.MCPConn.Close() // Close MCP connection
	log.Println("Aether Agent stopped all goroutines.")
}

// --- Agent Core Loop ---

func (a *AetherAgent) processInternalChannels(ctx context.Context) {
	defer a.wg.Done()
	log.Println("Aether Agent processing channels started.")
	for {
		select {
		case <-ctx.Done():
			log.Println("Processing channels goroutine stopped.")
			return
		case data := <-a.sensorDataChan:
			// Apply dynamic calibration
			calibratedData := a.DynamicSensorCalibration(data)
			a.currentSensorReadings[calibratedData.Type] = calibratedData
			a.DigitalTwinSynchronizer(calibratedData) // Update digital twin
			a.ContextualAnomalyDetection(calibratedData) // Check for anomalies
			a.EphemeralInsightSummarizer(calibratedData) // Extract ephemeral insights
			a.PredictiveSensorHealth(calibratedData) // Update sensor health

			// Example: Cross-modal fusion for combined insights
			if len(a.currentSensorReadings) >= 2 { // Need at least two sensor types
				_ = a.CrossModalPerceptionFusion()
			}

		case cmd := <-a.actuatorCommandChan:
			a.ExecuteAdaptiveActuation(cmd) // Send command to MCP
			a.ImmutableDLTLogger(models.DLTLogEntry{ // Log actions
				EventType: "ActuatorCommand",
				Data:      map[string]interface{}{"mcp_id": cmd.MCPID, "command": cmd.Command, "value": cmd.Value},
			})
			a.ExplainableActionRationale("ActuatorCommand:" + cmd.Command) // Explain action

		case hapticPattern := <-a.hapticCommandChan:
			a.MCPConn.Write(hapticPattern.MCPID, hapticPattern) // Send haptic command to MCP
			a.ImmutableDLTLogger(models.DLTLogEntry{ // Log haptic feedback
				EventType: "HapticFeedbackSent",
				Data:      map[string]interface{}{"mcp_id": hapticPattern.MCPID, "pattern_type": hapticPattern.PatternType},
			})

		case intent := <-a.intentChan:
			a.IntentDeconstructionEngine(intent) // Process user intent
			a.ProactiveNudgingEngine(ctx) // Trigger nudging based on intent/context

		case logEntry := <-a.dltLogChan:
			a.addDLTEntry(logEntry) // Add to DLT ledger
		}
	}
}

func (a *AetherAgent) runPeriodicTasks(ctx context.Context) {
	defer a.wg.Done()
	log.Println("Aether Agent periodic tasks started.")

	// Define periodic intervals
	learningTicker := time.NewTicker(a.Config.Agent.ProcessingInterval * 5) // e.g., every 5s for learning
	federatedTicker := time.NewTicker(a.Config.Learning.FederatedInterval) // e.g., every 5m for federated learning
	resourceTicker := time.NewTicker(a.Config.Agent.ProcessingInterval * 2) // e.g., every 2s for resource allocation
	selfHealingTicker := time.NewTicker(a.Config.Agent.ProcessingInterval * 10) // e.g., every 10s for self-healing

	defer learningTicker.Stop()
	defer federatedTicker.Stop()
	defer resourceTicker.Stop()
	defer selfHealingTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("Periodic tasks goroutine stopped.")
			return
		case <-learningTicker.C:
			a.ContextualBehavioralLearning()
			a.ReinforcementPolicyOptimizer()
			a.GenerativeScenarioAugmentor()
		case <-federatedTicker.C:
			// Simulate sending a model update (simplified)
			localModel := models.LearningModel{
				AgentID: a.ID,
				Version: fmt.Sprintf("%d", time.Now().Unix()),
				Parameters: map[string]interface{}{
					"temp_bias": utils.RandomFloat(-0.5, 0.5),
					"humidity_corr": utils.RandomFloat(0.1, 0.9),
				},
			}
			a.SecureFederatedModelAggregator(localModel)
		case <-resourceTicker.C:
			a.DynamicResourceAllocator()
		case <-selfHealingTicker.C:
			a.AutonomousSelfHealing()
		}
	}
}

func (a *AetherAgent) sendActuatorCommands(ctx context.Context) {
	defer a.wg.Done()
	log.Println("Aether Agent actuator command sender started.")
	for {
		select {
		case <-ctx.Done():
			log.Println("Actuator command sender goroutine stopped.")
			return
		case cmd := <-a.actuatorCommandChan:
			err := a.MCPConn.Write(cmd.MCPID, cmd)
			if err != nil {
				log.Printf("Error sending actuator command to MCP %s: %v", cmd.MCPID, err)
			} else {
				log.Printf("Sent actuator command to MCP %s: %s -> %v", cmd.MCPID, cmd.Command, cmd.Value)
			}
		}
	}
}

func (a *AetherAgent) sendHapticCommands(ctx context.Context) {
	defer a.wg.Done()
	log.Println("Aether Agent haptic command sender started.")
	for {
		select {
		case <-ctx.Done():
			log.Println("Haptic command sender goroutine stopped.")
			return
		case haptic := <-a.hapticCommandChan:
			err := a.MCPConn.Write(haptic.MCPID, haptic)
			if err != nil {
				log.Printf("Error sending haptic command to MCP %s: %v", haptic.MCPID, err)
			} else {
				log.Printf("Sent haptic feedback to MCP %s: %s (intensity %.2f)", haptic.MCPID, haptic.PatternType, haptic.Intensity)
			}
		}
	}
}


// IngestSensorData is a public method to push sensor data into the agent, e.g., from serial reader or mock.
func (a *AetherAgent) IngestSensorData(data models.SensorData) {
	select {
	case a.sensorDataChan <- data:
	default:
		log.Println("Sensor data channel full, dropping data.")
	}
}

// IngestUserIntent is a public method to push user intents into the agent.
func (a *AetherAgent) IngestUserIntent(intent models.UserIntent) {
	select {
	case a.intentChan <- intent:
	default:
		log.Println("Intent channel full, dropping intent.")
	}
}

// RequestHapticFeedback is a public method to request haptic feedback.
func (a *AetherAgent) RequestHapticFeedback(pattern models.HapticPattern) {
	select {
	case a.hapticCommandChan <- pattern:
	default:
		log.Println("Haptic command channel full, dropping request.")
	}
}

// RequestDLTLog is a public method to request a DLT log entry.
func (a *AetherAgent) RequestDLTLog(entry models.DLTLogEntry) {
	select {
	case a.dltLogChan <- entry:
	default:
		log.Println("DLT log channel full, dropping entry.")
	}
}


// --- Function Implementations (Refer to the outline for summaries) ---

// A. Core Perception & MCP Interface

// 1. IngestSerialSensorData
func (a *AetherAgent) IngestSerialSensorData(ctx context.Context) {
	defer a.wg.Done()
	log.Println("IngestSerialSensorData started.")

	// Mock serial reader logic. In a real scenario, this would read from a.MCPConn.Reader
	// and parse actual bytes into models.SensorData
	ticker := time.NewTicker(a.Config.MCP.ReadTimeout)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("IngestSerialSensorData stopped.")
			return
		case <-ticker.C:
			if a.MCPConn.IsConnected() {
				// Simulate reading from MCP (MCPConn.Read would be implemented to read actual serial data)
				// For this example, we're not actually reading; the main loop simulates data.
				// This function would typically look like:
				// dataBytes, err := a.MCPConn.Read()
				// if err == nil && len(dataBytes) > 0 {
				//     var sensorData models.SensorData
				//     if err := json.Unmarshal(dataBytes, &sensorData); err == nil {
				//         a.IngestSensorData(sensorData)
				//     } else {
				//         log.Printf("Failed to unmarshal sensor data: %v", err)
				//     }
				// }
				// For the example, we just keep this goroutine alive.
			} else {
				log.Println("MCP not connected for reading (mock behavior).")
			}
		}
	}
}

// 2. DynamicSensorCalibration
func (a *AetherAgent) DynamicSensorCalibration(data models.SensorData) models.SensorData {
	// Simplified calibration: apply a fixed offset and a random drift.
	// In a real system: Kalman filter, polynomial regression based on historical data.
	calibratedValue := data.Raw // Start with raw value
	offset := a.learnedPatterns[fmt.Sprintf("cal_offset_%s_%s", data.MCPID, data.Type)]
	drift := a.learnedPatterns[fmt.Sprintf("cal_drift_%s_%s", data.MCPID, data.Type)]

	if offsetVal, ok := offset.(float64); ok {
		calibratedValue += offsetVal
	} else {
		// Initialize or set default offset
		a.learnedPatterns[fmt.Sprintf("cal_offset_%s_%s", data.MCPID, data.Type)] = utils.RandomFloat(-0.1, 0.1)
	}

	if driftVal, ok := drift.(float64); ok {
		calibratedValue += driftVal * float64(time.Since(data.Timestamp).Seconds()/3600) // Drift over time (hourly)
	} else {
		// Initialize or set default drift
		a.learnedPatterns[fmt.Sprintf("cal_drift_%s_%s", data.MCPID, data.Type)] = utils.RandomFloat(-0.01, 0.01)
	}

	data.Value = calibratedValue // Update with calibrated value
	log.Printf("Calibrated %s (Raw: %.2f -> Calibrated: %.2f)", data.Type, data.Raw, data.Value)
	return data
}

// 3. CrossModalPerceptionFusion
func (a *AetherAgent) CrossModalPerceptionFusion() map[string]interface{} {
	fusedState := make(map[string]interface{})

	temp, tempOK := a.currentSensorReadings[models.Temperature]
	humidity, humOK := a.currentSensorReadings[models.Humidity]
	light, lightOK := a.currentSensorReadings[models.Light]

	if tempOK && humOK && lightOK {
		// Example fusion: calculate perceived comfort index (simplified)
		comfortIndex := (temp.Value*0.6 + humidity.Value*0.05) - (light.Value / 1000.0) // Arbitrary formula
		fusedState["PerceivedComfortIndex"] = fmt.Sprintf("%.2f", comfortIndex)

		// Example fusion: infer occupancy based on light and temperature stability
		isOccupied := false
		if light.Value > 100 && temp.Value > 20 && temp.Value < 28 { // Basic heuristics
			isOccupied = true
		}
		fusedState["InferredOccupancy"] = isOccupied

		log.Printf("Cross-Modal Fusion: Comfort Index: %s, Occupancy: %t", fusedState["PerceivedComfortIndex"], fusedState["InferredOccupancy"])
	} else {
		log.Println("Not enough sensor data for full Cross-Modal Fusion.")
	}
	return fusedState
}

// 4. GenerateHapticFeedback (This is called by RequestHapticFeedback)
// The function primarily puts the request on a channel. The `sendHapticCommands` goroutine handles sending to MCP.
// The core logic for *deciding* to generate haptic feedback might live in other functions (e.g., Anomaly Detection).
func (a *AetherAgent) GenerateHapticFeedback(pattern models.HapticPattern) {
	log.Printf("Agent deciding to generate haptic feedback: Type %s, Intensity %.2f", pattern.PatternType, pattern.Intensity)
	a.RequestHapticFeedback(pattern) // Puts it on the channel for sending
}

// 5. PredictiveSensorHealth
func (a *AetherAgent) PredictiveSensorHealth(data models.SensorData) {
	mcpID := data.MCPID
	if _, exists := a.mcpHealthStatus[mcpID]; !exists {
		a.mcpHealthStatus[mcpID] = MCPHealthData{
			LastContact: time.Now(),
			ErrorCount:  0,
			PowerDrawTrend: []float64{utils.RandomFloat(0.1, 0.5)}, // Simulate initial power draw
		}
	}

	status := a.mcpHealthStatus[mcpID]
	status.LastContact = time.Now()
	
	// Simulate anomaly if power draw goes out of range (very simplified)
	currentPower := utils.RandomFloat(0.1, 0.5) // Simulate current power draw
	status.PowerDrawTrend = append(status.PowerDrawTrend, currentPower)
	if len(status.PowerDrawTrend) > 10 { // Keep last 10 readings
		status.PowerDrawTrend = status.PowerDrawTrend[1:]
	}

	avgPower := 0.0
	for _, p := range status.PowerDrawTrend {
		avgPower += p
	}
	avgPower /= float64(len(status.PowerDrawTrend))

	// Simple prediction: if average power deviates significantly
	if avgPower > 0.6 || avgPower < 0.15 { // Arbitrary thresholds for high/low power
		log.Printf("WARNING: MCP %s (Sensor %s) shows abnormal power draw (avg %.2f). Possible degradation!", mcpID, data.Type, avgPower)
		a.RequestDLTLog(models.DLTLogEntry{
			EventType: "MCPHealthWarning",
			Data:      map[string]interface{}{"mcp_id": mcpID, "reason": "AbnormalPowerDraw", "avg_power": avgPower},
		})
	}
	a.mcpHealthStatus[mcpID] = status
}


// B. Reasoning & Decision-Making Engine

// 6. ContextualAnomalyDetection
func (a *AetherAgent) ContextualAnomalyDetection(data models.SensorData) {
	baseline, exists := a.anomalyBaselines[data.Type]
	if !exists || baseline.Count < 5 { // Not enough data to establish a robust baseline
		// Simple incremental update for mean and std dev
		a.anomalyBaselines[data.Type] = AnomalyBaseline{
			Mean:   (baseline.Mean*float64(baseline.Count) + data.Value) / (float64(baseline.Count) + 1),
			StdDev: baseline.StdDev, // More complex to update std dev incrementally accurately
			Count:  baseline.Count + 1,
		}
		if baseline.Count == 4 { // After 5 data points, calculate initial std dev
			// This would involve storing the last few values, omitted for brevity.
			log.Printf("Established initial baseline for %s: Mean %.2f", data.Type, a.anomalyBaselines[data.Type].Mean)
		}
		return
	}

	deviation := (data.Value - baseline.Mean) / (baseline.StdDev + 0.01) // Add small epsilon to prevent division by zero
	if deviation > a.Config.Agent.AnomalyThreshold || deviation < -a.Config.Agent.AnomalyThreshold {
		log.Printf("ANOMALY DETECTED for %s (%.2f): Deviation %.2f. Context: %s",
			data.Type, data.Value, deviation, a.digitalTwin.InferredStates["InferredOccupancy"])
		a.GenerateHapticFeedback(models.HapticPattern{
			MCPID:       data.MCPID,
			Intensity:   0.8,
			DurationMs:  1000,
			PatternType: models.PatternWarning,
		})
		a.RequestDLTLog(models.DLTLogEntry{
			EventType: "ContextualAnomaly",
			Data:      map[string]interface{}{"sensor": data.Type, "value": data.Value, "deviation": deviation, "context": a.digitalTwin.InferredStates},
		})
	}

	// Update baseline (simplified: using new value directly, in reality, use moving average/window)
	baseline.Mean = (baseline.Mean*0.9 + data.Value*0.1) // Exponential moving average
	// Update std dev (more complex, requires variance calculation, omitted for brevity)
	a.anomalyBaselines[data.Type] = baseline
}

// 7. ProactiveNudgingEngine
func (a *AetherAgent) ProactiveNudgingEngine(ctx context.Context) {
	userIntent, _ := a.learnedPatterns["currentUserIntent"].(models.UserIntent)
	comfortIndex, _ := a.digitalTwin.InferredStates["PerceivedComfortIndex"].(string) // from CrossModalFusion

	if userIntent.Goal == "I want to focus" {
		if temp, ok := a.currentSensorReadings[models.Temperature]; ok && temp.Value > 25 {
			log.Printf("Nudging: High temp (%.1fC) detected, adjusting fan for user focus.", temp.Value)
			a.actuatorCommandChan <- models.ActuatorCommand{
				MCPID:   "MCP001", // Assuming an MCP controls fans
				Command: "SetFanSpeed",
				Value:   0.7, // 70% speed
			}
			a.ExplainableActionRationale("Nudge: Fan adjusted for focus due to high temperature.")
		}
	} else if userIntent.Goal == "Optimize energy savings" {
		if occupied, ok := a.digitalTwin.InferredStates["InferredOccupancy"].(bool); ok && !occupied {
			if temp, ok := a.currentSensorReadings[models.Temperature]; ok && temp.Value < 20 {
				log.Println("Nudging: Room unoccupied and cool, lowering heating for energy savings.")
				a.actuatorCommandChan <- models.ActuatorCommand{
					MCPID:   "MCP001",
					Command: "SetHeating",
					Value:   18.0, // Lower setpoint
				}
				a.ExplainableActionRationale("Nudge: Heating lowered for energy savings as room is unoccupied.")
			}
		}
	} else {
		log.Printf("No specific proactive nudge for current intent '%s' or comfort index '%s'.", userIntent.Goal, comfortIndex)
	}
}

// 8. IntentDeconstructionEngine
func (a *AetherAgent) IntentDeconstructionEngine(intent models.UserIntent) {
	log.Printf("Deconstructing user intent: '%s' from %s", intent.Goal, intent.UserID)
	a.learnedPatterns["currentUserIntent"] = intent // Store for nudging

	switch intent.Goal {
	case "I want to focus":
		log.Println("Translating 'focus' intent: Adjusting lighting, disabling distractions.")
		a.actuatorCommandChan <- models.ActuatorCommand{
			MCPID:   "MCP002", // Assuming light control MCP
			Command: "SetLightColor",
			Value:   "#FFFACD", // Warm white
		}
		a.actuatorCommandChan <- models.ActuatorCommand{
			MCPID:   "MCP003", // Assuming sound control MCP
			Command: "SetNoiseCancelling",
			Value:   true,
		}
	case "Optimize for energy savings":
		log.Println("Translating 'energy savings' intent: Setting broader temperature ranges, optimizing ventilation.")
		a.actuatorCommandChan <- models.ActuatorCommand{
			MCPID:   "MCP001", // HVAC MCP
			Command: "SetTempRange",
			Value:   map[string]float64{"min": 18.0, "max": 28.0},
		}
	case "Prepare for sleep":
		log.Println("Translating 'sleep' intent: Dimming lights, cooling room, playing calming sounds.")
		a.actuatorCommandChan <- models.ActuatorCommand{
			MCPID:   "MCP002",
			Command: "SetLightIntensity",
			Value:   0.1,
		}
		a.actuatorCommandChan <- models.ActuatorCommand{
			MCPID:   "MCP001",
			Command: "SetTemperature",
			Value:   20.0,
		}
	default:
		log.Printf("Intent '%s' not recognized for direct translation.", intent.Goal)
	}
	a.ExplainableActionRationale("IntentTranslated:" + intent.Goal)
}

// 9. EthicalConstraintEvaluator
func (a *AetherAgent) EthicalConstraintEvaluator(proposedAction models.ActuatorCommand) models.ActuatorCommand {
	log.Printf("Evaluating proposed action: %s to %v for ethical constraints.", proposedAction.Command, proposedAction.Value)

	for _, rule := range a.ethicalRules {
		if rule.ActionMatch == proposedAction.Command {
			// Simplified condition evaluation
			if rule.Condition == "temperature > 28 and user_present" { // Example condition
				userPresent := a.digitalTwin.InferredStates["InferredOccupancy"].(bool)
				if temp, ok := a.currentSensorReadings[models.Temperature]; ok && temp.Value > 28 && userPresent {
					if rule.Constraint == "value < 26" {
						// Override value to stay within ethical bounds (e.g., don't make it too cold too fast)
						if proposedAction.Value.(float64) < 26.0 { // Assuming value is float64 for temp
							log.Printf("ETHICAL VIOLATION PREVENTED: Action %s to %.1f overridden to 26.0 (Rule: %s)",
								proposedAction.Command, proposedAction.Value, rule.Name)
							proposedAction.Value = 26.0
						}
					} else if rule.Constraint == "deny" {
						log.Printf("ETHICAL VIOLATION PREVENTED: Action %s denied (Rule: %s)", proposedAction.Command, rule.Name)
						return models.ActuatorCommand{} // Return empty command to deny
					}
				}
			}
		}
	}
	log.Printf("Proposed action cleared ethical evaluation.")
	return proposedAction
}

func (a *AetherAgent) loadEthicalRules(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("could not read ethical rules file: %w", err)
	}
	if err := json.Unmarshal(data, &a.ethicalRules); err != nil {
		return fmt.Errorf("could not unmarshal ethical rules: %w", err)
	}
	log.Printf("Loaded %d ethical rules.", len(a.ethicalRules))
	return nil
}

// 10. EphemeralInsightSummarizer
func (a *AetherAgent) EphemeralInsightSummarizer(data models.SensorData) {
	// Simple example: report significant changes in air quality or light levels
	if data.Type == models.AirQuality { // Assuming AirQuality sensor
		// Placeholder: In a real scenario, compare to last reading in a short-term window
		// and report if change > X%.
		lastAQ, ok := a.learnedPatterns["last_air_quality"].(models.SensorData)
		if ok && time.Since(lastAQ.Timestamp) < 5*time.Minute {
			change := (data.Value - lastAQ.Value) / lastAQ.Value
			if change > 0.2 { // 20% increase
				log.Printf("EPHEMERAL INSIGHT: Air quality dropped significantly by %.1f%% in %s!", change*100, data.Location)
				a.GenerateHapticFeedback(models.HapticPattern{
					MCPID: data.MCPID, Intensity: 0.6, DurationMs: 700, PatternType: models.PatternWarning,
				})
			}
		}
		a.learnedPatterns["last_air_quality"] = data
	} else if data.Type == models.Light {
		// Similar logic for rapid light changes
	}
}


// C. Action, Control & Self-Management

// 11. ExecuteAdaptiveActuation
// This function directly puts command onto channel, the `sendActuatorCommands` goroutine sends to MCP
func (a *AetherAgent) ExecuteAdaptiveActuation(command models.ActuatorCommand) {
	// In a real system, might check current state from DigitalTwin or get immediate feedback
	// to adapt output (e.g., if motor doesn't reach target speed, increase power).
	log.Printf("Executing adaptive actuation: MCP %s, Command %s, Value %v", command.MCPID, command.Command, command.Value)
	a.actuatorCommandChan <- command // Send to the channel for processing
}

// 12. DynamicResourceAllocator
func (a *AetherAgent) DynamicResourceAllocator() {
	// Simplified: Prioritize MCPs reporting anomalies or critical sensor data.
	// In a real system: monitor CPU/memory usage, network bandwidth, battery levels of MCPs.

	highPriorityMCPs := make(map[string]bool)
	for mcpID, health := range a.mcpHealthStatus {
		if health.ErrorCount > 0 || (len(health.PowerDrawTrend) > 0 && health.PowerDrawTrend[len(health.PowerDrawTrend)-1] > 0.6) {
			highPriorityMCPs[mcpID] = true
		}
	}

	for mcpID := range a.MCPConn.ConnectedMCPs() { // Iterate over connected MCPs from the interface
		if highPriorityMCPs[mcpID] {
			log.Printf("Allocating high priority resources to MCP %s (e.g., faster polling, more bandwidth).", mcpID)
			// Simulate sending resource adjustment commands to MCPs or internal modules
		} else {
			// log.Printf("Allocating standard resources to MCP %s.", mcpID)
		}
	}
	// Also adjust internal agent resources (goroutines, channel sizes)
	// Example: If many anomalies, dedicate more goroutines to anomaly detection.
}

// 13. AutonomousSelfHealing
func (a *AetherAgent) AutonomousSelfHealing() {
	// Check MCP health status
	for mcpID, health := range a.mcpHealthStatus {
		if time.Since(health.LastContact) > 10*time.Second { // No contact for 10 seconds
			log.Printf("SELF-HEALING: MCP %s unresponsive for too long. Attempting restart/reconnection.", mcpID)
			a.MCPConn.RestartMCP(mcpID) // Simulate a restart command
			a.RequestDLTLog(models.DLTLogEntry{
				EventType: "SelfHealingAction",
				Data:      map[string]interface{}{"mcp_id": mcpID, "action": "RestartAttempt", "reason": "Unresponsive"},
			})
			a.mcpHealthStatus[mcpID] = MCPHealthData{
				LastContact: time.Now(),
				ErrorCount:  health.ErrorCount + 1, // Increment error count
			}
		} else if health.ErrorCount > 3 { // Too many errors
			log.Printf("SELF-HEALING: MCP %s reporting too many errors. Initiating recalibration.", mcpID)
			a.MCPConn.SendGenericCommand(mcpID, "CalibrateAllSensors") // Simulate recalibration command
			a.RequestDLTLog(models.DLTLogEntry{
				EventType: "SelfHealingAction",
				Data:      map[string]interface{}{"mcp_id": mcpID, "action": "Recalibration", "reason": "HighErrorCount"},
			})
			status := a.mcpHealthStatus[mcpID]
			status.ErrorCount = 0 // Reset error count after action
			a.mcpHealthStatus[mcpID] = status
		}
	}
}

// 14. RealtimeDigitalTwinSynchronization
func (a *AetherAgent) DigitalTwinSynchronizer(data models.SensorData) {
	a.digitalTwin.LastUpdated = time.Now()
	key := fmt.Sprintf("%s_%s", data.Location, data.Type)
	a.digitalTwin.Environment[key] = data.Value

	// Update inferred states (simplified example from CrossModalFusion)
	fusedState := a.CrossModalPerceptionFusion()
	for k, v := range fusedState {
		a.digitalTwin.InferredStates[k] = v
	}

	// For actuators, assume the command sent is the new state
	if data.Type == models.Temperature { // Example: infer HVAC state from temperature changes
		if val, ok := a.digitalTwin.ActuatorStates["HVAC_TargetTemp"].(float64); ok {
			if data.Value > val+1 { // Temp is higher than target
				a.digitalTwin.ActuatorStates["HVAC_Mode"] = "Cooling"
			} else if data.Value < val-1 { // Temp is lower than target
				a.digitalTwin.ActuatorStates["HVAC_Mode"] = "Heating"
			} else {
				a.digitalTwin.ActuatorStates["HVAC_Mode"] = "Idle"
			}
		} else {
			a.digitalTwin.ActuatorStates["HVAC_TargetTemp"] = data.Value // Default
			a.digitalTwin.ActuatorStates["HVAC_Mode"] = "Idle"
		}
	}

	// log.Printf("Digital Twin updated: %v", a.digitalTwin.Environment)
}

// 15. ExplainableActionRationale
func (a *AetherAgent) ExplainableActionRationale(actionID string) string {
	rationale := fmt.Sprintf("Action '%s' taken because: ", actionID)

	switch actionID {
	case "IntentTranslated:I want to focus":
		rationale += fmt.Sprintf("User expressed intent to focus. Agent adjusted lighting to %v (current: %s) and enabled noise cancelling (current: %s).",
			a.digitalTwin.ActuatorStates["Light_Color"],
			a.currentSensorReadings[models.Light].Value,
			a.digitalTwin.ActuatorStates["NoiseCancelling_Status"],
		)
	case "Nudge: Fan adjusted for focus due to high temperature.":
		rationale += fmt.Sprintf("High temperature detected (%.1fC, current: %.1fC) while user is in 'focus' mode. Fan speed set to %.1f to improve comfort and focus.",
			a.currentSensorReadings[models.Temperature].Value,
			a.currentSensorReadings[models.Temperature].Value,
			a.digitalTwin.ActuatorStates["Fan_Speed"],
		)
	case "ActuatorCommand:SetTemperature":
		// Example explanation for a temperature set command
		val, _ := a.digitalTwin.ActuatorStates["HVAC_TargetTemp"].(float64)
		mode, _ := a.digitalTwin.ActuatorStates["HVAC_Mode"].(string)
		rationale += fmt.Sprintf("HVAC target temperature set to %.1fC. Current mode: %s. This was likely triggered by a user intent or environmental optimization.",
			val, mode,
		)
	default:
		rationale += "Specific rationale not available or action is routine."
	}
	log.Println("RATIONALE:", rationale)
	return rationale
}


// D. Learning, Adaptation & Advanced Concepts

// 16. ContextualBehavioralLearning
func (a *AetherAgent) ContextualBehavioralLearning() {
	// Simplified: Learn preferred temperature for "focus" context.
	// In a real system: Complex pattern recognition on time series data, clustering.
	if intent, ok := a.learnedPatterns["currentUserIntent"].(models.UserIntent); ok && intent.Goal == "I want to focus" {
		if temp, ok := a.currentSensorReadings[models.Temperature]; ok {
			// Simulate updating a "preferred temperature for focus"
			currentPreferredTemp, exists := a.learnedPatterns["preferred_temp_focus"].(float64)
			if !exists {
				currentPreferredTemp = temp.Value // Initialize
			}
			// Moving average update for preferred temp
			a.learnedPatterns["preferred_temp_focus"] = currentPreferredTemp*0.9 + temp.Value*0.1
			log.Printf("Behavioral Learning: Updated preferred temperature for 'focus' to %.2fC", a.learnedPatterns["preferred_temp_focus"])
		}
	}
	// Learn occupancy patterns based on motion sensor data over time
	// Example: a.learnedPatterns["occupancy_heatmap"] = update_heatmap(a.currentSensorReadings[models.Motion])
}

// 17. ReinforcementPolicyOptimizer
func (a *AetherAgent) ReinforcementPolicyOptimizer() {
	// Simplified Q-learning like update for optimizing fan speed for comfort.
	// State: (Current Temperature, Perceived Comfort Index)
	// Action: (Increase Fan, Decrease Fan, No Change)
	// Reward: (Comfort index increase: +1, decrease: -1)

	// In reality, this requires a defined state space, action space, and reward function.
	// For this example, we just simulate updating a policy rule based on recent outcomes.
	// Assume an initial policy: "If temp > 25, increase fan."
	// If, after increasing fan, comfort index improves, reinforce this rule. If not, penalize.

	log.Println("Reinforcement Policy Optimizer: Simulating update for fan control policy.")
	currentTemp, tempOK := a.currentSensorReadings[models.Temperature]
	comfortIndex, comfortOK := a.digitalTwin.InferredStates["PerceivedComfortIndex"].(string) // Assume this is a float string

	if tempOK && comfortOK {
		currentComfort, _ := utils.ParseFloat(comfortIndex)
		// Assume an action was taken (e.g., fan increased/decreased) and evaluate its effect.
		// This requires tracking previous state and action.
		log.Printf("Simulated RL update. Current temp: %.1f, Comfort: %.2f", currentTemp.Value, currentComfort)
		// Example: If fan was increased and comfort improved, update internal model.
		// a.learnedPatterns["fan_policy"] = updatedPolicy
	}
}

// 18. SecureFederatedModelAggregator
func (a *AetherAgent) SecureFederatedModelAggregator(localModelUpdate models.LearningModel) {
	log.Printf("Federated Learning: Received local model update from Agent %s (Version: %s).", localModelUpdate.AgentID, localModelUpdate.Version)

	// In a real federated learning setup:
	// 1. Encrypt and send localModelUpdate to a central aggregator (or other agents in peer-to-peer).
	// 2. The aggregator would securely combine (e.g., weighted average) multiple updates.
	// 3. The new global model would be sent back to agents.

	// For this example, we simulate aggregation by just averaging a simple parameter
	// and storing it as the 'last aggregated model'.
	if a.lastAggregatedModel.Parameters == nil {
		a.lastAggregatedModel = localModelUpdate // First model becomes the base
	} else {
		// Simulate weighted average for a simple parameter
		if oldTempBias, ok := a.lastAggregatedModel.Parameters["temp_bias"].(float64); ok {
			if newTempBias, ok := localModelUpdate.Parameters["temp_bias"].(float64); ok {
				// Simple average for demonstration
				a.lastAggregatedModel.Parameters["temp_bias"] = (oldTempBias + newTempBias) / 2.0
			}
		}
	}
	log.Printf("Federated Learning: Aggregated global model (simplified temp_bias: %.2f).", a.lastAggregatedModel.Parameters["temp_bias"])
	// Agents would then download and apply this aggregated model
}

// 19. GenerativeScenarioAugmentor
func (a *AetherAgent) GenerativeScenarioAugmentor() {
	// Simplified: Generate a synthetic "high temperature spike" scenario.
	// In a real system: Use Generative Adversarial Networks (GANs) or Variational Autoencoders (VAEs)
	// trained on real sensor data to create realistic, novel data sequences.

	log.Println("Generative Scenario Augmentor: Creating a synthetic 'temperature spike' scenario.")

	// Base temperature from learned patterns
	baseTemp, ok := a.learnedPatterns["preferred_temp_focus"].(float64)
	if !ok {
		baseTemp = 24.0 // Default
	}

	// Generate a sequence of synthetic data for a simulated event
	scenario := []models.SensorData{}
	startTime := time.Now().Add(-1 * time.Hour) // Scenario starting an hour ago

	for i := 0; i < 60; i++ { // Simulate 60 minutes of data
		currentTime := startTime.Add(time.Duration(i) * time.Minute)
		tempValue := baseTemp + utils.RandomFloat(-1.0, 1.0) // Baseline noise
		if i > 20 && i < 40 { // Introduce a spike
			tempValue += utils.RandomFloat(5.0, 10.0) // 5-10 degree spike
		}
		scenario = append(scenario, models.SensorData{
			Timestamp: currentTime,
			Type:      models.Temperature,
			Value:     tempValue,
			Unit:      "°C",
			Location:  "SyntheticRoom",
			MCPID:     "MCP_SIM",
			Raw:       tempValue, // For simplicity, raw equals value
		})
	}
	// This generated `scenario` could then be fed into the agent for testing/training.
	// For example: a.TestWithScenario(scenario)
	log.Printf("Generated %d synthetic sensor data points for scenario testing.", len(scenario))
}

// 20. ImmutableDLTLogger
func (a *AetherAgent) ImmutableDLTLogger(logEntry models.DLTLogEntry) {
	// Simplified DLT: an in-memory linked list of hashes.
	// In a real system: integration with a lightweight blockchain client (e.g., Fabric-SDK, Go-Ethereum).

	marshalledData, err := utils.JSONMarshal(logEntry.Data)
	if err != nil {
		log.Printf("Error marshaling DLT log data: %v", err)
		return
	}
	dataToHash := fmt.Sprintf("%s%s%s", logEntry.Timestamp.String(), logEntry.EventType, string(marshalledData))

	if len(a.dltLedger) > 0 {
		lastEntry := a.dltLedger[len(a.dltLedger)-1]
		logEntry.PrevHash = lastEntry.Hash
	} else {
		logEntry.PrevHash = "0000000000000000000000000000000000000000000000000000000000000000" // Genesis hash
	}

	logEntry.Hash = utils.CalculateSHA256Hash(logEntry.PrevHash + dataToHash)

	a.dltLedger = append(a.dltLedger, logEntry)
	log.Printf("DLT Logger: Logged event '%s' with hash: %s", logEntry.EventType, logEntry.Hash[:8])

	// Prune old entries if ledger grows too large
	if len(a.dltLedger) > a.Config.DLT.MaxLogEntries {
		a.dltLedger = a.dltLedger[len(a.dltLedger)-a.Config.DLT.MaxLogEntries:]
	}
}

// Helper to initialize DLT
func (a *AetherAgent) initDLT() error {
	// Load existing DLT logs from file if any, or create an empty ledger
	// For this example, it's just in-memory.
	log.Println("Initialized in-memory DLT ledger.")
	return nil
}

// Placeholder to convert string to float for example
func (utils *utils) ParseFloat(s string) (float64, error) {
	var f float64
	_, err := fmt.Sscanf(s, "%f", &f)
	return f, err
}

```

**`agent/mcp_interface.go`**
```go
package agent

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"sync"
	"time"

	"aether-agent/models"

	"github.com/tarm/serial" // For actual serial communication
)

// MCPInterface defines the contract for communicating with MCPs
type MCPInterface interface {
	Connect() error
	Close() error
	Read() ([]byte, error) // Reads raw data from MCP
	Write(mcpID string, data interface{}) error // Writes structured data to a specific MCP
	IsConnected() bool
	RestartMCP(mcpID string) error // Simulate restarting an MCP
	SendGenericCommand(mcpID, command string) error
	ConnectedMCPs() []string // Returns a list of currently connected MCP IDs
}

// SerialMCPInterface implements MCPInterface for actual serial communication
type SerialMCPInterface struct {
	port *serial.Port
	config *serial.Config
	readBuf []byte
	connected bool
	mu sync.Mutex // Protects access to serial port
	knownMCPs map[string]bool // Simplified list of MCPs we might communicate with
}

// NewMCPInterface creates a new SerialMCPInterface
func NewMCPInterface(portName string, baudRate int, readTimeout time.Duration) (*SerialMCPInterface, error) {
	cfg := &serial.Config{
		Name:        portName,
		Baud:        baudRate,
		ReadTimeout: readTimeout,
	}
	s, err := serial.OpenPort(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to open serial port %s: %w", portName, err)
	}

	log.Printf("Connected to serial port %s at %d baud.", portName, baudRate)
	return &SerialMCPInterface{
		port: s,
		config: cfg,
		readBuf: make([]byte, 2048), // Buffer for reading
		connected: true,
		knownMCPs: make(map[string]bool),
	}, nil
}

// Connect implements MCPInterface.Connect (already done in NewMCPInterface)
func (s *SerialMCPInterface) Connect() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.connected && s.port != nil {
		return nil // Already connected
	}
	newPort, err := serial.OpenPort(s.config)
	if err != nil {
		return fmt.Errorf("failed to reconnect serial port: %w", err)
	}
	s.port = newPort
	s.connected = true
	log.Printf("Reconnected to serial port %s.", s.config.Name)
	return nil
}

// Close implements MCPInterface.Close
func (s *SerialMCPInterface) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.port == nil {
		return nil
	}
	err := s.port.Close()
	s.connected = false
	s.port = nil
	log.Printf("Closed serial port %s.", s.config.Name)
	return err
}

// Read implements MCPInterface.Read
// This would need a robust protocol parser for real data.
func (s *SerialMCPInterface) Read() ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.connected || s.port == nil {
		return nil, fmt.Errorf("serial port not connected")
	}

	n, err := s.port.Read(s.readBuf)
	if err != nil {
		if err == io.EOF { // No data available within timeout
			return nil, nil
		}
		return nil, fmt.Errorf("failed to read from serial port: %w", err)
	}
	if n > 0 {
		// In a real system, you'd parse this data, e.g., JSON per line.
		// For this mock, assume it's valid JSON for a SensorData.
		log.Printf("Read %d bytes from serial: %s", n, string(s.readBuf[:n]))
		return s.readBuf[:n], nil
	}
	return nil, nil
}

// Write implements MCPInterface.Write
func (s *SerialMCPInterface) Write(mcpID string, data interface{}) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.connected || s.port == nil {
		return fmt.Errorf("serial port not connected")
	}

	// For a real system, you'd send data with a destination MCP ID.
	// This example assumes a single serial connection to one MCP or a hub.
	// We'll just marshal the data and send it.
	dataBytes, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal data for MCP %s: %w", mcpID, err)
	}
	
	// Prepend some identifier for the MCP or use a specific command protocol
	// For this example, simply write the JSON.
	fullData := append([]byte(fmt.Sprintf("CMD:%s:", mcpID)), dataBytes...)
	_, err = s.port.Write(fullData)
	if err != nil {
		return fmt.Errorf("failed to write to serial port for MCP %s: %w", mcpID, err)
	}
	// log.Printf("Wrote %d bytes to serial for MCP %s: %s", n, mcpID, string(fullData))
	return nil
}

// IsConnected implements MCPInterface.IsConnected
func (s *SerialMCPInterface) IsConnected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.connected
}

// RestartMCP simulates sending a restart command to an MCP
func (s *SerialMCPInterface) RestartMCP(mcpID string) error {
	log.Printf("Simulating restart command sent to MCP %s.", mcpID)
	// In reality: send a specific serial command sequence for restart
	s.SendGenericCommand(mcpID, "RESTART")
	return nil
}

// SendGenericCommand simulates sending a generic command to an MCP
func (s *SerialMCPInterface) SendGenericCommand(mcpID, command string) error {
	log.Printf("Simulating generic command '%s' sent to MCP %s.", command, mcpID)
	// In reality: Construct and send the appropriate serial message
	cmd := models.ActuatorCommand{
		MCPID: mcpID,
		Command: command,
		Value: nil, // Generic commands might not have a value
	}
	return s.Write(mcpID, cmd)
}

// ConnectedMCPs returns dummy connected MCPs for the serial example
func (s *SerialMCPInterface) ConnectedMCPs() []string {
	// In a real serial network (e.g., Modbus), this would involve polling or discovery.
	// For a direct connection, it might just be the one MCP.
	return []string{"MCP001", "MCP002", "MCP003"} // Example MCPs
}


// MockMCPInterface implements MCPInterface for testing without actual hardware
type MockMCPInterface struct {
	connected bool
	dataBuffer chan []byte
	knownMCPs map[string]bool
}

// NewMockMCPInterface creates a new MockMCPInterface
func NewMockMCPInterface() *MockMCPInterface {
	return &MockMCPInterface{
		connected: true,
		dataBuffer: make(chan []byte, 10), // Simulate a small buffer
		knownMCPs: map[string]bool{"MCP001": true, "MCP002": true, "MCP003": true},
	}
}

// Connect implements MCPInterface.Connect
func (m *MockMCPInterface) Connect() error {
	m.connected = true
	log.Println("Mock MCP connected.")
	return nil
}

// Close implements MCPInterface.Close
func (m *MockMCPInterface) Close() error {
	m.connected = false
	close(m.dataBuffer)
	log.Println("Mock MCP closed.")
	return nil
}

// Read implements MCPInterface.Read
func (m *MockMCPInterface) Read() ([]byte, error) {
	select {
	case data := <-m.dataBuffer:
		return data, nil
	case <-time.After(100 * time.Millisecond): // Simulate read timeout
		return nil, nil
	}
}

// Write implements MCPInterface.Write
func (m *MockMCPInterface) Write(mcpID string, data interface{}) error {
	if !m.connected {
		return fmt.Errorf("mock MCP not connected")
	}
	dataBytes, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal data for mock MCP %s: %w", mcpID, err)
	}
	log.Printf("MockMCP: Sent command to %s: %s", mcpID, string(dataBytes))
	return nil
}

// IsConnected implements MCPInterface.IsConnected
func (m *MockMCPInterface) IsConnected() bool {
	return m.connected
}

// RestartMCP simulates a restart command
func (m *MockMCPInterface) RestartMCP(mcpID string) error {
	log.Printf("MockMCP: Restarting MCP %s...", mcpID)
	// Simulate a brief disconnect/reconnect
	m.connected = false
	time.Sleep(500 * time.Millisecond)
	m.connected = true
	log.Printf("MockMCP: MCP %s restarted.", mcpID)
	return nil
}

// SendGenericCommand simulates sending a generic command
func (m *MockMCPInterface) SendGenericCommand(mcpID, command string) error {
	log.Printf("MockMCP: Sent generic command '%s' to MCP %s.", command, mcpID)
	return nil
}

// ConnectedMCPs returns a list of mock connected MCP IDs
func (m *MockMCPInterface) ConnectedMCPs() []string {
	keys := make([]string, 0, len(m.knownMCPs))
	for k := range m.knownMCPs {
		keys = append(keys, k)
	}
	return keys
}
```