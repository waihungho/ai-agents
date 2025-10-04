This Go-based AI Agent, named **RACE-AI (Resilient & Adaptive Environment Control AI Agent)**, is designed to autonomously monitor, analyze, and control dynamic environments. It features an **MCP (Microcontroller Peripheral) interface** which is abstracted via Go channels for robust and decoupled communication with sensors and actuators. RACE-AI integrates advanced concepts like predictive analytics, adaptive learning, contextual awareness, proactive intervention, and a basic digital twin for pre-computation, all while maintaining a Human-in-the-Loop for critical decisions. The design avoids direct duplication of large open-source libraries by outlining conceptual implementations for complex AI functionalities.

---

### AI Agent: RACE-AI (Resilient & Adaptive Environment Control AI Agent)
**Version: 1.0**
**Description:** RACE-AI is an intelligent agent designed to autonomously monitor, analyze, and control complex environments. It leverages an MCP (Microcontroller Peripheral) interface to interact with various sensors and actuators. The agent employs advanced concepts like predictive analytics, adaptive learning, contextual awareness, and proactive self-correction to maintain optimal environmental conditions, anticipate failures, and respond resiliently to dynamic challenges.

**MCP Interface:**
The MCP interface is implemented as a message-passing system via Go channels.
- `mcpCommandCh`: Agent sends control commands (e.g., "set_temp:22", "open_valve:3") to connected peripherals.
- `mcpSensorCh`: Agent receives telemetry/sensor data (e.g., "temp:25.1", "humidity:60") from connected peripherals.
In a real-world scenario, these channels would abstract communication over physical interfaces like MQTT, gRPC, Serial, I2C/SPI bridges, etc.

**Core Components:**
- `AgentState`: Holds the current operational state, configurations, and learned parameters.
- `DataStore`: Manages historical sensor data and operational logs (simplified as in-memory slice).
- `RuleEngine`: Processes contextual rules for decision making.
- `PredictiveModel`: Performs time-series forecasting and anomaly detection (simplified).
- `ActionQueue`: Manages pending and executing tasks.
- `DigitalTwinSimulator`: A lightweight simulation environment for pre-testing actions.

**Functions Summary (Total: 25 Functions):**

**I. Core Agent Lifecycle & MCP Interface**
1.  `NewRACEAI`: Initializes a new RACE-AI agent instance.
2.  `StartAgent`: Begins the agent's operational loop, listening for events.
3.  `ShutdownAgent`: Gracefully terminates the agent, saving state.
4.  `SendMCPCommand`: Dispatches a control command to a peripheral via MCP.
5.  `ListenForMCPTelemetry`: Processes incoming sensor data from peripherals.

**II. Data Ingestion, Processing & Storage**
6.  `IngestSensorData`: Stores and preprocesses raw telemetry, updating current readings.
7.  `UpdateContextualState`: Integrates new data to refine the agent's understanding of the environment.
8.  `StoreHistoricalData`: Archives sensor readings and operational metrics (conceptual, uses in-memory for this example).
9.  `RetrieveHistoricalData`: Fetches past data for analysis or learning within a specified duration.

**III. Environmental Analysis & Prediction**
10. `PerformSensorFusion`: Combines data from multiple sensors for a unified view or derived metrics.
11. `DetectEnvironmentalAnomaly`: Identifies unusual patterns or deviations from norms in sensor data.
12. `PredictFutureTrend`: Forecasts upcoming environmental changes (e.g., temperature, resource levels) based on historical data.
13. `AssessRiskFactor`: Evaluates potential threats or opportunities based on current context and future predictions.

**IV. Decision Making & Adaptive Control**
14. `EvaluateConditionRules`: Triggers actions based on predefined or dynamically learned rules.
15. `GenerateAdaptivePlan`: Formulates a series of actions to achieve a goal or resolve identified environmental issues.
16. `InitiateProactiveIntervention`: Takes action based on predictions, before an issue escalates to a critical state.
17. `AdjustBehaviorLearning`: Updates internal models and parameters based on the success or failure of past actions.
18. `RequestHumanConfirmation`: Prompts for human oversight on critical or high-impact decisions (Human-in-the-Loop).

**V. Action Execution & Resilience**
19. `EnqueueAction`: Adds a planned action to the execution queue for asynchronous processing.
20. `ExecuteQueuedActions`: Processes and dispatches actions from the queue to the MCP interface.
21. `MonitorActionOutcome`: Tracks the results of executed commands, determining success or failure.
22. `HandleExecutionFailure`: Implements fallback, retry, or recovery strategies for failed actions.
23. `RunSelfDiagnostic`: Checks the internal health, performance, and integrity of the AI agent itself.

**VI. Advanced & Creative Features**
24. `SimulateDigitalTwinAction`: Tests an action in a virtual environment to predict its outcome before real-world execution.
25. `ExplainDecisionLogic`: Provides a concise, human-readable rationale for a particular agent decision or action.

---

```go
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// --- datastructures.go ---

// SensorDataType enumerates types of sensor readings.
type SensorDataType string

const (
	Temperature SensorDataType = "temperature"
	Humidity    SensorDataType = "humidity"
	Pressure    SensorDataType = "pressure"
	Light       SensorDataType = "light"
	CO2         SensorDataType = "co2"
	Energy      SensorDataType = "energy"
	Occupancy   SensorDataType = "occupancy"
	// Add more as needed
)

// SensorData represents a single sensor reading.
type SensorData struct {
	ID        string         `json:"id"`        // Unique identifier for the sensor
	Type      SensorDataType `json:"type"`      // Type of data (e.g., "temperature")
	Value     float64        `json:"value"`     // Numeric value of the reading
	Unit      string         `json:"unit"`      // Unit of measurement (e.g., "C", "percent")
	Timestamp time.Time      `json:"timestamp"` // Time of the reading
}

// MCPCommandType enumerates types of commands sent to peripherals.
type MCPCommandType string

const (
	SetActuator    MCPCommandType = "set_actuator"    // Set a specific value (e.g., temp, light level)
	ToggleActuator MCPCommandType = "toggle_actuator" // Turn on/off
	Calibrate      MCPCommandType = "calibrate"       // Trigger calibration routine
	QueryStatus    MCPCommandType = "query_status"    // Request status update
	// Add more as needed
)

// MCPCommand represents a command to be sent to a peripheral.
type MCPCommand struct {
	TargetID string         `json:"target_id"` // ID of the peripheral/actuator
	Type     MCPCommandType `json:"type"`      // Type of command
	Payload  string         `json:"payload"`   // Command-specific parameters (e.g., "22.5C", "on")
	IssuedAt time.Time      `json:"issued_at"` // When the command was issued
}

// Action represents a high-level task or decision made by the agent.
type Action struct {
	ID          string         `json:"id"`           // Unique ID for the action
	Description string         `json:"description"`  // Human-readable description
	Command     MCPCommand     `json:"command"`      // The underlying MCP command to execute
	Priority    int            `json:"priority"`     // Higher values indicate higher priority
	Status      string         `json:"status"`       // "pending", "executing", "completed", "failed"
	ScheduledAt time.Time      `json:"scheduled_at"` // When the action is planned to run
	ExecutedAt  time.Time      `json:"executed_at"`  // When the action was actually executed
	Outcome     string         `json:"outcome"`      // Result of the action (e.g., "success", "temp_too_high")
	Rationale   string         `json:"rationale"`    // Why this action was chosen
}

// AgentConfiguration holds tunable parameters for the agent.
type AgentConfiguration struct {
	PollingInterval       time.Duration // How often to check for new sensor data
	DecisionInterval      time.Duration // How often to run decision logic
	AnomalyThreshold      float64       // Sensitivity for anomaly detection (e.g., number of std dev)
	HumanConfirmationFreq int           // Every N critical decisions requires human confirmation (0 for never)
	// ... other configurations
}

// EnvironmentContext represents the agent's current understanding of its environment.
type EnvironmentContext struct {
	CurrentSensorReadings map[string]SensorData // Latest readings from all sensors
	AggregatedMetrics     map[string]float64    // Derived metrics (e.g., average temp, light index)
	StatusIndicators      map[string]string     // System status (e.g., "HVAC:online", "Door:closed")
	KnownIssues           []string              // List of identified problems
	Predictions           map[string]float64    // Future state predictions
	LastUpdated           time.Time             // When the context was last refreshed
}

// PredictiveModel represents a simplified model for forecasting.
type PredictiveModel struct {
	// In a real scenario, this would hold actual model weights, algorithms, etc.
	// For this example, it's just a placeholder.
	ModelParameters map[string]float64
}

// Predict forecasts future values based on historical data.
func (pm *PredictiveModel) Predict(dataType SensorDataType, history []SensorData, steps int) (map[time.Time]float64, error) {
	// Placeholder for actual prediction logic.
	// In a real system, this would use time-series analysis (e.g., ARIMA, LSTM).
	if len(history) == 0 {
		return nil, fmt.Errorf("no historical data for prediction")
	}

	predictions := make(map[time.Time]float64)
	lastValue := history[len(history)-1].Value
	lastTime := history[len(history)-1].Timestamp

	// Simple linear extrapolation with some noise for demonstration
	for i := 1; i <= steps; i++ {
		predictedTime := lastTime.Add(time.Duration(i) * time.Minute * 10) // Predict every 10 minutes
		predictedValue := lastValue + float64(i)*0.05 + (rand.Float64()*0.2 - 0.1) // Simple trend + noise
		predictions[predictedTime] = predictedValue
		lastValue = predictedValue // Update for next step
	}
	return predictions, nil
}

// RuleEngine represents a simplified rule processing mechanism.
type RuleEngine struct {
	// In a real scenario, this would hold actual rules, perhaps a DSL parser.
	// For this example, it's just a placeholder.
	Rules map[string]Rule
}

// Rule defines a simple condition-action pair.
type Rule struct {
	Name      string
	Condition func(context *EnvironmentContext) bool
	ActionGen func(context *EnvironmentContext) Action // Function to generate action if condition is met
	Priority  int
}

// Evaluate applies all rules to the current environment context and returns potential actions.
func (re *RuleEngine) Evaluate(context *EnvironmentContext) []Action {
	var potentialActions []Action
	for _, rule := range re.Rules {
		if rule.Condition(context) {
			action := rule.ActionGen(context)
			action.Priority = rule.Priority // Set priority from rule
			potentialActions = append(potentialActions, action)
		}
	}
	return potentialActions
}

// DigitalTwinSimulator represents a simplified digital twin.
type DigitalTwinSimulator struct {
	// This would hold a computational model of the physical environment/system.
	// For this example, it's a simple placeholder.
}

// SimulateAction simulates the effect of an action on the environment.
func (dts *DigitalTwinSimulator) SimulateAction(action MCPCommand, currentContext *EnvironmentContext) (SimulatedOutcome, error) {
	// Placeholder for complex simulation logic.
	// This would predict changes in sensor readings, resource consumption, etc.
	fmt.Printf("Digital Twin: Simulating command '%s' with payload '%s'...\n", action.Type, action.Payload)

	// Simulate a delay for the simulation
	time.Sleep(50 * time.Millisecond)

	// A very basic simulation: if it's a "set_temp" command, assume success and predict new temp
	if action.Type == SetActuator && action.TargetID == "HVAC-01" {
		if strings.HasSuffix(action.Payload, "C") {
			targetTempStr := strings.TrimSuffix(action.Payload, "C")
			targetTemp, err := strconv.ParseFloat(targetTempStr, 64)
			if err == nil {
				return SimulatedOutcome{
					PredictedSensorChanges: map[SensorDataType]float64{
						Temperature: targetTemp + (rand.Float64()*0.5 - 0.25), // Predict new stable temp with some noise
					},
					PredictedResourceImpact: map[string]float64{
						"energy_consumption_kWh": 0.5 + rand.Float64()*0.2,
					},
					SuccessProbability: 0.95,
					Explanation:        fmt.Sprintf("HVAC system is expected to reach %.1fC within 15 minutes with minor energy cost.", targetTemp),
				}, nil
			}
		}
	}

	return SimulatedOutcome{
		PredictedSensorChanges:  make(map[SensorDataType]float64),
		PredictedResourceImpact: make(map[string]float64),
		SuccessProbability:      0.8, // Default success
		Explanation:             fmt.Sprintf("Action '%s' likely to succeed with no significant predicted changes.", action.Type),
	}, nil
}

// SimulatedOutcome represents the predicted results of a simulated action.
type SimulatedOutcome struct {
	PredictedSensorChanges  map[SensorDataType]float64
	PredictedResourceImpact map[string]float64 // e.g., energy, water
	SuccessProbability      float64
	Explanation             string
}

// AgentState holds the dynamic state of the RACE-AI agent.
type AgentState struct {
	IsRunning          bool
	CurrentConfig      AgentConfiguration
	EnvironmentContext EnvironmentContext
	ActionQueue        chan Action
	HistoricalData     []SensorData // Simplified; a real DB would be used for persistence and scale
	DecisionLog        []string     // Records of key decisions and their rationale
	CriticalEvents     []string     // List of critical events detected/handled
	KnownActuators     map[string]string // Map actuator ID to its function
	HumanConfirmationPending bool // Flag if HITL is waiting for approval
	// ... other state variables
}

// NewAgentState initializes a new AgentState.
func NewAgentState(config AgentConfiguration) *AgentState {
	return &AgentState{
		IsRunning:     false,
		CurrentConfig: config,
		EnvironmentContext: EnvironmentContext{
			CurrentSensorReadings: make(map[string]SensorData),
			AggregatedMetrics:     make(map[string]float64),
			StatusIndicators:      make(map[string]string),
			KnownIssues:           []string{},
			Predictions:           make(map[string]float64),
			LastUpdated:           time.Now(),
		},
		ActionQueue:    make(chan Action, 100), // Buffered channel for actions
		HistoricalData: []SensorData{},
		DecisionLog:    []string{},
		CriticalEvents: []string{},
		KnownActuators: map[string]string{
			"HVAC-01":   "temperature_control",
			"LIGHT-01":  "lighting_control",
			"VALVE-01":  "irrigation_control",
			"DOOR-01":   "access_control",
			"FAN-01":    "ventilation_control",
		},
	}
}

// --- agent.go ---

// RACEAI is the core structure for the AI agent.
type RACEAI struct {
	State              *AgentState
	mcpCommandCh       chan MCPCommand
	mcpSensorCh        chan SensorData
	stopCh             chan struct{}
	wg                 sync.WaitGroup // For graceful shutdown of goroutines
	predictiveModel    *PredictiveModel
	ruleEngine         *RuleEngine
	digitalTwin        *DigitalTwinSimulator
	humanConfirmation  chan bool // Channel for human feedback (true for approve, false for reject)
	lastDecisionTick   time.Time
	lastPollingTick    time.Time
}

// NewRACEAI initializes a new RACE-AI agent instance.
func NewRACEAI(config AgentConfiguration, cmdCh chan MCPCommand, sensorCh chan SensorData) *RACEAI {
	agent := &RACEAI{
		State:              NewAgentState(config),
		mcpCommandCh:       cmdCh,
		mcpSensorCh:        sensorCh,
		stopCh:             make(chan struct{}),
		predictiveModel:    &PredictiveModel{ModelParameters: map[string]float64{"temp_weight": 0.5, "humidity_weight": 0.3}}, // Example params
		ruleEngine:         &RuleEngine{},
		digitalTwin:        &DigitalTwinSimulator{},
		humanConfirmation:  make(chan bool, 1), // Buffered for quick response
		lastDecisionTick:   time.Now(),
		lastPollingTick:    time.Now(),
	}
	agent.InitRuleEngine() // Initialize rules
	return agent
}

// InitRuleEngine initializes the agent's rule engine with predefined rules.
func (r *RACEAI) InitRuleEngine() {
	r.ruleEngine.Rules = make(map[string]Rule)

	// Rule 1: Temperature too high, activate cooling
	r.ruleEngine.Rules["HighTempCooling"] = Rule{
		Name: "High Temperature Cooling",
		Condition: func(ctx *EnvironmentContext) bool {
			temp, ok := ctx.CurrentSensorReadings["temp-sensor-01"]
			return ok && temp.Value > 25.0 // Target: below 25C
		},
		ActionGen: func(ctx *EnvironmentContext) Action {
			return Action{
				Description: "Lowering temperature due to high readings.",
				Command: MCPCommand{
					TargetID: "HVAC-01",
					Type:     SetActuator,
					Payload:  "22.0C", // Set to target 22C
				},
				Priority:  80,
				Rationale: "Temperature exceeded 25.0C.",
			}
		},
		Priority: 80,
	}

	// Rule 2: Humidity too high, activate dehumidifier (or fan)
	r.ruleEngine.Rules["HighHumidityVentilation"] = Rule{
		Name: "High Humidity Ventilation",
		Condition: func(ctx *EnvironmentContext) bool {
			humidity, ok := ctx.CurrentSensorReadings["humid-sensor-01"]
			return ok && humidity.Value > 70.0 // Target: below 70%
		},
		ActionGen: func(ctx *EnvironmentContext) Action {
			return Action{
				Description: "Activating ventilation due to high humidity.",
				Command: MCPCommand{
					TargetID: "FAN-01",
					Type:     ToggleActuator,
					Payload:  "on",
				},
				Priority:  70,
				Rationale: "Humidity exceeded 70.0%.",
			}
		},
		Priority: 70,
	}

	// Rule 3: Low CO2/Occupancy, reduce lighting
	r.ruleEngine.Rules["LowOccupancyLightReduction"] = Rule{
		Name: "Low Occupancy Light Reduction",
		Condition: func(ctx *EnvironmentContext) bool {
			co2, co2Ok := ctx.CurrentSensorReadings["co2-sensor-01"]
			occupancy, occOk := ctx.CurrentSensorReadings["occupancy-sensor-01"]
			return (co2Ok && co2.Value < 500 && time.Now().Hour() > 18) || // After 6 PM, low CO2
				(occOk && occupancy.Value < 1 && time.Now().Hour() > 18) // Just occupancy
		},
		ActionGen: func(ctx *EnvironmentContext) Action {
			return Action{
				Description: "Reducing lighting due to low occupancy and time of day.",
				Command: MCPCommand{
					TargetID: "LIGHT-01",
					Type:     SetActuator,
					Payload:  "30%", // Dim to 30% (example for a light actuator)
				},
				Priority:  20,
				Rationale: "Low occupancy and evening hours detected.",
			}
		},
		Priority: 20,
	}

	log.Printf("RACE-AI: Rule engine initialized with %d rules.", len(r.ruleEngine.Rules))
}

// StartAgent begins the agent's operational loop, listening for events.
func (r *RACEAI) StartAgent() {
	if r.State.IsRunning {
		log.Println("RACE-AI: Agent already running.")
		return
	}

	r.State.IsRunning = true
	log.Println("RACE-AI: Starting agent operations...")

	// Goroutine for listening to MCP sensor data
	r.wg.Add(1)
	go func() {
		defer r.wg.Done()
		r.ListenForMCPTelemetry()
	}()

	// Goroutine for executing actions
	r.wg.Add(1)
	go func() {
		defer r.wg.Done()
		r.ExecuteQueuedActions()
	}()

	// Main decision and polling loop
	r.wg.Add(1)
	go func() {
		defer r.wg.Done()
		ticker := time.NewTicker(r.State.CurrentConfig.PollingInterval)
		decisionTicker := time.NewTicker(r.State.CurrentConfig.DecisionInterval)
		defer ticker.Stop()
		defer decisionTicker.Stop()

		for {
			select {
			case <-r.stopCh:
				log.Println("RACE-AI: Main agent loop stopping.")
				return
			case <-ticker.C:
				r.PollSensors() // Periodically request sensor data
				r.lastPollingTick = time.Now()
			case <-decisionTicker.C:
				if time.Since(r.lastDecisionTick) >= r.State.CurrentConfig.DecisionInterval {
					r.MakeDecisions() // Periodically make decisions
					r.lastDecisionTick = time.Now()
				}
			}
		}
	}()
	log.Println("RACE-AI: Agent started successfully.")
}

// ShutdownAgent gracefully terminates the agent, saving state.
func (r *RACEAI) ShutdownAgent() {
	if !r.State.IsRunning {
		log.Println("RACE-AI: Agent not running.")
		return
	}
	log.Println("RACE-AI: Initiating shutdown...")

	r.State.IsRunning = false
	close(r.stopCh) // Signal all goroutines to stop
	r.wg.Wait()     // Wait for all goroutines to finish

	// Perform any state saving or cleanup
	// Example: SaveHistoricalDataToFile("agent_history.log")
	log.Println("RACE-AI: Agent shut down gracefully.")
}

// SendMCPCommand dispatches a control command to a peripheral via MCP.
func (r *RACEAI) SendMCPCommand(cmd MCPCommand) error {
	select {
	case r.mcpCommandCh <- cmd:
		log.Printf("RACE-AI: Sent command to MCP: %s (Target: %s, Payload: %s)", cmd.Type, cmd.TargetID, cmd.Payload)
		return nil
	case <-time.After(1 * time.Second): // Timeout if channel is blocked
		return fmt.Errorf("timeout sending MCP command %s to %s", cmd.Type, cmd.TargetID)
	}
}

// ListenForMCPTelemetry processes incoming sensor data from peripherals.
func (r *RACEAI) ListenForMCPTelemetry() {
	log.Println("RACE-AI: Listening for MCP telemetry...")
	for {
		select {
		case <-r.stopCh:
			log.Println("RACE-AI: Stopping MCP telemetry listener.")
			return
		case data := <-r.mcpSensorCh:
			r.IngestSensorData(data)
			r.UpdateContextualState()
		}
	}
}

// IngestSensorData stores and preprocesses raw telemetry.
func (r *RACEAI) IngestSensorData(data SensorData) {
	r.State.HistoricalData = append(r.State.HistoricalData, data)
	// Keep historical data within a reasonable limit to prevent memory exhaustion
	if len(r.State.HistoricalData) > 1000 { // Max 1000 entries in history for this example
		r.State.HistoricalData = r.State.HistoricalData[len(r.State.HistoricalData)-1000:]
	}
	r.State.EnvironmentContext.CurrentSensorReadings[data.ID] = data
	// log.Printf("RACE-AI: Ingested sensor data - ID: %s, Type: %s, Value: %.2f%s",
	// 	data.ID, data.Type, data.Value, data.Unit) // Log less verbosely
}

// UpdateContextualState integrates new data into the agent's understanding of the environment.
func (r *RACEAI) UpdateContextualState() {
	// Aggregate metrics (e.g., calculate average temperature, check for critical thresholds)
	var totalTemp float64
	var tempCount int
	for _, sd := range r.State.EnvironmentContext.CurrentSensorReadings {
		if sd.Type == Temperature {
			totalTemp += sd.Value
			tempCount++
		}
	}
	if tempCount > 0 {
		r.State.EnvironmentContext.AggregatedMetrics["avg_temperature"] = totalTemp / float64(tempCount)
	}

	// Update last updated timestamp
	r.State.EnvironmentContext.LastUpdated = time.Now()

	// Detect issues based on current state (can be refined with more complex logic)
	r.State.EnvironmentContext.KnownIssues = []string{} // Clear and re-evaluate
	if r.State.EnvironmentContext.AggregatedMetrics["avg_temperature"] > 28.0 {
		r.State.EnvironmentContext.KnownIssues = append(r.State.EnvironmentContext.KnownIssues, "High ambient temperature detected.")
	}
	// ... more sophisticated logic to derive contextual understanding
	// log.Println("RACE-AI: Contextual state updated.") // Log less verbosely
}

// StoreHistoricalData archives sensor readings and operational metrics.
// (Simplified, as actual storage is just `HistoricalData` slice)
func (r *RACEAI) StoreHistoricalData(data SensorData) {
	// In a real system, this would write to a database, file, or cloud storage.
	// For this example, it appends to the in-memory slice which is handled by IngestSensorData.
	// This function serves as a conceptual placeholder.
}

// RetrieveHistoricalData fetches past data for analysis or learning.
func (r *RACEAI) RetrieveHistoricalData(dataType SensorDataType, duration time.Duration) []SensorData {
	var relevantData []SensorData
	cutoff := time.Now().Add(-duration)
	for _, data := range r.State.HistoricalData {
		if data.Type == dataType && data.Timestamp.After(cutoff) {
			relevantData = append(relevantData, data)
		}
	}
	// Sort by timestamp for consistency
	sort.Slice(relevantData, func(i, j int) bool {
		return relevantData[i].Timestamp.Before(relevantData[j].Timestamp)
	})
	// log.Printf("RACE-AI: Retrieved %d historical data points for %s over last %v.", len(relevantData), dataType, duration) // Log less verbosely
	return relevantData
}

// PerformSensorFusion combines data from multiple sensors for a unified view.
func (r *RACEAI) PerformSensorFusion() {
	// Example: Combining temp and humidity to get a "comfort index" or "heat index"
	temp, tempOk := r.State.EnvironmentContext.CurrentSensorReadings["temp-sensor-01"]
	humidity, humidOk := r.State.EnvironmentContext.CurrentSensorReadings["humid-sensor-01"]

	if tempOk && humidOk {
		// Simplified heat index calculation (for demonstration, not scientifically accurate)
		heatIndex := temp.Value + (humidity.Value/100)*temp.Value*0.1
		r.State.EnvironmentContext.AggregatedMetrics["heat_index"] = heatIndex
		log.Printf("RACE-AI: Sensor fusion: Heat Index calculated: %.2f", heatIndex)
	}

	// More complex fusion could involve Kalman filters, machine learning models, etc.
}

// DetectEnvironmentalAnomaly identifies unusual patterns or deviations from norms.
func (r *RACEAI) DetectEnvironmentalAnomaly() {
	// Simple anomaly detection: check if any sensor value is wildly outside historical bounds
	// In a real system, this would use statistical methods, ML models (e.g., Isolation Forest).
	for sensorID, data := range r.State.EnvironmentContext.CurrentSensorReadings {
		history := r.RetrieveHistoricalData(data.Type, 24*time.Hour)
		if len(history) < 10 { // Need enough data to compare
			continue
		}

		var sum, sumSq float64
		for _, h := range history {
			sum += h.Value
			sumSq += h.Value * h.Value
		}
		mean := sum / float64(len(history))
		variance := (sumSq / float64(len(history))) - (mean * mean)
		stdDev := 0.0
		if variance > 0 {
			stdDev = math.Sqrt(variance)
		}

		if stdDev > 0 && math.Abs(data.Value-mean) > r.State.CurrentConfig.AnomalyThreshold*stdDev {
			log.Printf("RACE-AI: ANOMALY DETECTED! Sensor %s (Type: %s) value %.2f is outside %.1f std dev (Mean: %.2f, StdDev: %.2f)",
				sensorID, data.Type, data.Value, r.State.CurrentConfig.AnomalyThreshold, mean, stdDev)
			r.State.CriticalEvents = append(r.State.CriticalEvents,
				fmt.Sprintf("Anomaly: %s value %.2f, significantly deviates from mean.", sensorID, data.Value))
		}
	}
}

// PredictFutureTrend forecasts upcoming environmental changes.
func (r *RACEAI) PredictFutureTrend(dataType SensorDataType, predictionHorizon time.Duration) {
	history := r.RetrieveHistoricalData(dataType, 24*time.Hour) // Use last 24 hours for prediction
	if len(history) == 0 {
		// log.Printf("RACE-AI: Cannot predict %s, no historical data.", dataType) // Log less verbosely
		return
	}

	steps := int(predictionHorizon / (10 * time.Minute)) // Predict every 10 minutes for the horizon
	predictions, err := r.predictiveModel.Predict(dataType, history, steps)
	if err != nil {
		log.Printf("RACE-AI: Error predicting %s: %v", dataType, err)
		return
	}

	for ts, val := range predictions {
		r.State.EnvironmentContext.Predictions[fmt.Sprintf("%s_at_%s", dataType, ts.Format(time.RFC3339))] = val
	}
	log.Printf("RACE-AI: Predicted %d future points for %s.", len(predictions), dataType)
}

// AssessRiskFactor evaluates potential threats or opportunities based on predictions.
func (r *RACEAI) AssessRiskFactor() {
	// Example: If predicted temperature exceeds critical threshold, raise a high risk.
	// If predicted energy consumption is low, it might be an opportunity for non-critical tasks.
	for predKey, predictedValue := range r.State.EnvironmentContext.Predictions {
		if strings.Contains(predKey, string(Temperature)) {
			if predictedValue > 27.0 { // High temperature risk threshold
				r.State.EnvironmentContext.KnownIssues = append(r.State.EnvironmentContext.KnownIssues,
					fmt.Sprintf("Predicted high temperature (%2.fC) in near future.", predictedValue))
				log.Printf("RACE-AI: High risk: %s will be %.2fC. Proactive action may be needed.", predKey, predictedValue)
			}
		}
		// ... more risk assessment logic based on other predictions
	}
}

// EvaluateConditionRules triggers actions based on predefined or learned rules.
func (r *RACEAI) EvaluateConditionRules() []Action {
	potentialActions := r.ruleEngine.Evaluate(&r.State.EnvironmentContext)
	if len(potentialActions) > 0 {
		log.Printf("RACE-AI: Rule engine evaluated, found %d potential actions.", len(potentialActions))
	}
	return potentialActions
}

// GenerateAdaptivePlan formulates a series of actions to achieve a goal or resolve an issue.
func (r *RACEAI) GenerateAdaptivePlan(goal string, currentIssues []string) []Action {
	// This is a placeholder for a complex planning algorithm (e.g., hierarchical task network, reinforcement learning).
	// For demonstration, it generates simple corrective actions based on issues.
	var plan []Action
	if len(currentIssues) == 0 {
		// log.Println("RACE-AI: No immediate issues to plan for. Maintaining optimal state.") // Log less verbosely
		return plan
	}

	log.Printf("RACE-AI: Generating adaptive plan for goal '%s' with issues: %v", goal, currentIssues)

	for _, issue := range currentIssues {
		if strings.Contains(issue, "High ambient temperature") {
			plan = append(plan, Action{
				Description: "Adaptive plan: Increase cooling due to high temperature.",
				Command: MCPCommand{
					TargetID: "HVAC-01",
					Type:     SetActuator,
					Payload:  "20.0C", // Aggressive cooling
				},
				Priority:  90,
				Rationale: "High ambient temperature detected, requires adaptive response.",
			})
			r.State.DecisionLog = append(r.State.DecisionLog, "Generated adaptive plan to lower temp.")
		}
		// ... more sophisticated plan generation based on multiple issues and desired goals
	}
	return plan
}

// InitiateProactiveIntervention takes action based on predictions, before an issue escalates.
func (r *RACEAI) InitiateProactiveIntervention() {
	// Example: If temperature is *predicted* to go high, pre-cool.
	for predKey, predictedValue := range r.State.EnvironmentContext.Predictions {
		if strings.Contains(predKey, string(Temperature)) {
			if predictedValue > 26.0 && predictedValue < 27.0 { // Predicted to be high, but not critical yet
				log.Printf("RACE-AI: Proactive: Temperature predicted to rise to %.2fC. Initiating pre-emptive cooling.", predictedValue)
				action := Action{
					Description: "Proactive pre-cooling initiated.",
					Command: MCPCommand{
						TargetID: "HVAC-01",
						Type:     SetActuator,
						Payload:  "23.0C", // Gentle pre-cooling
					},
					Priority:  75,
					Rationale: fmt.Sprintf("Predicted temp rise to %.2fC.", predictedValue),
				}
				r.EnqueueAction(action)
				r.State.DecisionLog = append(r.State.DecisionLog, "Initiated proactive pre-cooling.")
				// Clear prediction to avoid re-triggering for same event immediately
				delete(r.State.EnvironmentContext.Predictions, predKey)
				return // Only one proactive intervention per cycle for simplicity
			}
		}
	}
}

// AdjustBehaviorLearning updates internal models based on success/failure of past actions.
func (r *RACEAI) AdjustBehaviorLearning(action Action, success bool) {
	log.Printf("RACE-AI: Adjusting behavior learning for action '%s' (Success: %t)", action.Description, success)

	// Placeholder for actual learning.
	// In a real system, this would involve:
	// 1. Updating weights in predictive models.
	// 2. Modifying rule parameters or adding new rules based on outcomes.
	// 3. Reinforcement learning: updating Q-tables or neural network policies.
	if action.Command.Type == SetActuator && action.Command.TargetID == "HVAC-01" {
		if success {
			r.predictiveModel.ModelParameters["temp_cooling_effectiveness"] =
				r.predictiveModel.ModelParameters["temp_cooling_effectiveness"]*0.9 + 0.1 // Reinforce
		} else {
			r.predictiveModel.ModelParameters["temp_cooling_effectiveness"] =
				r.predictiveModel.ModelParameters["temp_cooling_effectiveness"]*0.9 - 0.1 // Penalize
		}
		log.Printf("RACE-AI: Predictive model parameter 'temp_cooling_effectiveness' adjusted to %.2f", r.predictiveModel.ModelParameters["temp_cooling_effectiveness"])
	}

	// Record the adjustment
	r.State.DecisionLog = append(r.State.DecisionLog,
		fmt.Sprintf("Learning: Action '%s' was %t. Model parameters adjusted.", action.Description, success))
}

// RequestHumanConfirmation prompts for human oversight on critical decisions (Human-in-the-Loop).
func (r *RACEAI) RequestHumanConfirmation(action Action) bool {
	if r.State.HumanConfirmationPending {
		log.Println("RACE-AI: Human confirmation already pending for another action. Skipping new request.")
		return false
	}

	r.State.HumanConfirmationPending = true
	log.Printf("RACE-AI: CRITICAL ACTION REQUIRES HUMAN CONFIRMATION: '%s' (Rationale: %s). Waiting for input...",
		action.Description, action.Rationale)

	// Simulate human interaction - In a real system, this would send a notification to a UI/API.
	// For now, it's blocked until `SimulateHumanInput` is called.
	select {
	case approved := <-r.humanConfirmation:
		r.State.HumanConfirmationPending = false
		log.Printf("RACE-AI: Human decision received: %t", approved)
		r.State.DecisionLog = append(r.State.DecisionLog, fmt.Sprintf("Human confirmed action '%s': %t", action.Description, approved))
		return approved
	case <-time.After(5 * time.Minute): // Timeout for human response
		r.State.HumanConfirmationPending = false
		log.Println("RACE-AI: Human confirmation timed out. Action automatically rejected for safety.")
		r.State.DecisionLog = append(r.State.DecisionLog, fmt.Sprintf("Human confirmation for '%s' timed out. Rejected.", action.Description))
		return false
	case <-r.stopCh:
		r.State.HumanConfirmationPending = false
		log.Println("RACE-AI: Agent shutting down, cancelling human confirmation request.")
		return false // Cannot confirm if agent is shutting down
	}
}

// SimulateHumanInput provides a way for an external system/user to approve/reject an action.
func (r *RACEAI) SimulateHumanInput(approve bool) {
	if r.State.HumanConfirmationPending {
		select {
		case r.humanConfirmation <- approve:
			log.Printf("RACE-AI: External human input provided: %t", approve)
		default:
			log.Println("RACE-AI: No pending human confirmation, or channel blocked. Input ignored.")
		}
	} else {
		log.Println("RACE-AI: No human confirmation pending. Input ignored.")
	}
}

// EnqueueAction adds a planned action to the execution queue.
func (r *RACEAI) EnqueueAction(action Action) {
	action.ID = fmt.Sprintf("action-%d-%d", time.Now().UnixNano(), rand.Intn(1000)) // Simple unique ID
	action.Status = "pending"
	action.ScheduledAt = time.Now()
	select {
	case r.State.ActionQueue <- action:
		log.Printf("RACE-AI: Action enqueued: '%s' (Priority: %d)", action.Description, action.Priority)
	default:
		log.Println("RACE-AI: Action queue is full, dropping action:", action.Description)
	}
}

// ExecuteQueuedActions processes and dispatches actions to MCP.
func (r *RACEAI) ExecuteQueuedActions() {
	log.Println("RACE-AI: Action executor started.")
	for {
		select {
		case <-r.stopCh:
			log.Println("RACE-AI: Stopping action executor.")
			return
		case action := <-r.State.ActionQueue:
			action.Status = "executing"
			action.ExecutedAt = time.Now()
			log.Printf("RACE-AI: Executing action: '%s' (Command: %s %s)", action.Description, action.Command.Type, action.Command.Payload)

			// Check for Human-in-the-Loop before critical actions
			isCritical := action.Priority >= 85 // Define "critical" by priority
			if isCritical && r.State.CurrentConfig.HumanConfirmationFreq > 0 {
				if rand.Intn(r.State.CurrentConfig.HumanConfirmationFreq) == 0 { // Randomly request confirmation based on frequency
					if !r.RequestHumanConfirmation(action) {
						action.Outcome = "rejected_by_human"
						r.MonitorActionOutcome(action) // Log rejection
						continue                      // Skip execution
					}
				}
			}

			// Simulate digital twin first for high-priority actions
			if action.Priority >= 70 {
				outcome, err := r.SimulateDigitalTwinAction(action.Command, &r.State.EnvironmentContext)
				if err != nil {
					log.Printf("RACE-AI: Digital twin simulation failed for action '%s': %v", action.Description, err)
					// Decide whether to proceed or abort
				} else {
					log.Printf("RACE-AI: Digital Twin predicted success probability: %.2f, changes: %v", outcome.SuccessProbability, outcome.PredictedSensorChanges)
					if outcome.SuccessProbability < 0.5 {
						log.Printf("RACE-AI: Digital Twin simulation suggests low success for action '%s'. Aborting actual execution.", action.Description)
						action.Outcome = "aborted_low_sim_success"
						r.MonitorActionOutcome(action)
						continue
					}
					// Update rationale with simulation insights
					action.Rationale = action.Rationale + " (Simulated: " + outcome.Explanation + ")"
				}
			}

			err := r.SendMCPCommand(action.Command)
			if err != nil {
				action.Outcome = "failed_send_command"
				r.HandleExecutionFailure(action, err)
			} else {
				// In a real system, we'd wait for feedback from the peripheral.
				// For simulation, assume immediate success or some delay for effect.
				time.AfterFunc(1*time.Second, func() { // Simulate a delay for command effect
					action.Outcome = "completed" // Assume success for now
					r.MonitorActionOutcome(action)
				})
			}
		}
	}
}

// MonitorActionOutcome tracks the results of executed commands.
func (r *RACEAI) MonitorActionOutcome(action Action) {
	log.Printf("RACE-AI: Action '%s' finished with outcome: %s", action.Description, action.Outcome)
	// Based on outcome, trigger learning adjustment
	r.AdjustBehaviorLearning(action, action.Outcome == "completed")
	r.State.DecisionLog = append(r.State.DecisionLog,
		fmt.Sprintf("Action ID %s ('%s') completed with outcome '%s'.", action.ID, action.Description, action.Outcome))
}

// HandleExecutionFailure implements fallback or recovery strategies for failed actions.
func (r *RACEAI) HandleExecutionFailure(action Action, err error) {
	log.Printf("RACE-AI: ERROR: Action '%s' failed: %v. Initiating recovery.", action.Description, err)
	r.State.CriticalEvents = append(r.State.CriticalEvents, fmt.Sprintf("Action failed: %s, Error: %v", action.Description, err))

	// Simple recovery: retry once or enqueue a diagnostic task
	if action.Command.Type == SetActuator || action.Command.Type == ToggleActuator {
		log.Printf("RACE-AI: Retrying command %s for %s...", action.Command.Type, action.Command.TargetID)
		r.EnqueueAction(Action{
			Description: "RETRY: " + action.Description,
			Command:     action.Command, // Retry the same command
			Priority:    action.Priority + 5,
			Rationale:   "Retrying due to previous failure.",
		})
	} else {
		// For other failures, perhaps enqueue a diagnostic task
		r.EnqueueAction(Action{
			Description: "DIAGNOSE: Check " + action.Command.TargetID + " for issues.",
			Command: MCPCommand{
				TargetID: action.Command.TargetID,
				Type:     QueryStatus,
				Payload:  "full_diagnostic",
			},
			Priority:  95, // High priority diagnostic
			Rationale: "Action failed, requires diagnostic.",
		})
	}

	// Update learning based on failure
	r.AdjustBehaviorLearning(action, false) // Mark as failure
}

// RunSelfDiagnostic checks the internal health and integrity of the agent.
func (r *RACEAI) RunSelfDiagnostic() {
	log.Println("RACE-AI: Running self-diagnostic...")
	healthy := true

	// Check if MCP channels are open/responsive (simplified for channels)
	select {
	case r.mcpCommandCh <- MCPCommand{TargetID: "self-test", Type: QueryStatus, Payload: "ping"}:
		// Successfully sent test command
	case <-time.After(100 * time.Millisecond):
		log.Println("RACE-AI: WARNING: MCP Command channel unresponsive or blocked.")
		healthy = false
	}

	// Check if action queue is backed up
	if len(r.State.ActionQueue) > cap(r.State.ActionQueue)/2 {
		log.Printf("RACE-AI: WARNING: Action queue is %d%% full. Potential bottleneck.", (len(r.State.ActionQueue)*100)/cap(r.State.ActionQueue))
		healthy = false
	}

	// Check if environment context is fresh
	if time.Since(r.State.EnvironmentContext.LastUpdated) > r.State.CurrentConfig.PollingInterval*2 {
		log.Println("RACE-AI: WARNING: Environment context is stale. Sensor data not updating or processing is slow.")
		healthy = false
	}

	if healthy {
		log.Println("RACE-AI: Self-diagnostic completed: Agent is healthy.")
	} else {
		log.Println("RACE-AI: Self-diagnostic completed: Agent has detected issues.")
	}
	r.State.DecisionLog = append(r.State.DecisionLog, fmt.Sprintf("Self-diagnostic run: Healthy=%t", healthy))
}

// SimulateDigitalTwinAction tests an action in a virtual environment before real-world execution.
func (r *RACEAI) SimulateDigitalTwinAction(command MCPCommand, currentContext *EnvironmentContext) (SimulatedOutcome, error) {
	log.Printf("RACE-AI: Preparing to simulate action: %s %s", command.Type, command.Payload)
	return r.digitalTwin.SimulateAction(command, currentContext)
}

// ExplainDecisionLogic provides a concise rationale for a particular agent decision or action.
func (r *RACEAI) ExplainDecisionLogic(action Action) string {
	explanation := fmt.Sprintf("Decision for Action '%s' (ID: %s):\n", action.Description, action.ID)
	explanation += fmt.Sprintf("  - Triggered by: %s\n", action.Rationale)

	// Elaborate based on state
	currentTemp, ok := r.State.EnvironmentContext.CurrentSensorReadings["temp-sensor-01"]
	if ok && action.Command.Type == SetActuator && action.Command.TargetID == "HVAC-01" {
		targetTemp := strings.TrimSuffix(action.Command.Payload, "C")
		targetTempVal, _ := strconv.ParseFloat(targetTemp, 64)
		if currentTemp.Value > targetTempVal {
			explanation += fmt.Sprintf("  - Current temperature (%.2f°C) was higher than target (%.2f°C).\n", currentTemp.Value, targetTempVal)
		} else {
			explanation += fmt.Sprintf("  - Current temperature (%.2f°C) was around or lower than target (%.2f°C). This might be proactive or maintenance.\n", currentTemp.Value, targetTempVal)
		}
	}

	// Add recent predictions if relevant
	if len(r.State.EnvironmentContext.Predictions) > 0 {
		explanation += "  - Relevant Predictions:\n"
		for key, val := range r.State.EnvironmentContext.Predictions {
			if strings.Contains(key, "temperature") {
				explanation += fmt.Sprintf("    - Predicted temperature: %.2f°C\n", val)
			}
		}
	}

	// Add recent critical events if relevant
	if len(r.State.CriticalEvents) > 0 {
		explanation += "  - Recent Critical Events:\n"
		for _, event := range r.State.CriticalEvents {
			explanation += fmt.Sprintf("    - %s\n", event)
		}
	}

	return explanation
}

// PollSensors requests updated sensor data from all known sensors.
func (r *RACEAI) PollSensors() {
	// In a real system, this might send a broadcast query or iterate through known sensor IDs.
	// For this simulation, we'll just log that polling occurred and rely on the peripheral simulator to push data.
	// log.Println("RACE-AI: Initiating sensor polling cycle.") // Log less verbosely
	// Example: send a query command to a specific sensor ID
	r.SendMCPCommand(MCPCommand{
		TargetID: "temp-sensor-01",
		Type:     QueryStatus,
		Payload:  "current_reading",
		IssuedAt: time.Now(),
	})
	r.SendMCPCommand(MCPCommand{
		TargetID: "humid-sensor-01",
		Type:     QueryStatus,
		Payload:  "current_reading",
		IssuedAt: time.Now(),
	})
	// Add more polling for other sensors if desired, or let them push proactively.
}

// MakeDecisions orchestrates the agent's decision-making process.
func (r *RACEAI) MakeDecisions() {
	log.Println("RACE-AI: Initiating decision-making cycle...")

	r.PerformSensorFusion()
	r.DetectEnvironmentalAnomaly()
	r.PredictFutureTrend(Temperature, 30*time.Minute)
	r.PredictFutureTrend(Humidity, 30*time.Minute)
	r.AssessRiskFactor()
	r.InitiateProactiveIntervention() // Proactive first

	// Evaluate rules for reactive actions
	ruleBasedActions := r.EvaluateConditionRules()
	for _, action := range ruleBasedActions {
		r.EnqueueAction(action)
	}

	// Generate adaptive plans for any major issues
	if len(r.State.EnvironmentContext.KnownIssues) > 0 {
		adaptivePlan := r.GenerateAdaptivePlan("Optimal Environment", r.State.EnvironmentContext.KnownIssues)
		for _, action := range adaptivePlan {
			r.EnqueueAction(action)
		}
	}

	r.RunSelfDiagnostic() // Check agent's health after decision cycle

	log.Println("RACE-AI: Decision-making cycle completed.")
}

// --- main.go ---

// PeripheralSimulator mimics a physical MCP device.
type PeripheralSimulator struct {
	ID            string
	Type          SensorDataType
	CurrentValue  float64
	Unit          string
	CommandCh     chan MCPCommand
	SensorDataCh  chan SensorData
	StopCh        chan struct{}
	Wg            sync.WaitGroup
	ValueMutex    sync.Mutex
	BehaviorFunc  func(currentVal float64) float64 // Function to simulate natural environmental changes
	RespondToPoll bool                           // Whether this sensor responds to QueryStatus
}

// NewPeripheralSimulator creates a new simulated peripheral.
func NewPeripheralSimulator(id string, sType SensorDataType, initialVal float64, unit string, cmdCh chan MCPCommand, dataCh chan SensorData, behavior func(currentVal float64) float64) *PeripheralSimulator {
	return &PeripheralSimulator{
		ID:            id,
		Type:          sType,
		CurrentValue:  initialVal,
		Unit:          unit,
		CommandCh:     cmdCh,
		SensorDataCh:  dataCh,
		StopCh:        make(chan struct{}),
		BehaviorFunc:  behavior,
		RespondToPoll: true, // Default to respond to polls
	}
}

// StartSimulating begins the peripheral's simulation loop.
func (ps *PeripheralSimulator) StartSimulating() {
	log.Printf("PeripheralSimulator [%s]: Starting simulation...", ps.ID)
	ps.Wg.Add(1)
	go ps.simulateSensorReadings()
	ps.Wg.Add(1)
	go ps.listenForCommands()
}

// StopSimulating gracefully stops the peripheral.
func (ps *PeripheralSimulator) StopSimulating() {
	log.Printf("PeripheralSimulator [%s]: Stopping simulation...", ps.ID)
	close(ps.StopCh)
	ps.Wg.Wait()
	log.Printf("PeripheralSimulator [%s]: Simulation stopped.", ps.ID)
}

// simulateSensorReadings simulates environmental changes and pushes data.
func (ps *PeripheralSimulator) simulateSensorReadings() {
	defer ps.Wg.Done()
	ticker := time.NewTicker(2 * time.Second) // Send data every 2 seconds
	defer ticker.Stop()

	for {
		select {
		case <-ps.StopCh:
			return
		case <-ticker.C:
			ps.ValueMutex.Lock()
			ps.CurrentValue = ps.BehaviorFunc(ps.CurrentValue) // Update based on behavior
			data := SensorData{
				ID:        ps.ID,
				Type:      ps.Type,
				Value:     ps.CurrentValue,
				Unit:      ps.Unit,
				Timestamp: time.Now(),
			}
			ps.ValueMutex.Unlock()
			select {
			case ps.SensorDataCh <- data:
				// log.Printf("PeripheralSimulator [%s]: Pushed data: %.2f%s", ps.ID, data.Value, data.Unit)
			case <-time.After(100 * time.Millisecond):
				log.Printf("PeripheralSimulator [%s]: SensorDataCh is blocked, dropping data.", ps.ID)
			}
		}
	}
}

// listenForCommands processes commands from the AI agent.
func (ps *PeripheralSimulator) listenForCommands() {
	defer ps.Wg.Done()
	for {
		select {
		case <-ps.StopCh:
			return
		case cmd := <-ps.CommandCh:
			if cmd.TargetID != ps.ID {
				continue // Not for this peripheral
			}
			log.Printf("PeripheralSimulator [%s]: Received command: %s (Payload: %s)", ps.ID, cmd.Type, cmd.Payload)
			ps.ValueMutex.Lock()
			switch cmd.Type {
			case SetActuator:
				// Example: Set temperature for a thermostat acting as a sensor
				if ps.Type == Temperature {
					valStr := strings.TrimSuffix(cmd.Payload, "C")
					targetVal, err := strconv.ParseFloat(valStr, 64)
					if err == nil {
						// Adjust behavior function to move towards target
						originalBehavior := ps.BehaviorFunc // Store current natural behavior
						ps.BehaviorFunc = func(currentVal float64) float64 {
							diff := targetVal - currentVal
							changeRate := 0.3 // How fast it responds to commands
							if math.Abs(diff) < 0.1 {
								// Near target, oscillate slightly
								return currentVal + (rand.Float64() - 0.5) * 0.1
							}
							adjustedChange := math.Copysign(changeRate, diff) // Move towards target
							return currentVal + adjustedChange + (originalBehavior(currentVal)-currentVal)*0.1 // Mix with natural flow
						}
						log.Printf("PeripheralSimulator [%s]: Setting %s to target %.2fC.", ps.ID, ps.Type, targetVal)
					}
				}
				// Example: Set light level
				if ps.Type == Light {
					percentStr := strings.TrimSuffix(cmd.Payload, "%")
					targetPercent, err := strconv.ParseFloat(percentStr, 64)
					if err == nil {
						ps.CurrentValue = targetPercent // Instantaneous for light
						log.Printf("PeripheralSimulator [%s]: Setting %s to %.0f%%.", ps.ID, ps.Type, targetPercent)
					}
				}
			case ToggleActuator:
				// Example: Fan (acts on humidity sensor)
				if ps.Type == Humidity { // Assuming this sensor can also be controlled (e.g., integrated dehumidifier)
					originalBehavior := ps.BehaviorFunc
					if cmd.Payload == "on" {
						// Actively reduce humidity
						ps.BehaviorFunc = func(currentVal float64) float64 { return currentVal - rand.Float64()*0.4 - 0.2 }
						log.Printf("PeripheralSimulator [%s]: Fan turned ON, actively reducing humidity.", ps.ID)
					} else if cmd.Payload == "off" {
						// Revert to natural humidity behavior
						ps.BehaviorFunc = originalBehavior
						log.Printf("PeripheralSimulator [%s]: Fan turned OFF, humidity behavior natural.", ps.ID)
					}
				}
			case QueryStatus:
				if ps.RespondToPoll {
					data := SensorData{
						ID:        ps.ID,
						Type:      ps.Type,
						Value:     ps.CurrentValue,
						Unit:      ps.Unit,
						Timestamp: time.Now(),
					}
					select {
					case ps.SensorDataCh <- data:
						log.Printf("PeripheralSimulator [%s]: Responded to QueryStatus with data.", ps.ID)
					case <-time.After(100 * time.Millisecond):
						log.Printf("PeripheralSimulator [%s]: SensorDataCh is blocked on QueryStatus, dropping data.", ps.ID)
					}
				}
			}
			ps.ValueMutex.Unlock()
		}
	}
}

func main() {
	log.SetOutput(os.Stdout)
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	log.Println("RACE-AI System: Starting up...")

	// MCP communication channels
	mcpCommandCh := make(chan MCPCommand, 10)
	mcpSensorCh := make(chan SensorData, 10)

	// Agent Configuration
	agentConfig := AgentConfiguration{
		PollingInterval:       5 * time.Second,
		DecisionInterval:      10 * time.Second,
		AnomalyThreshold:      2.5, // 2.5 standard deviations for anomaly detection
		HumanConfirmationFreq: 3,   // Request human confirmation for 1/3 of critical actions (0 for never)
	}

	// Initialize the AI Agent
	aiAgent := NewRACEAI(agentConfig, mcpCommandCh, mcpSensorCh)

	// Initialize Peripheral Simulators
	// Simulate natural environmental fluctuations for sensors
	tempBehavior := func(currentVal float64) float64 {
		change := (rand.Float64() - 0.5) * 0.5 // +- 0.25C
		currentVal += change
		if currentVal < 18.0 { currentVal = 18.0 } // Min temp
		if currentVal > 30.0 { currentVal = 30.0 } // Max temp
		return currentVal
	}
	humidBehavior := func(currentVal float64) float64 {
		change := (rand.Float64() - 0.5) * 1.5 // +- 0.75%
		currentVal += change
		if currentVal < 40.0 { currentVal = 40.0 }
		if currentVal > 90.0 { currentVal = 90.0 }
		return currentVal
	}
	co2Behavior := func(currentVal float64) float64 {
		change := (rand.Float64() - 0.5) * 50 // +- 25 ppm
		currentVal += change
		if currentVal < 400 { currentVal = 400 }
		if currentVal > 2000 { currentVal = 2000 }
		return currentVal
	}
	occupancyBehavior := func(currentVal float64) float64 {
		if rand.Float64() < 0.05 { // 5% chance to change occupancy
			return float64(int(rand.Float64() * 5)) // 0 to 4 people
		}
		return currentVal
	}

	// Create sensor simulators. For simplicity, temperature sensor will also respond to HVAC commands.
	tempSensor := NewPeripheralSimulator("temp-sensor-01", Temperature, 24.5, "C", mcpCommandCh, mcpSensorCh, tempBehavior)
	humidSensor := NewPeripheralSimulator("humid-sensor-01", Humidity, 65.0, "%", mcpCommandCh, mcpSensorCh, humidBehavior)
	co2Sensor := NewPeripheralSimulator("co2-sensor-01", CO2, 600.0, "ppm", mcpCommandCh, mcpSensorCh, co2Behavior)
	occupancySensor := NewPeripheralSimulator("occupancy-sensor-01", Occupancy, 0.0, "count", mcpCommandCh, mcpSensorCh, occupancyBehavior)
	lightActuator := NewPeripheralSimulator("LIGHT-01", Light, 100.0, "%", mcpCommandCh, mcpSensorCh, func(currentVal float64) float64 { return currentVal }) // Light doesn't fluctuate naturally, holds its value

	// Start all peripheral simulators
	peripherals := []*PeripheralSimulator{
		tempSensor, humidSensor, co2Sensor, occupancySensor, lightActuator,
	}

	for _, p := range peripherals {
		p.StartSimulating()
	}

	// Start the AI agent
	aiAgent.StartAgent()

	// Example of external interaction: simulate human confirmation
	// In a real system, this would come from a UI or external API call.
	go func() {
		time.Sleep(20 * time.Second)
		log.Println("Main: Simulating a human approving a critical action.")
		aiAgent.SimulateHumanInput(true) // Approve the first critical action
		time.Sleep(30 * time.Second)
		log.Println("Main: Simulating a human rejecting a critical action.")
		aiAgent.SimulateHumanInput(false) // Reject a subsequent critical action
	}()

	// Graceful shutdown on OS signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Println("RACE-AI System: Shutting down...")
	aiAgent.ShutdownAgent()
	for _, p := range peripherals {
		p.StopSimulating()
	}
	close(mcpCommandCh)
	close(mcpSensorCh)
	log.Println("RACE-AI System: All components stopped.")
}
```