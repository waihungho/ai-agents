This AI Agent, named **"ChronoMind"**, is designed with a **Multi-Channel Protocol (MCP) Interface** for highly adaptive and distributed interactions. ChronoMind embodies advanced concepts such as self-improving meta-learning, probabilistic world modeling, ephemeral sub-agent spawning, and ethical reasoning, all while ensuring explainability and adaptive persona emulation. It avoids duplicating existing open-source agent frameworks by focusing on a unique architecture and a custom, extensible MCP for communication.

## ChronoMind AI Agent: Outline and Function Summary

**Core Vision:** ChronoMind is a proactive, self-improving, and ethically-aware AI agent capable of operating across diverse digital environments through a modular, multi-channel communication protocol. It continuously refines its understanding of the world, learns from experience, and spawns specialized sub-agents to tackle complex, time-sensitive tasks.

---

### **I. Agent Core & Lifecycle Management**

The foundational components for starting, stopping, and managing the agent's fundamental operations.

1.  **`InitializeCoreComponents()`**:
    *   **Summary:** Sets up ChronoMind's essential modules including its internal state, memory, cognition engine, and communication dispatcher. This function ensures all necessary services are prepared before operations begin.
2.  **`StartAgentOperations()`**:
    *   **Summary:** Initiates the agent's primary operational loops, launching concurrent goroutines for perception, cognition, action execution, MCP message processing, and continuous learning. This brings ChronoMind online.
3.  **`ShutdownAgentGracefully()`**:
    *   **Summary:** Manages the orderly cessation of all agent activities. It sends termination signals to active goroutines, flushes pending data, persists critical state, and cleanly closes all communication channels to prevent data loss or resource leaks.
4.  **`ExecuteDeterminedAction(action ActionPlan)`**:
    *   **Summary:** Translates a high-level cognitive decision (`ActionPlan`) into a concrete set of environmental interactions. This involves orchestrating sub-actions, managing resources, and monitoring the execution for success or failure.
5.  **`UpdateInternalWorldModel(observations []Observation)`**:
    *   **Summary:** Incorporates new perceived data (`Observation`s) into the agent's dynamic, probabilistic model of its environment. This function is crucial for refining its understanding of the world, predicting future states, and identifying anomalies.

---

### **II. Multi-Channel Protocol (MCP) Interface**

The robust, custom communication layer enabling ChronoMind to interact seamlessly across various transports (HTTP, gRPC, NATS, etc.).

6.  **`HandleIncomingMCPMessage(msg AgentMessage)`**:
    *   **Summary:** Processes any message received from an external entity via any registered communication channel. It deserializes the message, validates its structure, and dispatches it to the appropriate internal agent component (e.g., perception, cognition, memory).
7.  **`DispatchOutgoingMCPMessage(msg AgentMessage)`**:
    *   **Summary:** Routes an internally generated message to the correct external communication channel. It uses the `ChannelHint` within the `AgentMessage` to select the appropriate `ChannelAdapter` for sending the message.
8.  **`RegisterMCPChannel(channelID string, adapter mcp.ChannelAdapter)`**:
    *   **Summary:** Dynamically adds a new communication channel (e.g., an HTTP server, gRPC client, NATS subscription) to the agent's MCP dispatcher. This allows ChronoMind to expand its communication reach at runtime.
9.  **`DeregisterMCPChannel(channelID string)`**:
    *   **Summary:** Removes an active communication channel from the MCP dispatcher. This can be used to free up resources or adapt to changes in the agent's operational environment.

---

### **III. Perception & Cognition Engine**

ChronoMind's capabilities for sensing its environment, processing information, making complex decisions, and understanding context.

10. **`IngestSensorData(sensorID string, data RawSensorData)`**:
    *   **Summary:** Accepts raw data streams from various virtual or physical sensors. This initial stage involves basic data validation and timestamping before forwarding to the perception processing unit.
11. **`ContextualReasoning(event EventData, historicalContext []MemoryEvent)`**:
    *   **Summary:** Analyzes incoming `EventData` not in isolation, but within its broader historical and current situational context, including relevant memories and the current world model. This allows for nuanced interpretation and decision-making.
12. **`PredictConsequences(action ActionPlan, simulationDepth int)`**:
    *   **Summary:** Simulates the likely short-term and long-term outcomes of potential `ActionPlan`s within its internal probabilistic world model. This function helps the agent evaluate the impact of its choices before committing to an action.
13. **`GenerateDecisionRationale(decisionID string)`**:
    *   **Summary:** Provides a human-readable, step-by-step explanation for a specific decision made by the agent. This is a core XAI (Explainable AI) feature, leveraging its audit trail of inputs, reasoning steps, and cognitive processes.
14. **`InterpretAmbiguousQuery(query string, currentContext Context)`**:
    *   **Summary:** Attempts to resolve vague, incomplete, or ambiguous natural language queries or commands. It uses probabilistic inference, contextual information, and past interactions to infer the most probable intent, requesting clarification if necessary.

---

### **IV. Memory, Learning & Self-Improvement**

How ChronoMind stores experiences, learns from them, and continually improves its own operational parameters and learning strategies.

15. **`StoreExperientialMemory(experience Experience)`**:
    *   **Summary:** Archives significant events, observations, and their outcomes (successes, failures, unexpected results) into long-term memory. This forms the basis for future learning and behavior adaptation.
16. **`RetrieveRelevantMemories(query string, context map[string]string)`**:
    *   **Summary:** Accesses past experiences and factual knowledge from its memory stores that are most relevant to a current query or situation. It employs semantic search and contextual matching for efficient recall.
17. **`SelfImproveLearningAlgorithm(metrics map[string]float64)`**:
    *   **Summary:** A meta-learning function that analyzes the agent's performance metrics (e.g., decision accuracy, task completion rate, resource efficiency). Based on these, it dynamically adjusts its own learning parameters, model weights, or even the choice of learning algorithms to enhance future performance.
18. **`SynthesizeNewKnowledge(facts []Fact, inferenceRules []InferenceRule)`**:
    *   **Summary:** Derives new information, relationships, or insights by applying logical inference rules and patterns to existing facts within its knowledge base. This allows ChronoMind to grow its understanding beyond direct observation.

---

### **V. Advanced & Creative Functionality**

ChronoMind's unique capabilities for distributed intelligence, ethical guidance, resource management, and autonomous self-maintenance.

19. **`SpawnEphemeralSubAgent(task Request, lifespan time.Duration)`**:
    *   **Summary:** Creates and deploys a temporary, highly specialized sub-agent instance to perform a focused task. These sub-agents operate autonomously within defined parameters and report back to the main ChronoMind, dissolving upon task completion or lifespan expiration.
20. **`ProposeEthicalResolution(dilemma EthicalDilemma, options []ActionPlan)`**:
    *   **Summary:** Evaluates potential `ActionPlan`s against a set of predefined ethical principles, guidelines, and weighted values. It suggests the most ethically aligned path, possibly highlighting trade-offs or potential conflicts, and can trigger human intervention for complex dilemmas.
21. **`AdaptCommunicationPersona(recipientProfile AgentProfile, communicationGoal CommunicationGoal)`**:
    *   **Summary:** Dynamically modifies its communication style, tone, vocabulary, and level of detail based on the perceived profile of the recipient (e.g., human user, another AI agent, technical expert) and the specific goal of the communication (e.g., inform, persuade, query).
22. **`SelfDiagnoseAndRepair(componentID string, symptoms []ErrorData)`**:
    *   **Summary:** Identifies internal malfunctions or performance degradation within its own sub-systems. It then attempts autonomous recovery actions such as restarting components, reconfiguring parameters, or isolating faulty modules to maintain operational integrity.
23. **`DynamicResourceAllocation(task string, priority Priority, constraints ResourceConstraints)`**:
    *   **Summary:** Optimizes the assignment of internal computational resources (e.g., CPU, memory, concurrent goroutines) or external physical resources (e.g., compute clusters, IoT devices) to pending tasks based on their priority, constraints, and current availability.
24. **`CollaborateWithExternalAgent(agentID string, task SharedTask, protocol string)`**:
    *   **Summary:** Initiates and manages collaborative tasks with other AI agents or human operators. This involves negotiating task divisions, sharing context and data, and coordinating actions through an agreed-upon communication protocol via the MCP.
25. **`AuditDecisionTrace(decisionID string)`**:
    *   **Summary:** Provides a complete, immutable record of all inputs, internal processing steps, contextual data, and outputs that led to a specific decision. This function is vital for debugging, compliance, and fulfilling explainability requirements.

---

```go
// ChronoMind AI Agent with Multi-Channel Protocol (MCP) Interface
//
// Core Vision: ChronoMind is a proactive, self-improving, and ethically-aware AI agent capable of
// operating across diverse digital environments through a modular, multi-channel communication protocol.
// It continuously refines its understanding of the world, learns from experience, and spawns specialized
// sub-agents to tackle complex, time-sensitive tasks.
//
// This implementation avoids duplicating existing open-source agent frameworks by focusing on a
// unique architecture and a custom, extensible MCP for communication.
//
// ====================================================================================================
// OUTLINE AND FUNCTION SUMMARY
// ====================================================================================================
//
// I. Agent Core & Lifecycle Management
//    The foundational components for starting, stopping, and managing the agent's fundamental operations.
//
// 1. InitializeCoreComponents():
//    Summary: Sets up ChronoMind's essential modules including its internal state, memory, cognition engine,
//             and communication dispatcher. This function ensures all necessary services are prepared
//             before operations begin.
//
// 2. StartAgentOperations():
//    Summary: Initiates the agent's primary operational loops, launching concurrent goroutines for
//             perception, cognition, action execution, MCP message processing, and continuous learning.
//             This brings ChronoMind online.
//
// 3. ShutdownAgentGracefully():
//    Summary: Manages the orderly cessation of all agent activities. It sends termination signals to
//             active goroutines, flushes pending data, persists critical state, and cleanly closes
//             all communication channels to prevent data loss or resource leaks.
//
// 4. ExecuteDeterminedAction(action ActionPlan):
//    Summary: Translates a high-level cognitive decision (ActionPlan) into a concrete set of environmental
//             interactions. This involves orchestrating sub-actions, managing resources, and monitoring
//             the execution for success or failure.
//
// 5. UpdateInternalWorldModel(observations []Observation):
//    Summary: Incorporates new perceived data (Observations) into the agent's dynamic, probabilistic model
//             of its environment. This function is crucial for refining its understanding of the world,
//             predicting future states, and identifying anomalies.
//
// II. Multi-Channel Protocol (MCP) Interface
//     The robust, custom communication layer enabling ChronoMind to interact seamlessly across various
//     transports (HTTP, gRPC, NATS, etc.).
//
// 6. HandleIncomingMCPMessage(msg AgentMessage):
//    Summary: Processes any message received from an external entity via any registered communication
//             channel. It deserializes the message, validates its structure, and dispatches it to the
//             appropriate internal agent component (e.g., perception, cognition, memory).
//
// 7. DispatchOutgoingMCPMessage(msg AgentMessage):
//    Summary: Routes an internally generated message to the correct external communication channel.
//             It uses the ChannelHint within the AgentMessage to select the appropriate ChannelAdapter
//             for sending the message.
//
// 8. RegisterMCPChannel(channelID string, adapter mcp.ChannelAdapter):
//    Summary: Dynamically adds a new communication channel (e.g., an HTTP server, gRPC client, NATS
//             subscription) to the agent's MCP dispatcher. This allows ChronoMind to expand its
//             communication reach at runtime.
//
// 9. DeregisterMCPChannel(channelID string):
//    Summary: Removes an active communication channel from the MCP dispatcher. This can be used to free
//             up resources or adapt to changes in the agent's operational environment.
//
// III. Perception & Cognition Engine
//      ChronoMind's capabilities for sensing its environment, processing information, making complex
//      decisions, and understanding context.
//
// 10. IngestSensorData(sensorID string, data RawSensorData):
//     Summary: Accepts raw data streams from various virtual or physical sensors. This initial stage
//              involves basic data validation and timestamping before forwarding to the perception
//              processing unit.
//
// 11. ContextualReasoning(event EventData, historicalContext []MemoryEvent):
//     Summary: Analyzes incoming EventData not in isolation, but within its broader historical and
//              current situational context, including relevant memories and the current world model.
//              This allows for nuanced interpretation and decision-making.
//
// 12. PredictConsequences(action ActionPlan, simulationDepth int):
//     Summary: Simulates the likely short-term and long-term outcomes of potential ActionPlans within
//              its internal probabilistic world model. This function helps the agent evaluate the impact
//              of its choices before committing to an action.
//
// 13. GenerateDecisionRationale(decisionID string):
//     Summary: Provides a human-readable, step-by-step explanation for a specific decision made by the
//              agent. This is a core XAI (Explainable AI) feature, leveraging its audit trail of inputs,
//              reasoning steps, and cognitive processes.
//
// 14. InterpretAmbiguousQuery(query string, currentContext Context):
//     Summary: Attempts to resolve vague, incomplete, or ambiguous natural language queries or commands.
//              It uses probabilistic inference, contextual information, and past interactions to infer
//              the most probable intent, requesting clarification if necessary.
//
// IV. Memory, Learning & Self-Improvement
//     How ChronoMind stores experiences, learns from them, and continually improves its own operational
//     parameters and learning strategies.
//
// 15. StoreExperientialMemory(experience Experience):
//     Summary: Archives significant events, observations, and their outcomes (successes, failures,
//              unexpected results) into long-term memory. This forms the basis for future learning
//              and behavior adaptation.
//
// 16. RetrieveRelevantMemories(query string, context map[string]string):
//     Summary: Accesses past experiences and factual knowledge from its memory stores that are most
//              relevant to a current query or situation. It employs semantic search and contextual
//              matching for efficient recall.
//
// 17. SelfImproveLearningAlgorithm(metrics map[string]float64):
//     Summary: A meta-learning function that analyzes the agent's performance metrics (e.g., decision
//              accuracy, task completion rate, resource efficiency). Based on these, it dynamically
//              adjusts its own learning parameters, model weights, or even the choice of learning
//              algorithms to enhance future performance.
//
// 18. SynthesizeNewKnowledge(facts []Fact, inferenceRules []InferenceRule):
//     Summary: Derives new information, relationships, or insights by applying logical inference rules
//              and patterns to existing facts within its knowledge base. This allows ChronoMind to grow
//              its understanding beyond direct observation.
//
// V. Advanced & Creative Functionality
//    ChronoMind's unique capabilities for distributed intelligence, ethical guidance, resource management,
//    and autonomous self-maintenance.
//
// 19. SpawnEphemeralSubAgent(task Request, lifespan time.Duration):
//     Summary: Creates and deploys a temporary, highly specialized sub-agent instance to perform a
//              focused task. These sub-agents operate autonomously within defined parameters and report
//              back to the main ChronoMind, dissolving upon task completion or lifespan expiration.
//
// 20. ProposeEthicalResolution(dilemma EthicalDilemma, options []ActionPlan):
//     Summary: Evaluates potential ActionPlans against a set of predefined ethical principles, guidelines,
//              and weighted values. It suggests the most ethically aligned path, possibly highlighting
//              trade-offs or potential conflicts, and can trigger human intervention for complex dilemmas.
//
// 21. AdaptCommunicationPersona(recipientProfile AgentProfile, communicationGoal CommunicationGoal):
//     Summary: Dynamically modifies its communication style, tone, vocabulary, and level of detail based
//              on the perceived profile of the recipient (e.g., human user, another AI agent, technical
//              expert) and the specific goal of the communication (e.g., inform, persuade, query).
//
// 22. SelfDiagnoseAndRepair(componentID string, symptoms []ErrorData):
//     Summary: Identifies internal malfunctions or performance degradation within its own sub-systems. It
//              then attempts autonomous recovery actions such as restarting components, reconfiguring
//              parameters, or isolating faulty modules to maintain operational integrity.
//
// 23. DynamicResourceAllocation(task string, priority Priority, constraints ResourceConstraints):
//     Summary: Optimizes the assignment of internal computational resources (e.g., CPU, memory, concurrent
//              goroutines) or external physical resources (e.g., compute clusters, IoT devices) to pending
//              tasks based on their priority, constraints, and current availability.
//
// 24. CollaborateWithExternalAgent(agentID string, task SharedTask, protocol string):
//     Summary: Initiates and manages collaborative tasks with other AI agents or human operators. This
//              involves negotiating task divisions, sharing context and data, and coordinating actions
//              through an agreed-upon communication protocol via the MCP.
//
// 25. AuditDecisionTrace(decisionID string):
//     Summary: Provides a complete, immutable record of all inputs, internal processing steps, contextual
//              data, and outputs that led to a specific decision. This function is vital for debugging,
//              compliance, and fulfilling explainability requirements.
//
// ====================================================================================================

package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"time"

	"ai-agent/agent"
	"ai-agent/mcp"
	"ai-agent/mcp/channels" // Example channel adapters
	"ai-agent/utils"
)

// main function to initialize and run the ChronoMind agent
func main() {
	// Initialize logger
	utils.InitLogger()

	slog.Info("Starting ChronoMind AI Agent initialization...")

	// 1. Initialize Core Components
	chronoMind, err := agent.NewAgent("ChronoMind-001")
	if err != nil {
		slog.Error("Failed to initialize ChronoMind agent", "error", err)
		os.Exit(1)
	}

	// 2. Register MCP Channels
	slog.Info("Registering MCP communication channels...")
	httpAdapter := channels.NewHTTPAdapter(":8080") // Listen on HTTP port 8080
	grpcAdapter := channels.NewGRPCAdapter(":50051") // Listen on gRPC port 50051
	// Example: NATS Adapter (requires a running NATS server)
	// natsAdapter, err := channels.NewNATSAdapter("nats://localhost:4222", "chronomind.inbox")
	// if err != nil {
	// 	slog.Warn("Could not initialize NATS adapter, continuing without it", "error", err)
	// } else {
	// 	chronoMind.RegisterMCPChannel("nats", natsAdapter)
	// }

	chronoMind.RegisterMCPChannel("http", httpAdapter)
	chronoMind.RegisterMCPChannel("grpc", grpcAdapter)

	// 3. Start Agent Operations
	slog.Info("Starting ChronoMind agent operations...")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel() // Ensure cancel is called to release resources

	go chronoMind.StartAgentOperations(ctx)

	// Simulate some agent activity or external interaction
	// This part would typically be driven by incoming MCP messages or sensor data
	time.AfterFunc(5*time.Second, func() {
		slog.Info("Simulating an external request via HTTP...")
		// In a real scenario, an external client would send this via HTTP
		mockMsg := mcp.AgentMessage{
			ID:          utils.GenerateUUID(),
			Sender:      "ExternalService-X",
			Recipient:   chronoMind.ID,
			Type:        "TaskRequest",
			Payload:     map[string]interface{}{"task": "analyze_system_logs", "threshold": 0.8},
			Timestamp:   time.Now(),
			ChannelHint: "http", // Or whatever the sending channel was
		}
		// Directly call HandleIncomingMCPMessage for demonstration purposes
		// In reality, httpAdapter/grpcAdapter would call this upon receiving a message
		chronoMind.HandleIncomingMCPMessage(mockMsg)
	})

	time.AfterFunc(10*time.Second, func() {
		slog.Info("Simulating an internal decision to spawn a sub-agent...")
		chronoMind.SpawnEphemeralSubAgent(agent.Request{
			ID:   utils.GenerateUUID(),
			Task: "diagnose_network_latency",
			Args: map[string]string{"target": "api.example.com"},
		}, 30*time.Second)
	})

	time.AfterFunc(15*time.Second, func() {
		slog.Info("ChronoMind self-evaluating and improving its learning algorithm...")
		chronoMind.SelfImproveLearningAlgorithm(map[string]float64{
			"task_completion_rate": 0.95,
			"decision_accuracy":    0.92,
			"resource_efficiency":  0.88,
		})
	})

	time.AfterFunc(20*time.Second, func() {
		slog.Info("ChronoMind attempting to generate an ethical resolution for a simulated dilemma...")
		chronoMind.ProposeEthicalResolution(
			agent.EthicalDilemma{ID: "ED-001", Description: "Allocate critical resource A between tasks B and C, where B is high-priority but less critical for human safety."},
			[]agent.ActionPlan{
				{ID: "AP-001", Description: "Allocate A to B"},
				{ID: "AP-002", Description: "Allocate A to C"},
			})
	})

	// Keep the main goroutine alive until an interrupt signal is received
	slog.Info("ChronoMind is running. Press CTRL+C to stop.")
	<-utils.SetupGracefulShutdown()

	// 4. Shutdown Agent Gracefully
	slog.Info("Initiating graceful shutdown of ChronoMind agent...")
	chronoMind.ShutdownAgentGracefully(ctx)
	slog.Info("ChronoMind agent stopped.")
}

// ====================================================================================================
// agent/types.go
// ====================================================================================================

package agent

import (
	"time"

	"ai-agent/mcp"
)

// Agent configuration
type Config struct {
	ID                 string
	PerceptionInterval time.Duration
	CognitionInterval  time.Duration
	MemoryRetention    time.Duration
	EthicalPrinciples  []EthicalPrinciple
}

// Agent's internal state
type AgentState struct {
	WorldModel map[string]interface{} // Probabilistic representation of the environment
	Status     string                 // e.g., "idle", "processing", "learning"
	Health     map[string]string      // Health status of internal components
}

// Observation represents structured data derived from raw sensor input
type Observation struct {
	ID        string
	SensorID  string
	Timestamp time.Time
	DataType  string
	Content   map[string]interface{}
	Certainty float64 // Probabilistic certainty of the observation
}

// ActionPlan represents a high-level decision or task to be executed
type ActionPlan struct {
	ID          string
	Description string
	Target      string                 // e.g., "ExternalSystem-X", "InternalComponent-Y"
	Parameters  map[string]interface{} // Specific parameters for the action
	Priority    Priority
	Deadline    time.Time
}

// ExecutionResult captures the outcome of an executed action
type ExecutionResult struct {
	ActionID string
	Success  bool
	Message  string
	Data     map[string]interface{}
	Duration time.Duration
}

// MemoryEvent for storing experiences and knowledge
type MemoryEvent struct {
	ID        string
	Timestamp time.Time
	Type      string                 // e.g., "Observation", "Decision", "ActionOutcome"
	Content   map[string]interface{} // Raw or processed data of the event
	Context   map[string]string      // Relevant contextual tags or identifiers
	Source    string                 // Where the memory originated
	Relevance float64                // How important this memory is
}

// LearningExperience for self-improvement
type LearningExperience struct {
	ID          string
	Timestamp   time.Time
	Observation Observation
	Action      ActionPlan
	Outcome     ExecutionResult
	Feedback    map[string]interface{} // e.g., human feedback, environmental reward
}

// RawSensorData is just raw input bytes or string from a sensor
type RawSensorData []byte

// EventData represents a structured event for cognitive processing
type EventData struct {
	ID        string
	Timestamp time.Time
	Type      string
	Payload   map[string]interface{}
}

// Request for ephemeral sub-agents
type Request struct {
	ID   string
	Task string
	Args map[string]string
}

// EphemeralSubAgent represents a temporary, specialized agent instance
type EphemeralSubAgent struct {
	ID        string
	ParentID  string
	Task      Request
	Status    string // e.g., "running", "completed", "failed"
	Lifespan  time.Duration
	CreatedAt time.Time
	// Could have its own MCP for limited interaction, or report back via parent's MCP
	ResultChannel chan EphemeralSubAgentResult
}

// EphemeralSubAgentResult is the outcome from a sub-agent
type EphemeralSubAgentResult struct {
	SubAgentID string
	Success    bool
	Output     map[string]interface{}
	Error      string
}

// EthicalDilemma for ethical reasoning
type EthicalDilemma struct {
	ID          string
	Description string
	Affected    []string // Entities affected
	ValuesAtRisk []string
}

// EthicalPrinciple represents a guideline for ethical decision making
type EthicalPrinciple struct {
	Name        string
	Description string
	Weight      float64 // How important this principle is
}

// AgentProfile describes characteristics of an interacting entity
type AgentProfile struct {
	ID         string
	Type       string // e.g., "human_user", "technical_agent", "junior_analyst"
	TrustLevel float64
	Preferences map[string]string
}

// CommunicationGoal defines the intent behind a communication
type CommunicationGoal string

const (
	GoalInform     CommunicationGoal = "inform"
	GoalPersuade   CommunicationGoal = "persuade"
	GoalQuery      CommunicationGoal = "query"
	GoalCommand    CommunicationGoal = "command"
	GoalApologize  CommunicationGoal = "apologize"
	GoalNegotiate  CommunicationGoal = "negotiate"
)

// ErrorData for self-diagnosis
type ErrorData struct {
	Component string
	Level     string // e.g., "warning", "error", "critical"
	Message   string
	Timestamp time.Time
	Details   map[string]interface{}
}

// Priority of a task or action
type Priority string

const (
	PriorityLow    Priority = "low"
	PriorityMedium Priority = "medium"
	PriorityHigh   Priority = "high"
	PriorityUrgent Priority = "urgent"
	PriorityCritical Priority = "critical"
)

// ResourceConstraints for dynamic allocation
type ResourceConstraints struct {
	CPULimit   float64 // e.g., 0.5 for 50% CPU
	MemoryLimit int64   // in bytes
	NetworkBandwidth int64 // in Mbps
}

// Fact for knowledge synthesis
type Fact struct {
	ID      string
	Content string
	Source  string
	Certainty float64
}

// InferenceRule for knowledge synthesis
type InferenceRule struct {
	ID          string
	Description string
	Condition   string // e.g., "IF A AND B THEN C" (simplified for example)
	Action      string // e.g., "INFER D"
}

// Context represents the current operational context for ambiguity resolution
type Context map[string]interface{}

// SharedTask for collaboration
type SharedTask struct {
	ID          string
	Description string
	Participants []string
	Status      string // e.g., "pending", "in_progress", "completed"
	SharedData  map[string]interface{}
}

// ====================================================================================================
// agent/config.go
// ====================================================================================================

package agent

import "time"

// DefaultConfig provides a basic configuration for the agent
func DefaultConfig(agentID string) Config {
	return Config{
		ID:                 agentID,
		PerceptionInterval: 2 * time.Second,
		CognitionInterval:  1 * time.Second,
		MemoryRetention:    30 * 24 * time.Hour, // 30 days
		EthicalPrinciples: []EthicalPrinciple{
			{Name: "DoNoHarm", Description: "Prioritize safety and well-being of all entities.", Weight: 1.0},
			{Name: "Transparency", Description: "Decisions should be explainable and auditable.", Weight: 0.8},
			{Name: "Fairness", Description: "Avoid bias and treat entities equitably.", Weight: 0.7},
		},
	}
}

// ====================================================================================================
// agent/agent.go
// ====================================================================================================

package agent

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"ai-agent/mcp"
	"ai-agent/utils"
)

// Agent represents the ChronoMind AI agent's core structure.
type Agent struct {
	ID        string
	Config    Config
	State     AgentState
	stop      context.CancelFunc
	wg        sync.WaitGroup

	// Internal communication channels
	mcpToAgentChan      chan mcp.AgentMessage // From MCP Dispatcher to Agent
	agentToMcpChan      chan mcp.AgentMessage // From Agent to MCP Dispatcher
	sensorDataChan      chan RawSensorData
	perceivedDataChan   chan Observation
	decisionChan        chan ActionPlan
	actionResultChan    chan ExecutionResult
	memoryEventChan     chan MemoryEvent
	learningExperienceChan chan LearningExperience
	ethicalDilemmaChan  chan EthicalDilemma // For receiving ethical dilemmas
	subAgentRequestChan chan Request        // For spawning sub-agents
	subAgentResultChan  chan EphemeralSubAgentResult // From sub-agents

	// External modules/interfaces
	MCPDispatcher *mcp.MessageDispatcher
	Memory        *MemoryModule
	// ... other modules could be added here (e.g., PerceptionEngine, CognitionEngine)
}

// NewAgent creates and initializes a new ChronoMind agent instance.
// This implements `InitializeCoreComponents()`.
func NewAgent(id string) (*Agent, error) {
	cfg := DefaultConfig(id)

	// Initialize internal communication channels
	mcpToAgentChan := make(chan mcp.AgentMessage, 100)
	agentToMcpChan := make(chan mcp.AgentMessage, 100)
	sensorDataChan := make(chan RawSensorData, 50)
	perceivedDataChan := make(chan Observation, 50)
	decisionChan := make(chan ActionPlan, 20)
	actionResultChan := make(chan ExecutionResult, 20)
	memoryEventChan := make(chan MemoryEvent, 100)
	learningExperienceChan := make(chan LearningExperience, 50)
	ethicalDilemmaChan := make(chan EthicalDilemma, 5)
	subAgentRequestChan := make(chan Request, 10)
	subAgentResultChan := make(chan EphemeralSubAgentResult, 10)

	// Initialize MCP Dispatcher
	mcpDispatcher := mcp.NewMessageDispatcher(mcpToAgentChan, agentToMcpChan)

	// Initialize Memory Module
	memoryModule := NewMemoryModule()

	agent := &Agent{
		ID:        id,
		Config:    cfg,
		State:     AgentState{WorldModel: make(map[string]interface{}), Status: "initializing", Health: make(map[string]string)},
		mcpToAgentChan: mcpToAgentChan,
		agentToMcpChan: agentToMcpChan,
		sensorDataChan: sensorDataChan,
		perceivedDataChan: perceivedDataChan,
		decisionChan: decisionChan,
		actionResultChan: actionResultChan,
		memoryEventChan: memoryEventChan,
		learningExperienceChan: learningExperienceChan,
		ethicalDilemmaChan: ethicalDilemmaChan,
		subAgentRequestChan: subAgentRequestChan,
		subAgentResultChan: subAgentResultChan,
		MCPDispatcher: mcpDispatcher,
		Memory:        memoryModule,
	}

	slog.Info("ChronoMind agent core components initialized.", "agent_id", agent.ID)
	return agent, nil
}

// StartAgentOperations initiates the agent's primary operational loops.
// This implements `StartAgentOperations()`.
func (a *Agent) StartAgentOperations(ctx context.Context) {
	slog.Info("Starting ChronoMind agent operations.", "agent_id", a.ID)
	a.State.Status = "running"
	ctx, a.stop = context.WithCancel(ctx) // Create a child context for agent's operations

	// Start MCP Dispatcher (non-blocking)
	a.MCPDispatcher.Start(ctx)

	// Launch all main loops as goroutines
	a.wg.Add(8) // Increment for each goroutine below

	go a.perceptionLoop(ctx)
	go a.cognitionLoop(ctx)
	go a.actionExecutionLoop(ctx)
	go a.mcpHandlingLoop(ctx)
	go a.memoryManagementLoop(ctx)
	go a.learningLoop(ctx)
	go a.subAgentManagementLoop(ctx)
	go a.selfMonitoringLoop(ctx) // New loop for self-diagnosis/resource allocation

	slog.Info("All ChronoMind agent operational loops started.", "agent_id", a.ID)
}

// ShutdownAgentGracefully manages the orderly cessation of all agent activities.
// This implements `ShutdownAgentGracefully()`.
func (a *Agent) ShutdownAgentGracefully(ctx context.Context) {
	slog.Info("Initiating graceful shutdown...", "agent_id", a.ID)
	a.State.Status = "shutting_down"

	// Signal all goroutines to stop
	if a.stop != nil {
		a.stop()
	}

	// Wait for all goroutines to finish
	a.wg.Wait()

	// Perform final cleanup
	a.MCPDispatcher.Stop() // Stop MCP dispatcher
	a.Memory.PersistAll()  // Persist memory
	close(a.mcpToAgentChan)
	close(a.agentToMcpChan)
	close(a.sensorDataChan)
	close(a.perceivedDataChan)
	close(a.decisionChan)
	close(a.actionResultChan)
	close(a.memoryEventChan)
	close(a.learningExperienceChan)
	close(a.ethicalDilemmaChan)
	close(a.subAgentRequestChan)
	close(a.subAgentResultChan)


	slog.Info("ChronoMind agent has successfully shut down all operations.", "agent_id", a.ID)
}

// perceptionLoop continuously gathers and processes raw sensor data.
func (a *Agent) perceptionLoop(ctx context.Context) {
	defer a.wg.Done()
	slog.Info("Perception loop started.", "agent_id", a.ID)
	ticker := time.NewTicker(a.Config.PerceptionInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			slog.Info("Perception loop stopped.", "agent_id", a.ID)
			return
		case <-ticker.C:
			// Simulate sensing the environment
			raw := a.SenseEnvironment() // This would fetch data from registered sensors
			if len(raw) > 0 {
				a.IngestSensorData("simulated_sensor", raw)
			}
		case rawData := <-a.sensorDataChan:
			// Process incoming raw sensor data
			a.ProcessPerception(rawData)
		}
	}
}

// ProcessPerception interprets raw sensory data into structured information.
// This is an internal helper function for `IngestSensorData`.
func (a *Agent) ProcessPerception(raw RawSensorData) {
	// In a real scenario, this would involve complex ML models, NLP, computer vision etc.
	// For now, it's a simple placeholder.
	slog.Debug("Processing raw sensor data...", "agent_id", a.ID, "data_size", len(raw))
	observation := Observation{
		ID:        utils.GenerateUUID(),
		SensorID:  "simulated_sensor",
		Timestamp: time.Now(),
		DataType:  "generic_data",
		Content:   map[string]interface{}{"raw_data": string(raw)}, // Example content
		Certainty: 0.9, // Placeholder
	}
	a.perceivedDataChan <- observation // Send structured observation for cognition
	a.memoryEventChan <- MemoryEvent{
		ID:        utils.GenerateUUID(),
		Timestamp: observation.Timestamp,
		Type:      "Observation",
		Content:   observation.Content,
		Context:   map[string]string{"sensor_id": observation.SensorID},
		Source:    a.ID,
		Relevance: observation.Certainty,
	}
}

// SenseEnvironment simulates gathering raw data from registered sensors.
// This is part of the `perceptionLoop`.
func (a *Agent) SenseEnvironment() RawSensorData {
	// In a real system, this would interact with various sensor modules (e.g., HTTP clients, file watchers, message queue consumers)
	// For demonstration, just return a dummy byte slice.
	slog.Debug("Sensing environment (simulated)...", "agent_id", a.ID)
	return []byte(fmt.Sprintf("Environmental data at %s", time.Now().Format(time.RFC3339)))
}

// cognitionLoop processes perceived data, makes decisions, and plans actions.
func (a *Agent) cognitionLoop(ctx context.Context) {
	defer a.wg.Done()
	slog.Info("Cognition loop started.", "agent_id", a.ID)
	ticker := time.NewTicker(a.Config.CognitionInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			slog.Info("Cognition loop stopped.", "agent_id", a.ID)
			return
		case <-ticker.C:
			// Periodically re-evaluate world model or internal state
			// For now, just a placeholder. Real cognition would be more event-driven.
			a.RefineWorldModel()
		case obs := <-a.perceivedDataChan:
			slog.Debug("Cognition processing new observation", "agent_id", a.ID, "obs_id", obs.ID)
			// Trigger decision making based on observation
			a.DecideAction(EventData{ID: obs.ID, Timestamp: obs.Timestamp, Type: "Observation", Payload: obs.Content})
		case dilemma := <-a.ethicalDilemmaChan:
			slog.Info("Cognition received ethical dilemma, proposing resolution", "agent_id", a.ID, "dilemma_id", dilemma.ID)
			// Process ethical dilemma, then potentially decide an action or request human intervention
			a.ProposeEthicalResolution(dilemma, []ActionPlan{
				{ID: "hypo-action-1", Description: "Option A", Priority: PriorityMedium},
				{ID: "hypo-action-2", Description: "Option B", Priority: PriorityHigh},
			})
		}
	}
}

// RefineWorldModel updates the probabilistic world model based on new data and experiences.
// This implements part of `UpdateInternalWorldModel()`.
func (a *Agent) RefineWorldModel() {
	// In a real system, this would involve Bayesian inference, Kalman filters, etc.
	// For now, it's a simple placeholder that updates a dummy value.
	newCertainty := utils.GenerateRandomFloat(0.7, 0.99)
	a.State.WorldModel["overall_certainty"] = newCertainty
	slog.Debug("Refined internal world model.", "agent_id", a.ID, "new_certainty", newCertainty)
}

// DecideAction is the main decision-making loop based on perception and state.
// This implements part of `ExecuteDeterminedAction()`.
func (a *Agent) DecideAction(event EventData) {
	slog.Debug("Agent deciding action based on event", "agent_id", a.ID, "event_type", event.Type)

	// Example: If a "TaskRequest" message came in via MCP
	if event.Type == "TaskRequest" {
		taskDesc := event.Payload["task"].(string) // Assuming payload has "task" key
		threshold := event.Payload["threshold"].(float64) // Assuming payload has "threshold" key

		slog.Info("Received task request", "task", taskDesc, "threshold", threshold)

		// Simulate cognitive processing and decision making
		action := ActionPlan{
			ID:          utils.GenerateUUID(),
			Description: fmt.Sprintf("Perform %s with threshold %f", taskDesc, threshold),
			Target:      "ExternalSystem-Processor", // Example target
			Parameters:  map[string]interface{}{"task_type": taskDesc, "min_accuracy": threshold},
			Priority:    PriorityHigh,
			Deadline:    time.Now().Add(10 * time.Minute),
		}

		// Simulate predicting consequences (function 12)
		a.PredictConsequences(action, 3)

		a.decisionChan <- action // Send action to execution loop
		a.memoryEventChan <- MemoryEvent{
			ID:        utils.GenerateUUID(),
			Timestamp: time.Now(),
			Type:      "Decision",
			Content:   map[string]interface{}{"action_id": action.ID, "description": action.Description},
			Context:   map[string]string{"source_event_id": event.ID},
			Source:    a.ID,
			Relevance: 0.95,
		}
	}
	// ... more complex decision logic based on current state, memories, goals etc.
}


// actionExecutionLoop receives and executes determined actions.
func (a *Agent) actionExecutionLoop(ctx context.Context) {
	defer a.wg.Done()
	slog.Info("Action execution loop started.", "agent_id", a.ID)

	for {
		select {
		case <-ctx.Done():
			slog.Info("Action execution loop stopped.", "agent_id", a.ID)
			return
		case action := <-a.decisionChan:
			a.ExecuteDeterminedAction(action)
		}
	}
}

// ExecuteDeterminedAction translates a high-level cognitive decision into an environmental action.
// This implements `ExecuteDeterminedAction()`.
func (a *Agent) ExecuteDeterminedAction(action ActionPlan) {
	slog.Info("Executing action plan", "agent_id", a.ID, "action_id", action.ID, "description", action.Description, "target", action.Target)

	// In a real system, this would involve calling external APIs, controlling devices, etc.
	// For now, simulate execution with a delay.
	go func() {
		startTime := time.Now()
		// Simulate a varying execution time
		time.Sleep(time.Duration(utils.GenerateRandomInt(500, 2000)) * time.Millisecond)

		success := utils.GenerateRandomFloat(0, 1) > 0.1 // 90% success rate
		result := ExecutionResult{
			ActionID: action.ID,
			Success:  success,
			Message:  "Action simulated completion",
			Data:     map[string]interface{}{"output": "Simulated output data"},
			Duration: time.Since(startTime),
		}
		if !success {
			result.Message = "Action simulated failure"
			result.Data["error"] = "Simulated error condition"
			a.SelfDiagnoseAndRepair("action_module", []ErrorData{{Component: "action_execution", Message: "Simulated action failure"}})
		}
		a.actionResultChan <- result // Send result back for learning/feedback
		slog.Info("Action execution completed", "agent_id", a.ID, "action_id", action.ID, "success", success)
	}()
}

// mcpHandlingLoop listens for incoming MCP messages and dispatches internal messages.
func (a *Agent) mcpHandlingLoop(ctx context.Context) {
	defer a.wg.Done()
	slog.Info("MCP handling loop started.", "agent_id", a.ID)

	for {
		select {
		case <-ctx.Done():
			slog.Info("MCP handling loop stopped.", "agent_id", a.ID)
			return
		case incomingMsg := <-a.mcpToAgentChan:
			a.HandleIncomingMCPMessage(incomingMsg)
		case outgoingMsg := <-a.agentToMcpChan:
			a.DispatchOutgoingMCPMessage(outgoingMsg)
		}
	}
}

// HandleIncomingMCPMessage processes messages received from any registered channel.
// This implements `HandleIncomingMCPMessage()`.
func (a *Agent) HandleIncomingMCPMessage(msg mcp.AgentMessage) {
	slog.Info("Received MCP message", "agent_id", a.ID, "sender", msg.Sender, "type", msg.Type, "channel", msg.ChannelHint)

	// Based on message type, forward to appropriate internal channel/logic
	switch msg.Type {
	case "TaskRequest":
		// Directly create an EventData for cognition loop based on the message payload
		a.perceivedDataChan <- Observation{
			ID:        utils.GenerateUUID(),
			SensorID:  "mcp_channel:" + msg.ChannelHint,
			Timestamp: msg.Timestamp,
			DataType:  "TaskRequest",
			Content:   msg.Payload.(map[string]interface{}), // Assume payload is map[string]interface{}
			Certainty: 1.0,
		}
	case "QueryKnowledge":
		query := msg.Payload.(map[string]interface{})["query"].(string)
		slog.Debug("Processing knowledge query", "query", query)
		// Simulate query and send response back
		responsePayload := a.QueryKnowledgeGraph(query)
		responseMsg := mcp.AgentMessage{
			ID:          utils.GenerateUUID(),
			Sender:      a.ID,
			Recipient:   msg.Sender,
			Type:        "QueryResponse",
			Payload:     responsePayload,
			Timestamp:   time.Now(),
			ChannelHint: msg.ChannelHint, // Respond on the same channel
			CorrelationID: msg.ID,
		}
		a.agentToMcpChan <- responseMsg
	// ... handle other message types (e.g., "SensorUpdate", "Command")
	default:
		slog.Warn("Unhandled MCP message type", "agent_id", a.ID, "type", msg.Type)
	}
}

// DispatchOutgoingMCPMessage routes messages to appropriate external channels.
// This implements `DispatchOutgoingMCPMessage()`.
func (a *Agent) DispatchOutgoingMCPMessage(msg mcp.AgentMessage) {
	slog.Info("Dispatching outgoing MCP message", "agent_id", a.ID, "recipient", msg.Recipient, "type", msg.Type, "channel_hint", msg.ChannelHint)
	// The MCPDispatcher handles the actual sending to the ChannelAdapter
	err := a.MCPDispatcher.SendMessage(msg)
	if err != nil {
		slog.Error("Failed to dispatch outgoing MCP message", "agent_id", a.ID, "error", err, "message_id", msg.ID)
	}
}

// RegisterMCPChannel dynamically adds a new communication channel.
// This implements `RegisterMCPChannel()`.
func (a *Agent) RegisterMCPChannel(channelID string, adapter mcp.ChannelAdapter) {
	a.MCPDispatcher.RegisterChannel(channelID, adapter)
	slog.Info("MCP Channel registered", "agent_id", a.ID, "channel_id", channelID)
}

// DeregisterMCPChannel removes an active communication channel.
// This implements `DeregisterMCPChannel()`.
func (a *Agent) DeregisterMCPChannel(channelID string) {
	a.MCPDispatcher.DeregisterChannel(channelID)
	slog.Info("MCP Channel deregistered", "agent_id", a.ID, "channel_id", channelID)
}

// IngestSensorData accepts raw data streams from various sensors.
// This implements `IngestSensorData()`.
func (a *Agent) IngestSensorData(sensorID string, data RawSensorData) {
	slog.Debug("Ingesting raw sensor data", "agent_id", a.ID, "sensor_id", sensorID, "data_size", len(data))
	a.sensorDataChan <- data // Push raw data to the perception loop
}

// ContextualReasoning analyzes events within their broader historical and current context.
// This implements `ContextualReasoning()`.
func (a *Agent) ContextualReasoning(event EventData, historicalContext []MemoryEvent) {
	slog.Info("Performing contextual reasoning", "agent_id", a.ID, "event_type", event.Type)
	// Complex logic here:
	// 1. Query `a.Memory` for relevant historical events based on `event` and current world state.
	// 2. Combine with `historicalContext` passed in.
	// 3. Apply inference rules, probabilistic models, etc., to derive deeper meaning.
	// For now, it's a log statement.
	slog.Debug("Contextual reasoning completed for event", "agent_id", a.ID, "event_id", event.ID, "historical_context_count", len(historicalContext))
}

// PredictConsequences simulates the likely outcomes of potential actions.
// This implements `PredictConsequences()`.
func (a *Agent) PredictConsequences(action ActionPlan, simulationDepth int) {
	slog.Info("Predicting consequences for action", "agent_id", a.ID, "action_id", action.ID, "depth", simulationDepth)
	// This would involve:
	// 1. Using the `a.State.WorldModel` as a simulation environment.
	// 2. Applying the `action` to the model.
	// 3. Iteratively simulating `simulationDepth` steps, predicting subsequent states.
	// 4. Evaluating outcomes (e.g., success probability, resource cost, ethical impact).
	predictedOutcome := map[string]interface{}{
		"success_probability": utils.GenerateRandomFloat(0.6, 0.99),
		"estimated_cost":      utils.GenerateRandomFloat(10, 100),
		"ethical_alignment":   utils.GenerateRandomFloat(0.7, 0.95),
	}
	slog.Info("Predicted consequences", "agent_id", a.ID, "action_id", action.ID, "outcome", predictedOutcome)
	// Store this prediction in memory
	a.memoryEventChan <- MemoryEvent{
		ID:        utils.GenerateUUID(),
		Timestamp: time.Now(),
		Type:      "Prediction",
		Content:   map[string]interface{}{"action_id": action.ID, "prediction_data": predictedOutcome},
		Context:   map[string]string{"simulation_depth": fmt.Sprintf("%d", simulationDepth)},
		Source:    a.ID,
		Relevance: 0.8,
	}
}

// GenerateDecisionRationale provides a human-readable explanation for a specific decision.
// This implements `GenerateDecisionRationale()`.
func (a *Agent) GenerateDecisionRationale(decisionID string) string {
	slog.Info("Generating decision rationale", "agent_id", a.ID, "decision_id", decisionID)
	// This would involve querying `a.Memory` for the decision, associated observations, context,
	// and predicted outcomes, then synthesizing a narrative.
	// For simplicity, returning a mock string.
	rationale := fmt.Sprintf("Decision %s was made based on current observations and a high predicted success probability. Ethical principles were considered.", decisionID)
	slog.Debug("Generated rationale", "agent_id", a.ID, "rationale", rationale)
	return rationale
}

// InterpretAmbiguousQuery resolves vague or incomplete queries.
// This implements `InterpretAmbiguousQuery()`.
func (a *Agent) InterpretAmbiguousQuery(query string, currentContext Context) string {
	slog.Info("Interpreting ambiguous query", "agent_id", a.ID, "query", query, "context", currentContext)
	// This would typically involve NLP models, semantic parsing, and querying internal knowledge
	// or memories to find the most probable intent or meaning.
	// For example, if query is "that thing", context might indicate "the last sensor reading".
	if query == "that thing" && currentContext["last_subject"] != nil {
		resolvedQuery := fmt.Sprintf("Resolved '%s' to '%s'", query, currentContext["last_subject"])
		slog.Debug("Query resolved", "agent_id", a.ID, "resolved_query", resolvedQuery)
		return resolvedQuery
	}
	slog.Debug("Query remains ambiguous", "agent_id", a.ID, "query", query)
	return fmt.Sprintf("Could not fully resolve: '%s'. Please be more specific.", query)
}


// memoryManagementLoop handles storing and retrieving memories.
func (a *Agent) memoryManagementLoop(ctx context.Context) {
	defer a.wg.Done()
	slog.Info("Memory management loop started.", "agent_id", a.ID)

	for {
		select {
		case <-ctx.Done():
			slog.Info("Memory management loop stopped.", "agent_id", a.ID)
			return
		case event := <-a.memoryEventChan:
			a.StoreExperientialMemory(event)
		}
	}
}

// StoreExperientialMemory archives significant events and their outcomes.
// This implements `StoreExperientialMemory()`.
func (a *Agent) StoreExperientialMemory(event MemoryEvent) {
	a.Memory.Store(event)
	slog.Debug("Stored memory event", "agent_id", a.ID, "event_type", event.Type, "event_id", event.ID)
}

// RetrieveRelevantMemories accesses past experiences and knowledge.
// This implements `RetrieveRelevantMemories()`.
func (a *Agent) RetrieveRelevantMemories(query string, context map[string]string) []MemoryEvent {
	slog.Info("Retrieving relevant memories", "agent_id", a.ID, "query", query, "context", context)
	// Use the `a.Memory` module to perform the actual retrieval
	memories := a.Memory.Retrieve(query, context)
	slog.Debug("Retrieved memories", "agent_id", a.ID, "query", query, "count", len(memories))
	return memories
}

// QueryKnowledgeGraph retrieves information from its knowledge base.
// This is a specialized version of `RetrieveRelevantMemories` for graph-like data.
func (a *Agent) QueryKnowledgeGraph(query string) map[string]interface{} {
	slog.Info("Querying knowledge graph", "agent_id", a.ID, "query", query)
	// This would involve a dedicated knowledge graph database (e.g., Neo4j, Dgraph)
	// or an in-memory semantic graph.
	// For this example, we'll simulate a simple query.
	if query == "what is agent status?" {
		return map[string]interface{}{
			"status": a.State.Status,
			"health": a.State.Health,
		}
	}
	// Simulate semantic search in memory
	relevant := a.RetrieveRelevantMemories(query, nil)
	if len(relevant) > 0 {
		return map[string]interface{}{"result": fmt.Sprintf("Found %d relevant memories for '%s'", len(relevant), query)}
	}
	return map[string]interface{}{"result": "No direct knowledge found for query."}
}

// learningLoop processes action results and updates internal models.
func (a *Agent) learningLoop(ctx context.Context) {
	defer a.wg.Done()
	slog.Info("Learning loop started.", "agent_id", a.ID)

	for {
		select {
		case <-ctx.Done():
			slog.Info("Learning loop stopped.", "agent_id", a.ID)
			return
		case result := <-a.actionResultChan:
			slog.Debug("Learning from action result", "agent_id", a.ID, "action_id", result.ActionID, "success", result.Success)
			experience := LearningExperience{
				ID:          utils.GenerateUUID(),
				Timestamp:   time.Now(),
				Action:      ActionPlan{ID: result.ActionID}, // Populate fully if needed
				Outcome:     result,
				// Add relevant observation/context if available
			}
			a.LearnFromExperience(experience)
		case exp := <-a.learningExperienceChan:
			a.LearnFromExperience(exp) // Process explicit learning experiences
		}
	}
}

// LearnFromExperience updates internal models/weights based on outcomes.
// This implements `LearnFromExperience()`.
func (a *Agent) LearnFromExperience(experience LearningExperience) {
	slog.Info("Learning from experience", "agent_id", a.ID, "action_id", experience.Action.ID, "outcome_success", experience.Outcome.Success)
	// This would update internal ML models, reinforcement learning policies,
	// or weights in a decision-making graph.
	// For now, it's a placeholder.
	if experience.Outcome.Success {
		slog.Debug("Experience was successful, reinforcing positive weights.", "agent_id", a.ID)
	} else {
		slog.Debug("Experience failed, adjusting negative weights or exploring alternatives.", "agent_id", a.ID)
	}
	// Store the learning experience itself
	a.memoryEventChan <- MemoryEvent{
		ID:        utils.GenerateUUID(),
		Timestamp: experience.Timestamp,
		Type:      "LearningExperience",
		Content:   map[string]interface{}{"action_id": experience.Action.ID, "outcome": experience.Outcome.Success},
		Context:   map[string]string{"type": "reinforcement"},
		Source:    a.ID,
		Relevance: 1.0,
	}
}

// SelfImproveLearningAlgorithm adjusts its own learning parameters or strategies.
// This implements `SelfImproveLearningAlgorithm()`.
func (a *Agent) SelfImproveLearningAlgorithm(metrics map[string]float64) {
	slog.Info("Self-improving learning algorithm based on metrics", "agent_id", a.ID, "metrics", metrics)
	// This is meta-learning: the agent observes its own learning performance
	// and adjusts its learning strategy (e.g., changing learning rates,
	// switching optimization algorithms, focusing on specific data types).
	// For example: if `decision_accuracy` is low, it might increase `memory_retrieval_depth`.
	if metrics["decision_accuracy"] < 0.90 {
		slog.Warn("Decision accuracy is below threshold, adjusting learning focus!", "agent_id", a.ID)
		a.Config.PerceptionInterval = a.Config.PerceptionInterval / 2 // Perceive more frequently
		slog.Debug("Perception interval adjusted", "new_interval", a.Config.PerceptionInterval)
	}
	slog.Info("Learning algorithm self-improvement complete.", "agent_id", a.ID)
}

// SynthesizeNewKnowledge derives new information or insights from existing data.
// This implements `SynthesizeNewKnowledge()`.
func (a *Agent) SynthesizeNewKnowledge(facts []Fact, inferenceRules []InferenceRule) map[string]interface{} {
	slog.Info("Synthesizing new knowledge", "agent_id", a.ID, "facts_count", len(facts), "rules_count", len(inferenceRules))
	// This would involve a rule engine or a knowledge graph inference engine.
	// For simplicity, simulate a basic inference.
	newKnowledge := make(map[string]interface{})
	if len(facts) > 0 && len(inferenceRules) > 0 {
		for _, rule := range inferenceRules {
			// Very basic simulation: if rule condition matches any fact, infer action.
			for _, fact := range facts {
				if rule.Condition == "IF has_high_latency THEN investigate_network" && fact.Content == "server_A_high_latency" {
					newKnowledge["server_A_status"] = "requires_investigation"
					a.memoryEventChan <- MemoryEvent{
						ID:        utils.GenerateUUID(),
						Timestamp: time.Now(),
						Type:      "KnowledgeSynthesis",
						Content:   newKnowledge,
						Context:   map[string]string{"rule": rule.ID, "fact": fact.ID},
						Source:    a.ID,
						Relevance: 0.9,
					}
					slog.Info("New knowledge synthesized", "agent_id", a.ID, "knowledge", newKnowledge)
					return newKnowledge // Return first synthesized knowledge for simplicity
				}
			}
		}
	}
	slog.Debug("No new knowledge synthesized.", "agent_id", a.ID)
	return newKnowledge
}


// subAgentManagementLoop handles the spawning and monitoring of ephemeral sub-agents.
func (a *Agent) subAgentManagementLoop(ctx context.Context) {
	defer a.wg.Done()
	slog.Info("Sub-agent management loop started.", "agent_id", a.ID)
	activeSubAgents := make(map[string]*EphemeralSubAgent)
	var mu sync.Mutex // Mutex to protect activeSubAgents map

	for {
		select {
		case <-ctx.Done():
			slog.Info("Sub-agent management loop stopped.", "agent_id", a.ID)
			// Signal all active sub-agents to stop, wait for them
			mu.Lock()
			for _, sa := range activeSubAgents {
				slog.Warn("Forcing shutdown of active sub-agent", "sub_agent_id", sa.ID)
				// In a real scenario, a cancel context would be passed to sub-agent
			}
			mu.Unlock()
			return
		case req := <-a.subAgentRequestChan:
			sa := a.createAndRunSubAgent(ctx, req)
			mu.Lock()
			activeSubAgents[sa.ID] = sa
			mu.Unlock()
		case result := <-a.subAgentResultChan:
			mu.Lock()
			if sa, ok := activeSubAgents[result.SubAgentID]; ok {
				slog.Info("Received result from sub-agent", "sub_agent_id", result.SubAgentID, "success", result.Success)
				sa.Status = "completed"
				if !result.Success {
					sa.Status = "failed"
					slog.Error("Sub-agent task failed", "sub_agent_id", result.SubAgentID, "error", result.Error)
					a.SelfDiagnoseAndRepair("sub_agent_module", []ErrorData{{Component: "sub_agent", Message: fmt.Sprintf("Sub-agent %s failed: %s", result.SubAgentID, result.Error)}})
				}
				// Process sub-agent's output, e.g., update world model, trigger new actions
				a.UpdateInternalWorldModel([]Observation{
					{
						ID:        utils.GenerateUUID(),
						SensorID:  fmt.Sprintf("sub_agent:%s", result.SubAgentID),
						Timestamp: time.Now(),
						DataType:  "SubAgentReport",
						Content:   result.Output,
						Certainty: 0.99, // Assuming sub-agent results are highly certain
					},
				})
				delete(activeSubAgents, result.SubAgentID)
			}
			mu.Unlock()
		case <-time.After(1 * time.Second): // Periodically check for expired sub-agents
			mu.Lock()
			for id, sa := range activeSubAgents {
				if time.Since(sa.CreatedAt) > sa.Lifespan && sa.Status == "running" {
					slog.Warn("Sub-agent lifespan expired, forcefully terminating", "sub_agent_id", id, "task", sa.Task.Task)
					sa.Status = "terminated_expired"
					// In a real scenario, actively terminate the sub-agent goroutine
					delete(activeSubAgents, id)
				}
			}
			mu.Unlock()
		}
	}
}

// createAndRunSubAgent creates and deploys a temporary, highly specialized sub-agent instance.
// This implements `SpawnEphemeralSubAgent()`.
func (a *Agent) createAndRunSubAgent(ctx context.Context, req Request) *EphemeralSubAgent {
	subAgent := &EphemeralSubAgent{
		ID:        utils.GenerateUUID(),
		ParentID:  a.ID,
		Task:      req,
		Status:    "running",
		Lifespan:  req.Args["lifespan"] == "" ? 5*time.Second : utils.ParseDurationOrDefault(req.Args["lifespan"], 5*time.Second), // Default lifespan
		CreatedAt: time.Now(),
		ResultChannel: a.subAgentResultChan,
	}
	slog.Info("Spawning ephemeral sub-agent", "agent_id", a.ID, "sub_agent_id", subAgent.ID, "task", subAgent.Task.Task, "lifespan", subAgent.Lifespan)

	// In a real system, this would involve creating a new goroutine or even a new process/container
	// with limited scope and resources.
	go func(sa *EphemeralSubAgent) {
		slog.Debug("Sub-agent started its task", "sub_agent_id", sa.ID, "task", sa.Task.Task)
		// Simulate sub-agent work
		time.Sleep(time.Duration(utils.GenerateRandomInt(1, 4)) * time.Second) // Task takes 1-4 seconds

		result := EphemeralSubAgentResult{
			SubAgentID: sa.ID,
			Success:    true,
			Output:     map[string]interface{}{"task_result": fmt.Sprintf("Successfully processed %s", sa.Task.Task)},
		}
		if utils.GenerateRandomFloat(0,1) < 0.1 { // 10% chance of failure
			result.Success = false
			result.Error = "Simulated sub-agent failure due to resource contention"
			result.Output = nil
		}

		sa.ResultChannel <- result // Send result back to parent agent
		slog.Debug("Sub-agent completed its task", "sub_agent_id", sa.ID, "task", sa.Task.Task, "success", result.Success)
	}(subAgent)
	return subAgent
}

// SpawnEphemeralSubAgent is the external function to request a sub-agent.
// This implements `SpawnEphemeralSubAgent()`.
func (a *Agent) SpawnEphemeralSubAgent(task Request, lifespan time.Duration) {
	slog.Info("Requesting to spawn ephemeral sub-agent", "agent_id", a.ID, "task", task.Task, "requested_lifespan", lifespan)
	if task.Args == nil {
		task.Args = make(map[string]string)
	}
	task.Args["lifespan"] = lifespan.String() // Pass lifespan as arg
	a.subAgentRequestChan <- task
}

// ProposeEthicalResolution evaluates actions against ethical guidelines.
// This implements `ProposeEthicalResolution()`.
func (a *Agent) ProposeEthicalResolution(dilemma EthicalDilemma, options []ActionPlan) []ActionPlan {
	slog.Info("Proposing ethical resolution for dilemma", "agent_id", a.ID, "dilemma_id", dilemma.ID, "options_count", len(options))

	type WeightedOption struct {
		ActionPlan
		EthicalScore float64
	}
	weightedOptions := make([]WeightedOption, len(options))

	for i, opt := range options {
		ethicalScore := 0.0
		// Simulate ethical evaluation based on principles
		for _, principle := range a.Config.EthicalPrinciples {
			// Very simplified: check if action description implies harm or transparency issues
			if principle.Name == "DoNoHarm" && (opt.Description == "Allocate A to B" || opt.Description == "Option A") {
				// Assuming option A might cause more harm in this specific dilemma context
				ethicalScore += 0.2 * principle.Weight // Lower score
			} else if principle.Name == "Transparency" && opt.Description == "Option B" {
				// Assuming option B is more transparent
				ethicalScore += 0.9 * principle.Weight
			} else {
				ethicalScore += 0.7 * principle.Weight // Default
			}
		}
		weightedOptions[i] = WeightedOption{ActionPlan: opt, EthicalScore: ethicalScore}
		slog.Debug("Evaluated option", "action_id", opt.ID, "ethical_score", ethicalScore)
	}

	// Sort options by ethical score (descending)
	for i := 0; i < len(weightedOptions); i++ {
		for j := i + 1; j < len(weightedOptions); j++ {
			if weightedOptions[i].EthicalScore < weightedOptions[j].EthicalScore {
				weightedOptions[i], weightedOptions[j] = weightedOptions[j], weightedOptions[i]
			}
		}
	}

	slog.Info("Ethical resolution proposed (sorted by score)", "agent_id", a.ID, "top_option", weightedOptions[0].Description, "score", weightedOptions[0].EthicalScore)
	// Trigger human intervention if ethical score is too low or highly conflicting
	if weightedOptions[0].EthicalScore < 0.5 {
		a.RequestHumanIntervention("Low ethical score for all options", map[string]interface{}{
			"dilemma": dilemma.Description, "options": weightedOptions,
		})
	}

	// Return the sorted action plans (highest ethical score first)
	resolvedPlans := make([]ActionPlan, len(weightedOptions))
	for i, wo := range weightedOptions {
		resolvedPlans[i] = wo.ActionPlan
	}
	return resolvedPlans
}

// RequestHumanIntervention notifies a human for complex/critical situations.
// This implements `RequestHumanIntervention()`.
func (a *Agent) RequestHumanIntervention(reason string, data map[string]interface{}) {
	slog.Warn("Human intervention requested!", "agent_id", a.ID, "reason", reason, "data", data)
	// Send an alert via MCP
	alertMsg := mcp.AgentMessage{
		ID:          utils.GenerateUUID(),
		Sender:      a.ID,
		Recipient:   "HumanOperator-1", // Example human recipient
		Type:        "HumanInterventionRequest",
		Payload:     map[string]interface{}{"reason": reason, "details": data},
		Timestamp:   time.Now(),
		ChannelHint: "http", // Or specific alerting channel
		Priority:    mcp.PriorityHigh,
	}
	a.agentToMcpChan <- alertMsg
}

// AdaptCommunicationPersona adjusts communication style and behavior.
// This implements `AdaptCommunicationPersona()`.
func (a *Agent) AdaptCommunicationPersona(recipientProfile AgentProfile, communicationGoal CommunicationGoal) {
	slog.Info("Adapting communication persona", "agent_id", a.ID, "recipient_type", recipientProfile.Type, "goal", communicationGoal)
	// This would involve dynamically loading different language models, tone adjusters,
	// or response templates based on the recipient's profile and the communication objective.
	// For example:
	if recipientProfile.Type == "human_user" && communicationGoal == GoalInform {
		slog.Debug("Adopting formal, clear, and concise persona for human information sharing.")
		// Update internal text generation parameters
	} else if recipientProfile.Type == "technical_agent" && communicationGoal == GoalQuery {
		slog.Debug("Adopting technical, direct, and API-oriented persona for agent query.")
		// Update internal text generation parameters
	}
	slog.Info("Communication persona adapted.", "agent_id", a.ID)
}

// selfMonitoringLoop continuously checks agent health, optimizes resources.
func (a *Agent) selfMonitoringLoop(ctx context.Context) {
	defer a.wg.Done()
	slog.Info("Self-monitoring loop started.", "agent_id", a.ID)
	ticker := time.NewTicker(5 * time.Second) // Check every 5 seconds
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			slog.Info("Self-monitoring loop stopped.", "agent_id", a.ID)
			return
		case <-ticker.C:
			// Simulate health check for a component
			component := "cognition_engine"
			if utils.GenerateRandomFloat(0,1) < 0.05 { // 5% chance of a simulated error
				a.State.Health[component] = "degraded"
				a.SelfDiagnoseAndRepair(component, []ErrorData{{Component: component, Message: "Simulated high latency", Level: "warning"}})
			} else {
				a.State.Health[component] = "healthy"
			}

			// Simulate dynamic resource allocation
			a.DynamicResourceAllocation("critical_task", PriorityCritical, ResourceConstraints{CPULimit: 0.8, MemoryLimit: 512 * 1024 * 1024}) // 512MB
		}
	}
}


// SelfDiagnoseAndRepair identifies internal malfunctions and attempts autonomous recovery.
// This implements `SelfDiagnoseAndRepair()`.
func (a *Agent) SelfDiagnoseAndRepair(componentID string, symptoms []ErrorData) {
	slog.Warn("Self-diagnosis triggered for component", "agent_id", a.ID, "component_id", componentID, "symptoms_count", len(symptoms))
	// In a real system, this would involve:
	// 1. Analyzing `symptoms` to determine root cause.
	// 2. Consulting a knowledge base of repair procedures.
	// 3. Attempting corrective actions (e.g., restarting a goroutine, reconfiguring parameters, rolling back updates).
	if componentID == "action_module" && len(symptoms) > 0 && symptoms[0].Message == "Simulated action failure" {
		slog.Info("Attempting to reconfigure action module parameters due to failure.", "agent_id", a.ID)
		// Simulate reconfiguration
		time.Sleep(500 * time.Millisecond)
		slog.Info("Action module parameters reconfigured. Monitoring for recovery.", "agent_id", a.ID)
		a.State.Health[componentID] = "reconfigured"
		return
	}
	slog.Info("No specific repair procedure found, escalating or logging for manual review.", "agent_id", a.ID)
	a.State.Health[componentID] = "error" // Mark as error if no repair
}

// DynamicResourceAllocation optimizes the assignment of internal/external resources.
// This implements `DynamicResourceAllocation()`.
func (a *Agent) DynamicResourceAllocation(task string, priority Priority, constraints ResourceConstraints) {
	slog.Info("Dynamically allocating resources", "agent_id", a.ID, "task", task, "priority", priority, "constraints", constraints)
	// This would involve:
	// 1. Monitoring current resource usage (CPU, memory, network).
	// 2. Evaluating `priority` and `constraints` of the `task`.
	// 3. Adjusting limits for internal goroutines or requesting/releasing external compute.
	// For simplicity, simulate a resource adjustment.
	currentCPU := utils.GenerateRandomFloat(0.1, 0.7) // Simulate current usage
	if currentCPU > constraints.CPULimit {
		slog.Warn("CPU usage is high for task, attempting to scale down non-critical processes or request more resources.",
			"agent_id", a.ID, "task", task, "current_cpu", fmt.Sprintf("%.2f", currentCPU), "limit", fmt.Sprintf("%.2f", constraints.CPULimit))
		// Log this as a memory event
		a.memoryEventChan <- MemoryEvent{
			ID:        utils.GenerateUUID(),
			Timestamp: time.Now(),
			Type:      "ResourceAdjustment",
			Content:   map[string]interface{}{"task": task, "adjustment": "scale_down", "reason": "cpu_limit_exceeded"},
			Source:    a.ID,
			Relevance: 0.7,
		}
	} else {
		slog.Debug("Resources seem adequate for task.", "agent_id", a.ID, "task", task)
	}
	slog.Info("Dynamic resource allocation check complete.", "agent_id", a.ID)
}

// CollaborateWithExternalAgent initiates and manages collaborative tasks.
// This implements `CollaborateWithExternalAgent()`.
func (a *Agent) CollaborateWithExternalAgent(agentID string, task SharedTask, protocol string) {
	slog.Info("Initiating collaboration with external agent", "agent_id", a.ID, "external_agent", agentID, "task", task.Description, "protocol", protocol)
	// This would involve:
	// 1. Sending a `CollaborationRequest` message via MCP to `agentID`.
	// 2. Establishing a sub-protocol or shared communication channel.
	// 3. Exchanging `SharedTask` updates.
	collabMsg := mcp.AgentMessage{
		ID:        utils.GenerateUUID(),
		Sender:    a.ID,
		Recipient: agentID,
		Type:      "CollaborationRequest",
		Payload:   map[string]interface{}{"task_id": task.ID, "description": task.Description, "protocol": protocol},
		Timestamp: time.Now(),
		// ChannelHint would depend on the `protocol` and `agentID`'s capabilities
		ChannelHint: "grpc", // Example
		Priority:    mcp.PriorityMedium,
	}
	a.agentToMcpChan <- collabMsg
	slog.Info("Collaboration request sent.", "agent_id", a.ID)
}

// AuditDecisionTrace provides a complete, immutable record of inputs, processing, and outputs.
// This implements `AuditDecisionTrace()`.
func (a *Agent) AuditDecisionTrace(decisionID string) map[string]interface{} {
	slog.Info("Auditing decision trace", "agent_id", a.ID, "decision_id", decisionID)
	// This would involve retrieving all related `MemoryEvent`s (observations, decisions, actions, predictions, learning events)
	// that led to a specific `decisionID`. It effectively reconstructs the agent's thought process.
	relatedMemories := a.Memory.Retrieve("", map[string]string{"decision_id": decisionID}) // Assuming Memory can query by decisionID
	auditRecord := map[string]interface{}{
		"decision_id": decisionID,
		"timestamp":   time.Now(),
		"inputs":      []map[string]interface{}{},
		"process_steps": []map[string]interface{}{},
		"outputs":     []map[string]interface{}{},
		"related_memories_count": len(relatedMemories),
	}

	// Populate auditRecord from relatedMemories
	for _, mem := range relatedMemories {
		switch mem.Type {
		case "Observation":
			auditRecord["inputs"] = append(auditRecord["inputs"].([]map[string]interface{}), mem.Content)
		case "Decision", "Prediction", "KnowledgeSynthesis":
			auditRecord["process_steps"] = append(auditRecord["process_steps"].([]map[string]interface{}), mem.Content)
		case "ActionOutcome":
			auditRecord["outputs"] = append(auditRecord["outputs"].([]map[string]interface{}), mem.Content)
		}
	}

	slog.Info("Decision trace audited.", "agent_id", a.ID, "decision_id", decisionID, "audit_details_count", len(relatedMemories))
	return auditRecord
}

// ====================================================================================================
// agent/memory.go
// ====================================================================================================

package agent

import (
	"log/slog"
	"sync"
	"time"

	"ai-agent/utils"
)

// MemoryModule manages the agent's long-term and short-term memory.
type MemoryModule struct {
	sync.RWMutex
	store []MemoryEvent // Simple in-memory slice for demonstration
	// In a real system, this would be backed by a database, knowledge graph, or vector store.
}

// NewMemoryModule creates a new memory module.
func NewMemoryModule() *MemoryModule {
	return &MemoryModule{
		store: make([]MemoryEvent, 0, 1000), // Pre-allocate some capacity
	}
}

// Store adds a new MemoryEvent to the memory.
func (m *MemoryModule) Store(event MemoryEvent) {
	m.Lock()
	defer m.Unlock()
	m.store = append(m.store, event)
	slog.Debug("Memory stored", "event_type", event.Type, "id", event.ID, "size", len(m.store))
}

// Retrieve fetches relevant memories based on query and context.
func (m *MemoryModule) Retrieve(query string, context map[string]string) []MemoryEvent {
	m.RLock()
	defer m.RUnlock()

	var relevant []MemoryEvent
	// This is a highly simplified retrieval mechanism.
	// In a real system, this would involve semantic search, vector embeddings,
	// knowledge graph traversals, or complex filtering based on time, relevance, and content.
	for _, event := range m.store {
		// Simple keyword match for query
		if query == "" || utils.ContainsKeyword(fmt.Sprintf("%v", event.Content), query) || utils.ContainsKeyword(event.Type, query) {
			// Simple context match
			contextMatch := true
			if context != nil {
				for k, v := range context {
					if val, ok := event.Context[k]; !ok || val != v {
						contextMatch = false
						break
					}
				}
			}
			if contextMatch {
				relevant = append(relevant, event)
			}
		}
	}
	slog.Debug("Memory retrieved", "query", query, "context", context, "results", len(relevant))
	return relevant
}

// PersistAll saves all current memories to a persistent storage.
func (m *MemoryModule) PersistAll() error {
	m.RLock()
	defer m.RUnlock()
	slog.Info("Persisting all memories (simulated)...", "count", len(m.store))
	// In a real application, this would write to a database, file, or cloud storage.
	time.Sleep(500 * time.Millisecond) // Simulate I/O operation
	slog.Info("Memories successfully persisted (simulated).")
	return nil
}

// ClearOlderThan removes memories older than a specified duration.
func (m *MemoryModule) ClearOlderThan(duration time.Duration) {
	m.Lock()
	defer m.Unlock()
	cutoff := time.Now().Add(-duration)
	var newStore []MemoryEvent
	for _, event := range m.store {
		if event.Timestamp.After(cutoff) {
			newStore = append(newStore, event)
		}
	}
	slog.Info("Cleared old memories", "before_count", len(m.store), "after_count", len(newStore))
	m.store = newStore
}


// ====================================================================================================
// mcp/mcp.go
// ====================================================================================================

package mcp

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// ChannelAdapter defines the interface for any communication channel ChronoMind uses.
// This is the core of the Multi-Channel Protocol.
type ChannelAdapter interface {
	ID() string
	Start(ctx context.Context) error                          // Start listening/connecting
	Stop() error                                            // Stop and clean up
	Send(msg AgentMessage) error                            // Send a message
	Receive() (<-chan AgentMessage)                           // Get a channel for incoming messages
}

// MessageDispatcher manages all registered ChannelAdapters and routes messages.
type MessageDispatcher struct {
	sync.RWMutex
	adapters        map[string]ChannelAdapter // ChannelID -> Adapter
	agentInbound    chan AgentMessage         // For messages coming INTO the agent core
	agentOutbound   chan AgentMessage         // For messages going FROM the agent core
	cancelFuncs     map[string]context.CancelFunc // To stop individual channel adapters
	wg              sync.WaitGroup
}

// NewMessageDispatcher creates a new MessageDispatcher.
func NewMessageDispatcher(agentInbound, agentOutbound chan AgentMessage) *MessageDispatcher {
	return &MessageDispatcher{
		adapters:        make(map[string]ChannelAdapter),
		agentInbound:    agentInbound,
		agentOutbound:   agentOutbound,
		cancelFuncs:     make(map[string]context.CancelFunc),
	}
}

// RegisterChannel adds a new ChannelAdapter to the dispatcher.
func (md *MessageDispatcher) RegisterChannel(id string, adapter ChannelAdapter) {
	md.Lock()
	defer md.Unlock()
	if _, exists := md.adapters[id]; exists {
		slog.Warn("Channel adapter already registered, overwriting.", "channel_id", id)
	}
	md.adapters[id] = adapter
	slog.Info("MCP Channel adapter registered.", "channel_id", id)
}

// DeregisterChannel removes a ChannelAdapter from the dispatcher.
func (md *MessageDispatcher) DeregisterChannel(id string) {
	md.Lock()
	defer md.Unlock()
	if adapter, exists := md.adapters[id]; exists {
		if cancel, ok := md.cancelFuncs[id]; ok {
			cancel() // Signal to stop the channel's receive loop
			delete(md.cancelFuncs, id)
		}
		adapter.Stop() // Stop the adapter gracefully
		delete(md.adapters, id)
		slog.Info("MCP Channel adapter deregistered.", "channel_id", id)
	}
}

// Start initiates all registered channel adapters and begins message routing.
func (md *MessageDispatcher) Start(ctx context.Context) {
	slog.Info("Starting MCP Message Dispatcher.")
	md.RLock()
	defer md.RUnlock()

	// Start all registered adapters
	for id, adapter := range md.adapters {
		go func(adapter ChannelAdapter) {
			if err := adapter.Start(ctx); err != nil {
				slog.Error("Failed to start channel adapter", "channel_id", adapter.ID(), "error", err)
				return
			}
			slog.Info("Channel adapter started.", "channel_id", adapter.ID())

			// Start a goroutine to continuously receive messages from this adapter
			md.wg.Add(1)
			go func() {
				defer md.wg.Done()
				channelCtx, channelCancel := context.WithCancel(ctx)
				md.Lock()
				md.cancelFuncs[adapter.ID()] = channelCancel
				md.Unlock()

				for {
					select {
					case <-channelCtx.Done():
						slog.Info("Stopping receive loop for channel adapter.", "channel_id", adapter.ID())
						return
					case msg := <-adapter.Receive():
						slog.Debug("Received message from channel, forwarding to agent.", "channel_id", adapter.ID(), "message_id", msg.ID)
						md.agentInbound <- msg // Forward to agent's inbound channel
					}
				}
			}()
		}(adapter)
	}

	// Start a goroutine to continuously send messages from agent core to adapters
	md.wg.Add(1)
	go func() {
		defer md.wg.Done()
		for {
			select {
			case <-ctx.Done():
				slog.Info("Stopping outbound message routing.", "dispatcher_id", "main")
				return
			case msg := <-md.agentOutbound:
				md.routeAndSend(msg)
			}
		}
	}()
	slog.Info("MCP Message Dispatcher started all internal routing goroutines.")
}

// Stop gracefully stops the dispatcher and all its adapters.
func (md *MessageDispatcher) Stop() {
	slog.Info("Stopping MCP Message Dispatcher.")
	md.Lock()
	defer md.Unlock()

	// Signal all channel receive loops to stop
	for _, cancel := range md.cancelFuncs {
		cancel()
	}

	// Stop all adapters
	for id, adapter := range md.adapters {
		if err := adapter.Stop(); err != nil {
			slog.Error("Error stopping channel adapter", "channel_id", id, "error", err)
		} else {
			slog.Info("Channel adapter stopped.", "channel_id", id)
		}
	}
	md.wg.Wait() // Wait for all goroutines to finish
	slog.Info("MCP Message Dispatcher fully stopped.")
}

// SendMessage routes an AgentMessage to the appropriate ChannelAdapter.
func (md *MessageDispatcher) SendMessage(msg AgentMessage) error {
	md.RLock()
	defer md.RUnlock()

	adapterID := msg.ChannelHint
	if adapterID == "" {
		slog.Warn("No ChannelHint provided, attempting to send via a default channel (e.g., first available).", "message_id", msg.ID)
		// Fallback to a default if hint is missing (e.g., the first registered)
		for id, adapter := range md.adapters {
			adapterID = id
			return adapter.Send(msg)
		}
		return fmt.Errorf("no channel hint and no registered adapters to send message %s", msg.ID)
	}

	if adapter, ok := md.adapters[adapterID]; ok {
		return adapter.Send(msg)
	}
	return fmt.Errorf("no channel adapter registered for ID: %s", adapterID)
}

// routeAndSend is an internal function used by the outbound routing goroutine.
func (md *MessageDispatcher) routeAndSend(msg AgentMessage) {
	if err := md.SendMessage(msg); err != nil {
		slog.Error("Failed to route and send outbound MCP message", "error", err, "message_id", msg.ID, "channel_hint", msg.ChannelHint)
	} else {
		slog.Debug("Successfully routed and sent outbound MCP message", "message_id", msg.ID, "channel_hint", msg.ChannelHint)
	}
}


// ====================================================================================================
// mcp/messages.go
// ====================================================================================================

package mcp

import (
	"encoding/json"
	"time"
)

// AgentMessage defines the standardized structure for all messages exchanged via the MCP.
type AgentMessage struct {
	ID            string                 `json:"id"`             // Unique message ID
	Sender        string                 `json:"sender"`         // ID of the sender agent/entity
	Recipient     string                 `json:"recipient"`      `json:",omitempty"` // ID of the recipient agent/entity (optional for broadcast)
	Type          string                 `json:"type"`           // Type of message (e.g., "TaskRequest", "SensorData", "Query")
	Payload       interface{}            `json:"payload"`        // The actual content of the message
	Timestamp     time.Time              `json:"timestamp"`      // Time of message creation
	ChannelHint   string                 `json:"channel_hint"`   `json:",omitempty"` // Hint for preferred communication channel (e.g., "http", "grpc", "nats")
	CorrelationID string                 `json:"correlation_id"` `json:",omitempty"` // For linking request/response
	Priority      Priority               `json:"priority"`       `json:",omitempty"` // Message priority
}

// Priority of an MCP message
type Priority string

const (
	PriorityLow    Priority = "low"
	PriorityMedium Priority = "medium"
	PriorityHigh   Priority = "high"
	PriorityCritical Priority = "critical"
)

// ToJSON marshals the AgentMessage into a JSON byte slice.
func (m AgentMessage) ToJSON() ([]byte, error) {
	return json.Marshal(m)
}

// FromJSON unmarshals a JSON byte slice into an AgentMessage.
func FromJSON(data []byte) (AgentMessage, error) {
	var msg AgentMessage
	err := json.Unmarshal(data, &msg)
	return msg, err
}

// ====================================================================================================
// mcp/channels/http.go
// ====================================================================================================

package channels

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"ai-agent/mcp"
)

// HTTPAdapter implements the ChannelAdapter interface for HTTP communication.
// It can act as both an HTTP server (receiving messages) and an HTTP client (sending messages).
type HTTPAdapter struct {
	id         string
	server     *http.Server
	addr       string
	inbound    chan mcp.AgentMessage
	client     *http.Client
	mu         sync.Mutex // Protects handlers map
	handlers   map[string]http.HandlerFunc
	isStarted  bool
}

// NewHTTPAdapter creates a new HTTPAdapter.
func NewHTTPAdapter(addr string) *HTTPAdapter {
	return &HTTPAdapter{
		id:       "http", // Fixed ID for HTTP adapter
		addr:     addr,
		inbound:  make(chan mcp.AgentMessage, 100), // Buffered channel for incoming messages
		client:   &http.Client{Timeout: 10 * time.Second},
		handlers: make(map[string]http.HandlerFunc),
	}
}

// ID returns the identifier for this channel adapter.
func (h *HTTPAdapter) ID() string {
	return h.id
}

// Start initiates the HTTP server for receiving messages.
func (h *HTTPAdapter) Start(ctx context.Context) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.isStarted {
		return fmt.Errorf("HTTP adapter already started")
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/mcp", h.handleIncomingMCPMessage) // Main endpoint for MCP messages
	// You can add more specific handlers if needed
	// mux.HandleFunc("/status", h.handleStatus)

	h.server = &http.Server{
		Addr:    h.addr,
		Handler: mux,
	}

	go func() {
		slog.Info("HTTP Adapter server starting", "addr", h.addr)
		if err := h.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("HTTP Adapter server failed to start", "error", err)
		}
		slog.Info("HTTP Adapter server shut down.", "addr", h.addr)
	}()

	h.isStarted = true
	return nil
}

// Stop shuts down the HTTP server.
func (h *HTTPAdapter) Stop() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if !h.isStarted {
		return fmt.Errorf("HTTP adapter not started")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := h.server.Shutdown(ctx); err != nil {
		slog.Error("HTTP Adapter server shutdown error", "error", err)
		return err
	}
	close(h.inbound) // Close inbound channel
	h.isStarted = false
	slog.Info("HTTP Adapter server gracefully stopped.")
	return nil
}

// Send sends an AgentMessage via HTTP POST to the recipient specified in the message.
func (h *HTTPAdapter) Send(msg mcp.AgentMessage) error {
	// For HTTP, the recipient determines the target URL.
	// This is a simplification; in a real distributed system, a registry would map Recipient IDs to URLs.
	// For now, assume the Recipient's ID is its base URL for receiving MCP messages.
	targetURL := fmt.Sprintf("http://%s/mcp", msg.Recipient) // Example: recipient could be "localhost:8080"
	if msg.Recipient == "" {
		return fmt.Errorf("cannot send HTTP message: recipient ID is empty")
	}
	
	// If the message is intended for the agent running this adapter, send it internally.
	// This prevents infinite loops if an agent tries to send to itself via HTTP by mistake.
	if msg.Recipient == h.server.Addr { // Check if recipient address matches this server's address
		slog.Warn("Attempted to send HTTP message to self, handling internally.", "recipient", msg.Recipient)
		h.inbound <- msg
		return nil
	}

	jsonData, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("failed to marshal AgentMessage to JSON: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, targetURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := h.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send HTTP message to %s: %w", targetURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("received non-OK response (%d) from %s: %s", resp.StatusCode, targetURL, string(body))
	}

	slog.Debug("HTTP message sent successfully", "target", targetURL, "message_id", msg.ID)
	return nil
}

// Receive returns a channel for incoming messages.
func (h *HTTPAdapter) Receive() <-chan mcp.AgentMessage {
	return h.inbound
}

// handleIncomingMCPMessage is the HTTP handler for incoming AgentMessages.
func (h *HTTPAdapter) handleIncomingMCPMessage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error reading request body: %v", err), http.StatusInternalServerError)
		return
	}

	var msg mcp.AgentMessage
	if err := json.Unmarshal(body, &msg); err != nil {
		http.Error(w, fmt.Sprintf("Error unmarshalling JSON: %v", err), http.StatusBadRequest)
		return
	}

	// Add the channel hint for the dispatcher to know it came via HTTP
	msg.ChannelHint = h.ID()

	// Non-blocking send to the inbound channel
	select {
	case h.inbound <- msg:
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "Message received and queued.")
		slog.Debug("Incoming HTTP message queued successfully", "message_id", msg.ID, "sender", msg.Sender)
	case <-time.After(1 * time.Second): // Timeout if channel is full
		http.Error(w, "Agent inbound channel full, message dropped.", http.StatusServiceUnavailable)
		slog.Error("Incoming HTTP message dropped, inbound channel full", "message_id", msg.ID, "sender", msg.Sender)
	}
}


// ====================================================================================================
// mcp/channels/grpc.go
// ====================================================================================================

package channels

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"sync"
	"time"

	"ai-agent/mcp"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "ai-agent/mcp/proto" // Generated protobuf code
)

// ensure interface compliance
var _ mcp.ChannelAdapter = (*GRPCAdapter)(nil)
var _ pb.MCPServiceServer = (*GRPCAdapter)(nil)

// GRPCAdapter implements the ChannelAdapter interface for gRPC communication.
type GRPCAdapter struct {
	pb.UnimplementedMCPServiceServer // Required for gRPC service
	id            string
	addr          string
	grpcServer    *grpc.Server
	inbound       chan mcp.AgentMessage
	clientConn    *grpc.ClientConn
	client        pb.MCPServiceClient // Client to send messages to other gRPC agents
	mu            sync.Mutex
	isStarted     bool
}

// NewGRPCAdapter creates a new GRPCAdapter.
func NewGRPCAdapter(addr string) *GRPCAdapter {
	return &GRPCAdapter{
		id:      "grpc",
		addr:    addr,
		inbound: make(chan mcp.AgentMessage, 100),
	}
}

// ID returns the identifier for this channel adapter.
func (g *GRPCAdapter) ID() string {
	return g.id
}

// Start initiates the gRPC server for receiving messages.
func (g *GRPCAdapter) Start(ctx context.Context) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.isStarted {
		return fmt.Errorf("gRPC adapter already started")
	}

	lis, err := net.Listen("tcp", g.addr)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", g.addr, err)
	}

	g.grpcServer = grpc.NewServer()
	pb.RegisterMCPServiceServer(g.grpcServer, g) // Register this adapter as the service handler

	go func() {
		slog.Info("gRPC Adapter server starting", "addr", g.addr)
		if err := g.grpcServer.Serve(lis); err != nil && err != grpc.ErrServerStopped {
			slog.Error("gRPC Adapter server failed to serve", "error", err)
		}
		slog.Info("gRPC Adapter server stopped.", "addr", g.addr)
	}()

	g.isStarted = true
	return nil
}

// Stop shuts down the gRPC server and client connections.
func (g *GRPCAdapter) Stop() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if !g.isStarted {
		return fmt.Errorf("gRPC adapter not started")
	}

	if g.grpcServer != nil {
		g.grpcServer.GracefulStop()
		slog.Info("gRPC Adapter server gracefully stopped.")
	}
	if g.clientConn != nil {
		g.clientConn.Close()
		slog.Info("gRPC client connection closed.")
	}
	close(g.inbound)
	g.isStarted = false
	return nil
}

// Send sends an AgentMessage via gRPC.
func (g *GRPCAdapter) Send(msg mcp.AgentMessage) error {
	// For gRPC, the recipient determines the target address.
	// Again, a simplification: assume Recipient ID can be directly resolved to an address.
	targetAddr := msg.Recipient // Example: "localhost:50051"

	if targetAddr == "" {
		return fmt.Errorf("cannot send gRPC message: recipient address is empty")
	}

	// If the message is intended for the agent running this adapter, send it internally.
	if msg.Recipient == g.addr {
		slog.Warn("Attempted to send gRPC message to self, handling internally.", "recipient", msg.Recipient)
		g.inbound <- msg
		return nil
	}

	// Establish client connection if not already established or if target changes
	if g.clientConn == nil || g.clientConn.Target() != targetAddr {
		if g.clientConn != nil {
			g.clientConn.Close() // Close old connection if target changed
		}
		conn, err := grpc.Dial(targetAddr, grpc.WithInsecure()) // grpc.WithInsecure() for development
		if err != nil {
			return fmt.Errorf("failed to dial gRPC server at %s: %w", targetAddr, err)
		}
		g.clientConn = conn
		g.client = pb.NewMCPServiceClient(conn)
		slog.Info("Established new gRPC client connection.", "target", targetAddr)
	}

	payloadBytes, err := json.Marshal(msg.Payload)
	if err != nil {
		return fmt.Errorf("failed to marshal payload: %w", err)
	}

	pbMsg := &pb.AgentMessage{
		Id:            msg.ID,
		Sender:        msg.Sender,
		Recipient:     msg.Recipient,
		Type:          msg.Type,
		Payload:       payloadBytes,
		Timestamp:     msg.Timestamp.Format(time.RFC3339Nano),
		ChannelHint:   msg.ChannelHint,
		CorrelationId: msg.CorrelationID,
		Priority:      string(msg.Priority),
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err = g.client.SendMessage(ctx, pbMsg)
	if err != nil {
		return fmt.Errorf("failed to send gRPC message to %s: %w", targetAddr, err)
	}

	slog.Debug("gRPC message sent successfully", "target", targetAddr, "message_id", msg.ID)
	return nil
}

// Receive returns a channel for incoming messages.
func (g *GRPCAdapter) Receive() <-chan mcp.AgentMessage {
	return g.inbound
}

// SendMessage implements the gRPC service method for receiving messages.
func (g *GRPCAdapter) SendMessage(ctx context.Context, pbMsg *pb.AgentMessage) (*pb.SendMessageResponse, error) {
	var payload interface{}
	if err := json.Unmarshal(pbMsg.Payload, &payload); err != nil {
		slog.Error("Error unmarshalling gRPC payload", "error", err, "message_id", pbMsg.Id)
		return nil, status.Errorf(codes.InvalidArgument, "invalid payload format: %v", err)
	}

	timestamp, err := time.Parse(time.RFC3339Nano, pbMsg.Timestamp)
	if err != nil {
		slog.Error("Error parsing gRPC timestamp", "error", err, "message_id", pbMsg.Id)
		return nil, status.Errorf(codes.InvalidArgument, "invalid timestamp format: %v", err)
	}

	msg := mcp.AgentMessage{
		ID:            pbMsg.Id,
		Sender:        pbMsg.Sender,
		Recipient:     pbMsg.Recipient,
		Type:          pbMsg.Type,
		Payload:       payload,
		Timestamp:     timestamp,
		ChannelHint:   g.ID(), // Always set the hint to gRPC for messages received by this adapter
		CorrelationID: pbMsg.CorrelationId,
		Priority:      mcp.Priority(pbMsg.Priority),
	}

	select {
	case g.inbound <- msg:
		slog.Debug("Incoming gRPC message queued successfully", "message_id", msg.ID, "sender", msg.Sender)
		return &pb.SendMessageResponse{Success: true, Message: "Message received and queued."}, nil
	case <-ctx.Done():
		slog.Error("Incoming gRPC message dropped, context cancelled", "message_id", msg.ID, "sender", msg.Sender)
		return nil, status.Errorf(codes.Canceled, "server context cancelled before message could be processed")
	case <-time.After(1 * time.Second): // Timeout if channel is full
		slog.Error("Incoming gRPC message dropped, inbound channel full", "message_id", msg.ID, "sender", msg.Sender)
		return nil, status.Errorf(codes.ResourceExhausted, "agent inbound channel full, message dropped")
	}
}


// ====================================================================================================
// mcp/proto/mcp.proto (content for this file - then needs `protoc` to generate go files)
// ====================================================================================================

/*
syntax = "proto3";

package mcp_service;

option go_package = "ai-agent/mcp/proto";

// AgentMessage defines the standardized structure for messages.
message AgentMessage {
  string id = 1;
  string sender = 2;
  string recipient = 3;
  string type = 4;
  bytes payload = 5; // Use bytes for generic JSON payload
  string timestamp = 6; // RFC3339Nano string
  string channel_hint = 7;
  string correlation_id = 8;
  string priority = 9;
}

// SendMessageResponse is the response from a SendMessage call.
message SendMessageResponse {
  bool success = 1;
  string message = 2;
}

// MCPService defines the gRPC service for Multi-Channel Protocol.
service MCPService {
  rpc SendMessage (AgentMessage) returns (SendMessageResponse);
}
*/

// To generate the Go files from the .proto, you would run:
// `protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative mcp/proto/mcp.proto`
// from your project root. This would create `mcp.pb.go` and `mcp_grpc.pb.go` in `mcp/proto`.


// ====================================================================================================
// utils/logger.go
// ====================================================================================================

package utils

import (
	"log/slog"
	"os"
	"time"
)

// InitLogger initializes the global structured logger.
func InitLogger() {
	// Customize the logger to include time, level, and source by default
	handlerOpts := &slog.HandlerOptions{
		AddSource: true,  // Include file and line number
		Level:     slog.LevelDebug, // Set global log level
	}

	// Use JSON handler for structured logs
	logger := slog.New(slog.NewJSONHandler(os.Stdout, handlerOpts))
	slog.SetDefault(logger)
	slog.Info("Logger initialized successfully.", "level", handlerOpts.Level)
}

// ====================================================================================================
// utils/helpers.go
// ====================================================================================================

package utils

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

// GenerateUUID creates a simple UUID-like string.
func GenerateUUID() string {
	b := make([]byte, 16)
	_, err := rand.Read(b)
	if err != nil {
		// Fallback for non-cryptographic UUID if rand fails (should not happen in most cases)
		return fmt.Sprintf("fallback-%d", time.Now().UnixNano())
	}
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

// GenerateRandomFloat generates a random float64 between min and max.
func GenerateRandomFloat(min, max float64) float64 {
	val, _ := rand.Int(rand.Reader, big.NewInt(1000000))
	return min + float64(val.Int64())/1000000*(max-min)
}

// GenerateRandomInt generates a random int between min and max (inclusive).
func GenerateRandomInt(min, max int) int {
	val, _ := rand.Int(rand.Reader, big.NewInt(int64(max-min+1)))
	return min + int(val.Int64())
}

// ContainsKeyword checks if a string contains a keyword (case-insensitive).
func ContainsKeyword(s, keyword string) bool {
	return strings.Contains(strings.ToLower(s), strings.ToLower(keyword))
}

// SetupGracefulShutdown listens for OS signals (SIGINT, SIGTERM) and returns a channel
// that will be closed when a signal is received, allowing for graceful shutdown.
func SetupGracefulShutdown() <-chan os.Signal {
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	return stop
}

// ParseDurationOrDefault parses a string into time.Duration or returns a default.
func ParseDurationOrDefault(s string, defaultDuration time.Duration) time.Duration {
	d, err := time.ParseDuration(s)
	if err != nil {
		return defaultDuration
	}
	return d
}

```