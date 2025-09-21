```go
/*
AetherCore: The Adaptive & Self-Evolving AI Agent with SynapseNexus (MCP) Interface

AetherCore is an advanced, self-orchestrating AI agent designed for dynamic environments. Its core is the 'SynapseNexus', a Master Control Program (MCP) component responsible for orchestrating, monitoring, and adapting the behavior of numerous specialized sub-agents. AetherCore aims to achieve high levels of autonomy, resilience, and cognitive sophistication by going beyond reactive processing, incorporating meta-learning, predictive analytics, much like an intelligent operating system for AI modules.

Key Design Principles:
-   **SynapseNexus-Centric Orchestration:** All sub-agent interactions, task allocations, and resource management are mediated and optimized by the central SynapseNexus.
-   **Dynamic Adaptability:** Continuously learns and adapts its operational strategies, internal knowledge, and sub-agent configurations based on performance, environment, and goals.
-   **Proactive Intelligence:** Anticipates needs, predicts potential failures or opportunities, and prepares resources or strategies before explicit requests.
-   **Ethical & Aligned:** Incorporates built-in mechanisms for bias mitigation, ethical guardrails, and adversarial resilience to ensure safe and responsible operation.
-   **Emergent Capabilities:** Designed to facilitate and leverage novel capabilities arising from the complex and dynamic interactions of its constituent sub-agents.
-   **No Duplication of Open Source:** Focuses on unique architectural patterns, named functions, and the *meta-level* control offered by SynapseNexus, rather than specific existing ML library integrations.

--------------------------------------------------------------------------------
SYNAPSENEXUS (MCP) FUNCTIONS SUMMARY:
--------------------------------------------------------------------------------

The SynapseNexus is the core of AetherCore, acting as its Master Control Program. It manages and orchestrates all sub-agents, internal state, and external interactions.

1.  RegisterSubAgent(agentID string, agent Agent):
    Purpose: Dynamically registers a new AI sub-agent module with SynapseNexus, making its capabilities available for task dispatch and orchestration.
    Concept: Enables a modular and extensible architecture, allowing for seamless integration, hot-swapping, or removal of specialized AI components without restarting the core.

2.  DispatchTask(ctx context.Context, task TaskSpec): (Conceptual for actual execution)
    Purpose: Assigns a high-level task to the most appropriate registered sub-agent(s) based on their declared capabilities, current operational state, and SynapseNexus's learned routing policies.
    Concept: Intelligent task routing and load balancing across diverse AI modules, optimizing for performance, resource utilization, and reliability.

3.  MonitorAgentHealth():
    Purpose: Continuously assesses the operational status, resource consumption, and responsiveness of all registered sub-agents, reporting anomalies to the SynapseNexus.
    Concept: System health monitoring, foundational for resilience, predictive failure anticipation, and resource arbitration, ensuring the overall stability of AetherCore.

4.  ResourceArbiter(ctx context.Context, request ResourceRequest):
    Purpose: Arbitrates and allocates computational resources (e.g., CPU, memory, GPU, API quotas, network bandwidth) among competing sub-agents based on task priority, system load, and predefined policies.
    Concept: Intelligent, dynamic resource management, preventing bottlenecks, ensuring critical task execution, and optimizing global system efficiency.

5.  CognitiveStateReconciliation(): (Internal, runs periodically)
    Purpose: Actively merges and resolves potential conflicts or discrepancies in the cognitive state (e.g., beliefs, environmental models, knowledge representations) reported by different sub-agents.
    Concept: Maintains a coherent, unified, and non-contradictory global understanding of the agent's environment and internal state across multiple perspectives.

6.  BehavioralDirectiveInjection(ctx context.Context, directive BehavioralDirective):
    Purpose: Proactively injects high-level behavioral directives, ethical constraints, or strategic mandates into sub-agent operations to guide their actions and outputs, ensuring alignment with AetherCore's overarching goals.
    Concept: Top-down control mechanism for alignment, safety, and policy enforcement, allowing for dynamic adjustment of agent behavior.

7.  MetaLearningPolicySynthesizer(): (Internal, adaptive)
    Purpose: Analyzes past performance data of tasks and sub-agent learning processes to learn and synthesize optimal learning strategies, hyperparameter configurations, or data augmentation policies for individual or groups of sub-agents.
    Concept: Self-improvement on *how to learn*, allowing AetherCore to adapt its internal training and knowledge acquisition methodologies for better efficacy over time.

8.  PredictiveFailureAnticipation(): (Internal, continuous)
    Purpose: Utilizes advanced machine learning models (e.g., time-series analysis, anomaly detection) to analyze operational logs, performance metrics, and environmental variables to predict potential sub-agent failures or degradation *before* they occur.
    Concept: Proactive maintenance, self-healing capabilities, and preventative task re-assignment, minimizing downtime and improving robustness.

9.  EmergentSkillDiscovery(): (Internal, continuous monitoring)
    Purpose: Observes and analyzes the interaction patterns and combined outputs of multiple sub-agents to identify and catalog novel, synergistic capabilities or problem-solving approaches not explicitly programmed or designed.
    Concept: Fosters the development of unpredicted, complex behaviors and innovative solutions arising from simpler, interacting components.

10. SelfRewiringKnowledgeGraph(): (Internal, adaptive)
    Purpose: Dynamically updates, optimizes, and prunes its internal knowledge representation graph based on new experiences, query patterns, identified semantic relationships, and forgetting mechanisms.
    Concept: Adaptive long-term memory, continuously refining its understanding of concepts and their interconnections, ensuring relevance and efficiency.

11. AnticipatoryContextPreload(ctx context.Context, currentTaskID string):
    Purpose: Based on current tasks, historical patterns, user profiles, and environmental cues, proactively loads relevant knowledge, data, and prepares necessary sub-agents for probable future requests or scenarios.
    Concept: Reduces latency and improves responsiveness by pre-fetching information and pre-processing tasks, leading to a more fluid and efficient user experience.

12. TemporalCoherenceEngine(): (Internal, real-time)
    Purpose: Ensures a consistent and logically ordered understanding of events and information over time, resolving temporal ambiguities, predicting sequences, and maintaining narrative integrity across disparate data sources.
    Concept: Advanced temporal reasoning, crucial for understanding evolving situations, complex event processing, and maintaining an accurate historical context.

13. SemanticIntentForecasting(ctx context.Context, currentInteraction string):
    Purpose: Predicts future user intents or system needs by analyzing interaction history, external trends, real-time contextual data, and probabilistic models of human/system behavior.
    Concept: Proactive assistance, goal-oriented planning, and preemptive action, allowing AetherCore to be one step ahead of explicit commands.

14. MultiModalCrossReferencing(ctx context.Context, data []interface{}):
    Purpose: Correlates and integrates information across diverse modalities (e.g., text, image, audio, video, sensor data) to form a richer, more robust, and comprehensive understanding of concepts or events.
    Concept: Holistic perception, breaking down modality silos to create a unified cognitive model of the environment.

15. EthicalGuardrailEnforcement(ctx context.Context, proposedAction string):
    Purpose: Actively filters, modifies, or blocks sub-agent outputs and actions in real-time to ensure strict compliance with predefined ethical principles, safety guidelines, and legal frameworks.
    Concept: Core safety and alignment mechanism, preventing harmful, biased, or unethical behavior, operating as a final cognitive check.

16. AdversarialResilienceFortifier(): (Internal, continuous monitoring)
    Purpose: Continuously monitors inputs, internal states, and model parameters for signs of adversarial attacks (e.g., data poisoning, model evasion, prompt injection) and deploys countermeasures to maintain integrity and reliability.
    Concept: Robustness against malicious attacks, ensuring trust and reliability of AetherCore's intelligence.

17. BiasDetectionAndMitigation(): (Internal, adaptive)
    Purpose: Scans input data, model training processes, and sub-agent outputs for systemic biases, applying adaptive techniques (e.g., re-sampling, re-weighting, debiasing algorithms) to reduce or counteract their impact.
    Concept: Promotes fairness and equity in AI decision-making, ensuring AetherCore operates responsibly and justly.

18. AbstractPatternGenerator(ctx context.Context, inputData interface{}):
    Purpose: Identifies underlying abstract patterns, structures, or relationships across disparate data sets or problem domains and generates novel representations or creative solutions based on these insights.
    Concept: Creative problem-solving and innovation, going beyond direct analogy to discover fundamental principles and generate new ideas.

19. SyntheticPersonaProjection(ctx context.Context, audience Profile):
    Purpose: Dynamically generates and adopts different communication styles, tones, vocabulary, or 'personas' to optimize interaction effectiveness with diverse users, external systems, or cultural contexts.
    Concept: Adaptive social intelligence and contextual communication, enhancing engagement and understanding.

20. RecursiveSelfReflectionQuery(): (Internal, triggered by events/periodically)
    Purpose: Initiates internal cognitive queries about its own decision-making processes, knowledge gaps, potential biases, or reasoning flaws, prompting self-correction, further investigation, or requesting external validation.
    Concept: Meta-cognition, allowing the agent to reason about its own reasoning, a crucial step towards true autonomy and continuous improvement.

21. DynamicOntologyEvolution(): (Internal, adaptive)
    Purpose: Allows its internal semantic representation of concepts, relationships, and taxonomies (ontology) to evolve, update, and adapt dynamically based on new information, learning experiences, and interactions.
    Concept: Flexible and continuously updated knowledge representation, avoiding brittle, static models and reflecting the changing nature of its environment.

22. EnvironmentalAnomalyDetection(): (Internal, real-time)
    Purpose: Monitors its operational environment (e.g., system load, external API response times, sensor data, network traffic, user behavior patterns) for unusual patterns indicating potential issues, threats, or emerging opportunities.
    Concept: Proactive environmental awareness, enabling rapid adaptive response to unforeseen circumstances.

23. DistributedConsensusIntegrator(ctx context.Context, proposals []DecisionProposal):
    Purpose: When interacting with other AetherCore instances, federated learning setups, or human decision-makers, integrates and resolves discrepancies in distributed knowledge, decisions, or goal states to achieve a coherent consensus.
    Concept: Enables collaborative intelligence and coherent decision-making in complex, multi-agent, or human-AI hybrid systems.
*/

package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"sync"
	"time"

	"aethercore/pkg/agents"
	"aethercore/pkg/config"
	"aethercore/pkg/events"
	"aethercore/pkg/storage"
	"aethercore/pkg/synapse"
	"aethercore/pkg/util"
)

func main() {
	cfg := config.LoadConfig()
	logger := util.NewLogger(cfg.LogLevel)

	kvStore := storage.NewInMemoryKVStore()
	eventBus := events.NewEventBus()

	// Initialize SynapseNexus (MCP)
	synapseNexus := synapse.NewSynapseNexus(cfg, logger, kvStore, eventBus)

	// Register some example sub-agents
	analyticAgent := agents.NewAnalyticAgent("analytic-v1", logger, eventBus)
	creativeAgent := agents.NewCreativeAgent("creative-v1", logger, eventBus)
	decisionAgent := agents.NewDecisionAgent("decision-v1", logger, eventBus)

	synapseNexus.RegisterSubAgent(analyticAgent.ID(), analyticAgent)
	synapseNexus.RegisterSubAgent(creativeAgent.ID(), creativeAgent)
	synapseNexus.RegisterSubAgent(decisionAgent.ID(), decisionAgent)

	// Start SynapseNexus and all registered sub-agents
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel() // Ensure cancel is called on exit

	synapseNexus.Start(ctx)
	analyticAgent.Start(ctx)
	creativeAgent.Start(ctx)
	decisionAgent.Start(ctx)

	logger.Info("AetherCore and sub-agents started. SynapseNexus is active.")

	// --- Demonstrate SynapseNexus Functions ---

	// 1. RegisterSubAgent (already done above)

	// 2. DispatchTask
	go func() {
		time.Sleep(2 * time.Second) // Give agents time to start
		logger.Info("Attempting to dispatch a task...")
		task := synapse.TaskSpec{
			ID:          "initial-analysis-001",
			Description: "Analyze current market trends for Q3 earnings.",
			Capabilities: []string{
				"data-analysis",
				"trend-prediction",
			},
			Priority: synapse.PriorityHigh,
		}
		result, err := synapseNexus.DispatchTask(ctx, task)
		if err != nil {
			logger.Errorf("Failed to dispatch task %s: %v", task.ID, err)
		} else {
			logger.Infof("Task %s dispatched, result: %v", task.ID, result)
		}

		task2 := synapse.TaskSpec{
			ID:          "creative-idea-002",
			Description: "Generate 5 new marketing slogans for a futuristic product.",
			Capabilities: []string{
				"creative-generation",
				"text-synthesis",
			},
			Priority: synapse.PriorityMedium,
		}
		result2, err2 := synapseNexus.DispatchTask(ctx, task2)
		if err2 != nil {
			logger.Errorf("Failed to dispatch task %s: %v", task2.ID, err2)
		} else {
			logger.Infof("Task %s dispatched, result: %v", task2.ID, result2)
		}
	}()

	// 3. MonitorAgentHealth (SynapseNexus runs this internally)
	go func() {
		time.Sleep(5 * time.Second)
		logger.Info("SynapseNexus initiating internal health check...")
		synapseNexus.MonitorAgentHealth() // Trigger an explicit check
	}()

	// 4. ResourceArbiter
	go func() {
		time.Sleep(7 * time.Second)
		logger.Info("Agent 'analytic-v1' requesting resources...")
		req := synapse.ResourceRequest{
			AgentID:    "analytic-v1",
			ResourceType: synapse.ResourceCPU,
			Amount:     2,
			Priority:   synapse.PriorityHigh,
		}
		if synapseNexus.ResourceArbiter(ctx, req) {
			logger.Infof("Resource CPU (2 units) granted to 'analytic-v1'")
		} else {
			logger.Warnf("Resource CPU (2 units) denied to 'analytic-v1'")
		}
	}()

	// 5. CognitiveStateReconciliation
	go func() {
		time.Sleep(10 * time.Second)
		logger.Info("SynapseNexus performing CognitiveStateReconciliation...")
		// In a real scenario, agents would update a shared state, then MCP reconciles
		kvStore.Set("analytic-state", "market_up_strong")
		kvStore.Set("creative-state", "market_uncertain") // Conflicting state
		synapseNexus.CognitiveStateReconciliation()
	}()

	// 6. BehavioralDirectiveInjection
	go func() {
		time.Sleep(12 * time.Second)
		logger.Info("SynapseNexus injecting a BehavioralDirective...")
		directive := synapse.BehavioralDirective{
			ID:          "ethical-filter-001",
			Description: "Ensure all creative outputs are ethically neutral.",
			ScopeAgent:  "creative-v1",
			Action:      "filter_unethical_content",
		}
		synapseNexus.BehavioralDirectiveInjection(ctx, directive)
	}()

	// 7. MetaLearningPolicySynthesizer
	go func() {
		time.Sleep(15 * time.Second)
		logger.Info("SynapseNexus initiating MetaLearningPolicySynthesis...")
		// Simulate some past performance data
		kvStore.Set("agent:analytic-v1:performance", "accuracy:0.85,latency:100ms")
		kvStore.Set("agent:creative-v1:performance", "novelty:0.7,relevance:0.65")
		synapseNexus.MetaLearningPolicySynthesizer()
	}()

	// 8. PredictiveFailureAnticipation
	go func() {
		time.Sleep(18 * time.Second)
		logger.Info("SynapseNexus performing PredictiveFailureAnticipation...")
		// Simulate a warning metric
		kvStore.Set("agent:analytic-v1:metric:error_rate", "0.05") // High error rate
		synapseNexus.PredictiveFailureAnticipation()
	}()

	// 9. EmergentSkillDiscovery
	go func() {
		time.Sleep(21 * time.Second)
		logger.Info("SynapseNexus scanning for EmergentSkillDiscovery...")
		// This would involve analyzing complex interaction logs
		synapseNexus.EmergentSkillDiscovery()
	}()

	// 10. SelfRewiringKnowledgeGraph
	go func() {
		time.Sleep(24 * time.Second)
		logger.Info("SynapseNexus triggering SelfRewiringKnowledgeGraph...")
		synapseNexus.SelfRewiringKnowledgeGraph()
	}()

	// 11. AnticipatoryContextPreload
	go func() {
		time.Sleep(27 * time.Second)
		logger.Info("SynapseNexus performing AnticipatoryContextPreload...")
		synapseNexus.AnticipatoryContextPreload(ctx, "future-task-003")
	}()

	// 12. TemporalCoherenceEngine
	go func() {
		time.Sleep(30 * time.Second)
		logger.Info("SynapseNexus engaging TemporalCoherenceEngine...")
		// Simulate events
		eventBus.Publish(events.Event{Type: "data_ingested", Data: "event_t1"})
		time.Sleep(100 * time.Millisecond)
		eventBus.Publish(events.Event{Type: "analysis_completed", Data: "event_t2"})
		synapseNexus.TemporalCoherenceEngine()
	}()

	// 13. SemanticIntentForecasting
	go func() {
		time.Sleep(33 * time.Second)
		logger.Info("SynapseNexus attempting SemanticIntentForecasting...")
		synapseNexus.SemanticIntentForecasting(ctx, "user query: 'what's next for AI?'")
	}()

	// 14. MultiModalCrossReferencing
	go func() {
		time.Sleep(36 * time.Second)
		logger.Info("SynapseNexus performing MultiModalCrossReferencing...")
		// Imagine data from different sources
		multimodalData := []interface{}{"text: AI is evolving.", "image_caption: a futuristic robot."}
		synapseNexus.MultiModalCrossReferencing(ctx, multimodalData)
	}()

	// 15. EthicalGuardrailEnforcement
	go func() {
		time.Sleep(39 * time.Second)
		logger.Info("SynapseNexus checking EthicalGuardrailEnforcement...")
		synapseNexus.EthicalGuardrailEnforcement(ctx, "proposed_action: generate misleading news")
		synapseNexus.EthicalGuardrailEnforcement(ctx, "proposed_action: summarize recent tech news")
	}()

	// 16. AdversarialResilienceFortifier
	go func() {
		time.Sleep(42 * time.Second)
		logger.Info("SynapseNexus activating AdversarialResilienceFortifier...")
		synapseNexus.AdversarialResilienceFortifier()
	}()

	// 17. BiasDetectionAndMitigation
	go func() {
		time.Sleep(45 * time.Second)
		logger.Info("SynapseNexus initiating BiasDetectionAndMitigation...")
		synapseNexus.BiasDetectionAndMitigation()
	}()

	// 18. AbstractPatternGenerator
	go func() {
		time.Sleep(48 * time.Second)
		logger.Info("SynapseNexus engaging AbstractPatternGenerator...")
		synapseNexus.AbstractPatternGenerator(ctx, "finance_data_A, biology_data_B")
	}()

	// 19. SyntheticPersonaProjection
	go func() {
		time.Sleep(51 * time.Second)
		logger.Info("SynapseNexus projecting SyntheticPersonaProjection...")
		synapseNexus.SyntheticPersonaProjection(ctx, synapse.Profile{Name: "Executive", Traits: []string{"formal", "concise"}})
	}()

	// 20. RecursiveSelfReflectionQuery
	go func() {
		time.Sleep(54 * time.Second)
		logger.Info("SynapseNexus initiating RecursiveSelfReflectionQuery...")
		synapseNexus.RecursiveSelfReflectionQuery()
	}()

	// 21. DynamicOntologyEvolution
	go func() {
		time.Sleep(57 * time.Second)
		logger.Info("SynapseNexus triggering DynamicOntologyEvolution...")
		synapseNexus.DynamicOntologyEvolution()
	}()

	// 22. EnvironmentalAnomalyDetection
	go func() {
		time.Sleep(60 * time.Second)
		logger.Info("SynapseNexus performing EnvironmentalAnomalyDetection...")
		// Simulate an external anomaly event
		eventBus.Publish(events.Event{Type: "external_api_latency_spike", Data: 5000})
		synapseNexus.EnvironmentalAnomalyDetection()
	}()

	// 23. DistributedConsensusIntegrator
	go func() {
		time.Sleep(63 * time.Second)
		logger.Info("SynapseNexus engaging DistributedConsensusIntegrator...")
		proposals := []synapse.DecisionProposal{
			{AgentID: "analytic-v1", Decision: "buy_stock_A", Confidence: 0.8},
			{AgentID: "decision-v1", Decision: "hold_stock_A", Confidence: 0.7},
		}
		synapseNexus.DistributedConsensusIntegrator(ctx, proposals)
	}()

	// Keep the main goroutine alive to allow sub-agents and SynapseNexus to run
	// In a real application, you'd have graceful shutdown handlers (e.g., OS signals)
	select {
	case <-time.After(70 * time.Second): // Run for a fixed duration to show examples
		logger.Info("AetherCore demonstration complete. Shutting down...")
		cancel()
	}

	// Give time for goroutines to clean up
	time.Sleep(2 * time.Second)
	logger.Info("AetherCore gracefully shut down.")
}

```
```go
// pkg/config/config.go
package config

import (
	"log"
	"os"
	"strconv"
)

type Config struct {
	LogLevel       string
	MaxAgents      int
	ResourceLimits map[string]int // e.g., "CPU": 8, "Memory": 16 (GB)
}

func LoadConfig() *Config {
	maxAgents, err := strconv.Atoi(getEnv("AETHERCORE_MAX_AGENTS", "10"))
	if err != nil {
		log.Printf("Warning: Invalid AETHERCORE_MAX_AGENTS, using default 10. Error: %v", err)
		maxAgents = 10
	}

	return &Config{
		LogLevel:  getEnv("AETHERCORE_LOG_LEVEL", "info"),
		MaxAgents: maxAgents,
		ResourceLimits: map[string]int{
			"CPU":    8,  // Cores
			"Memory": 16, // GB
			"GPU":    1,  // Units
			"API":    1000, // API calls per minute
		},
	}
}

func getEnv(key, defaultValue string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return defaultValue
}

```
```go
// pkg/events/bus.go
package events

import (
	"fmt"
	"sync"
	"time"
)

// Event represents a message payload for the event bus.
type Event struct {
	Type      string
	Timestamp time.Time
	Data      interface{}
	Source    string
}

// EventHandler defines the signature for functions that process events.
type EventHandler func(Event)

// EventBus provides a simple publish-subscribe mechanism.
type EventBus struct {
	subscribers map[string][]EventHandler
	mu          sync.RWMutex
}

// NewEventBus creates and returns a new EventBus instance.
func NewEventBus() *EventBus {
	return &EventBus{
		subscribers: make(map[string][]EventHandler),
	}
}

// Subscribe registers an EventHandler for a specific event type.
func (eb *EventBus) Subscribe(eventType string, handler EventHandler) {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	eb.subscribers[eventType] = append(eb.subscribers[eventType], handler)
}

// Publish sends an event to all registered handlers for its type.
func (eb *EventBus) Publish(event Event) {
	eb.mu.RLock()
	defer eb.mu.RUnlock()

	event.Timestamp = time.Now() // Stamp event at publish time

	if handlers, found := eb.subscribers[event.Type]; found {
		for _, handler := range handlers {
			// Run handlers in goroutines to avoid blocking the publisher
			go func(h EventHandler, e Event) {
				// Add basic error recovery for handlers
				defer func() {
					if r := recover(); r != nil {
						fmt.Printf("Event handler for type '%s' panicked: %v\n", e.Type, r)
					}
				}()
				h(e)
			}(handler, event)
		}
	}
}

```
```go
// pkg/storage/kvstore.go
package storage

import (
	"fmt"
	"sync"
)

// KVStore defines the interface for a simple Key-Value store.
type KVStore interface {
	Set(key string, value string)
	Get(key string) (string, bool)
	Delete(key string)
	GetAll() map[string]string
}

// InMemoryKVStore is a basic, in-memory implementation of KVStore.
type InMemoryKVStore struct {
	data map[string]string
	mu   sync.RWMutex
}

// NewInMemoryKVStore creates and returns a new InMemoryKVStore.
func NewInMemoryKVStore() *InMemoryKVStore {
	return &InMemoryKVStore{
		data: make(map[string]string),
	}
}

// Set stores a key-value pair.
func (s *InMemoryKVStore) Set(key string, value string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data[key] = value
	fmt.Printf("[KVStore] Set: %s = %s\n", key, value) // Added for visibility
}

// Get retrieves a value by key. Returns the value and a boolean indicating if the key was found.
func (s *InMemoryKVStore) Get(key string) (string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	value, found := s.data[key]
	return value, found
}

// Delete removes a key-value pair.
func (s *InMemoryKVStore) Delete(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.data, key)
	fmt.Printf("[KVStore] Deleted: %s\n", key) // Added for visibility
}

// GetAll returns a copy of all data in the store.
func (s *InMemoryKVStore) GetAll() map[string]string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	copyData := make(map[string]string, len(s.data))
	for k, v := range s.data {
		copyData[k] = v
	}
	return copyData
}

```
```go
// pkg/util/logger.go
package util

import (
	"log"
	"os"
	"strings"
)

// Logger provides a simple logging interface.
type Logger interface {
	Debug(format string, v ...interface{})
	Info(format string, v ...interface{})
	Warn(format string, v ...interface{})
	Error(format string, v ...interface{})
	Errorf(format string, v ...interface{}) // For consistency with Errorf style
}

type logLevel int

const (
	levelDebug logLevel = iota
	levelInfo
	levelWarn
	levelError
)

type stdLogger struct {
	minLevel logLevel
	prefix   string
	stdlog   *log.Logger
}

// NewLogger creates a new Logger instance based on the provided log level string.
func NewLogger(level string) Logger {
	sl := &stdLogger{
		prefix:   "[AetherCore] ",
		stdlog:   log.New(os.Stdout, "", log.Ldate|log.Ltime|log.Lshortfile),
	}

	switch strings.ToLower(level) {
	case "debug":
		sl.minLevel = levelDebug
	case "info":
		sl.minLevel = levelInfo
	case "warn":
		sl.minLevel = levelWarn
	case "error":
		sl.minLevel = levelError
	default:
		sl.minLevel = levelInfo // Default to info
	}
	return sl
}

func (l *stdLogger) log(level logLevel, format string, v ...interface{}) {
	if level >= l.minLevel {
		msg := fmt.Sprintf(format, v...)
		switch level {
		case levelDebug:
			l.stdlog.Printf("%sDEBUG: %s", l.prefix, msg)
		case levelInfo:
			l.stdlog.Printf("%sINFO: %s", l.prefix, msg)
		case levelWarn:
			l.stdlog.Printf("%sWARN: %s", l.prefix, msg)
		case levelError:
			l.stdlog.Printf("%sERROR: %s", l.prefix, msg)
		}
	}
}

func (l *stdLogger) Debug(format string, v ...interface{}) {
	l.log(levelDebug, format, v...)
}

func (l *stdLogger) Info(format string, v ...interface{}) {
	l.log(levelInfo, format, v...)
}

func (l *stdLogger) Warn(format string, v ...interface{}) {
	l.log(levelWarn, format, v...)
}

func (l *stdLogger) Error(format string, v ...interface{}) {
	l.log(levelError, format, v...)
}

func (l *stdLogger) Errorf(format string, v ...interface{}) {
	l.log(levelError, format, v...)
}

```
```go
// pkg/synapse/types.go
package synapse

import (
	"context"
	"time"
)

// Priority represents the urgency or importance of a task or resource request.
type Priority int

const (
	PriorityLow Priority = iota
	PriorityMedium
	PriorityHigh
	PriorityCritical
)

// AgentConfig holds configuration for a sub-agent.
type AgentConfig struct {
	ID         string
	Name       string
	Type       string // e.g., "analytic", "creative", "decision"
	Capabilities []string
	ResourceProfile map[string]int // e.g., "CPU": 2, "Memory": 4
}

// TaskSpec defines a task to be processed by an agent.
type TaskSpec struct {
	ID          string
	Description string
	Capabilities []string // Capabilities required to handle this task
	InputData   interface{}
	Priority    Priority
	CreatedAt   time.Time
}

// ResourceType defines different types of computational resources.
type ResourceType string

const (
	ResourceCPU    ResourceType = "CPU"
	ResourceMemory ResourceType = "Memory"
	ResourceGPU    ResourceType = "GPU"
	ResourceAPI    ResourceType = "API" // External API tokens/calls
)

// ResourceRequest represents a request for resources from an agent.
type ResourceRequest struct {
	AgentID      string
	ResourceType ResourceType
	Amount       int
	Priority     Priority
}

// BehavioralDirective represents a high-level instruction or ethical guardrail.
type BehavioralDirective struct {
	ID          string
	Description string
	ScopeAgent  string // "" for all agents, or specific agent ID
	Action      string // e.g., "filter_unethical_content", "prioritize_safety"
	Constraint  string // Additional parameters for the action
	Severity    Priority
}

// Profile represents a user or system persona for interaction.
type Profile struct {
	Name  string
	Traits []string // e.g., "formal", "concise", "empathetic"
}

// DecisionProposal represents a proposed decision from an agent in a distributed context.
type DecisionProposal struct {
	AgentID    string
	Decision   string
	Confidence float64
	Timestamp  time.Time
}

// Agent is the interface that all sub-agents must implement to be managed by SynapseNexus.
type Agent interface {
	ID() string
	Capabilities() []string
	Start(ctx context.Context)
	Stop()
	ProcessTask(ctx context.2015.11.10.15.55.51_a_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s_1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2_s-1.2