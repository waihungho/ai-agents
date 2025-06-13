Okay, here is the Go implementation of an AI Agent system using a Message Passing Interface (MCP).

This design focuses on conceptual AI tasks and agent coordination via structured messages. The functions are chosen to be diverse, representing different facets of agent capability – data processing, state management, coordination, rudimentary learning/adaptation, and control.

---

```go
// AI Agent System with MCP Interface

/*
Outline:

1.  Introduction
    - Defines the concept of an AI Agent interacting via Message Passing.
    - Describes the core components: Message, MessageBus, Agent.
2.  Core MCP Components
    - Message Types (iota)
    - Message Struct
    - MessageBus Struct and Methods (Register, Deregister, SendMessage, Run)
3.  Agent Structure and Core Logic
    - Agent Struct (ID, channels, state, parameters, knowledge, bus, context)
    - NewAgent Constructor
    - Run Method (Main processing loop, message handling switch)
    - Stop Method
4.  Agent Functions (Implemented as Message Handlers)
    - Grouped by category for clarity.
    - Each corresponds to a MessageType and a handle method.
5.  Example Usage (in main function)
    - Setting up the MessageBus.
    - Creating and registering agents.
    - Sending initial messages.
    - Running the system.
    - Graceful shutdown.

Function Summary (Message Types and Handlers):

Core Agent Management:
-   MessageTypeRequestState: Agent requests state of another agent. Handler: handleRequestState (Responds with MessageTypeResponseState)
-   MessageTypeResponseState: Agent sends its state. Handler: handleResponseState (Receiving agent updates knowledge or logs).
-   MessageTypeUpdateState: Agent requests another agent to update its internal state. Handler: handleUpdateState (Modifies agent.State).
-   MessageTypeSpawnAgent: Agent requests the MessageBus (or a managing agent) to spawn a new agent instance. Handler: handleSpawnAgent (Abstracted, bus/manager handles creation/registration).
-   MessageTypeTerminateAgent: Agent requests another agent to shut down or requests bus to terminate an agent. Handler: handleTerminateAgent (Triggers agent.Stop).
-   MessageTypeDiscoverAgents: Agent requests a list of active agents from the bus or a registry. Handler: handleDiscoverAgents (Bus/registry responds with agent IDs).

Information Processing & Analysis:
-   MessageTypeProcessDataChunk: Agent is sent raw data for processing (e.g., parsing, validation). Handler: handleProcessDataChunk (Processes data, potentially updates state or sends results).
-   MessageTypeSynthesizeReport: Agent is requested to compile information from its state/knowledge base into a report. Handler: handleSynthesizeReport (Generates summary/report, sends MessageTypeReportSynthesized).
-   MessageTypePerformCalculation: Agent is requested to perform a specific calculation or simulation step. Handler: handlePerformCalculation (Performs computation, sends MessageTypeCalculationResult).
-   MessageTypeCategorizeData: Agent is given data and asked to categorize it based on internal models/rules. Handler: handleCategorizeData (Assigns categories, sends MessageTypeDataCategorized).
-   MessageTypeSummarizeText: Agent receives text and is asked to provide a summary. Handler: handleSummarizeText (Performs text summarization, sends MessageTypeTextSummarized).
-   MessageTypeAnalyzeSentiment: Agent receives text and analyzes its sentiment (positive/negative/neutral). Handler: handleAnalyzeSentiment (Sends MessageTypeSentimentResult).

Coordination & Interaction:
-   MessageTypeRequestCollaboration: Agent asks another agent to collaborate on a task. Handler: handleRequestCollaboration (Agent evaluates request, responds with acceptance/rejection).
-   MessageTypeProposeActionPlan: Agent suggests a sequence of actions/tasks to another agent or a group. Handler: handleProposeActionPlan (Agent evaluates plan, potentially adds to its task queue or responds).
-   MessageTypeNegotiateParameter: Agent initiates negotiation with another agent to agree on a shared parameter value. Handler: handleNegotiateParameter (Engages in negotiation logic, sends proposals/counter-proposals).
-   MessageTypeBroadcastAlert: Agent sends a critical alert message to multiple or all agents. Handler: handleBroadcastAlert (Received by agents, triggers urgent processing/state change).
-   MessageTypeRequestResource: Agent requests an abstract resource (e.g., processing time, data access) from another agent or manager. Handler: handleRequestResource (Agent evaluates resource availability, responds with grant/denial).
-   MessageTypeOfferResource: Agent proactively offers a resource it possesses. Handler: handleOfferResource (Received by agents, they can request if needed).

Rudimentary Learning & Adaptation:
-   MessageTypeUpdateInternalModel: Agent is sent feedback or new data to update its internal parameters or rules. Handler: handleUpdateInternalModel (Modifies agent.Parameters or KnowledgeBase).
-   MessageTypeAdjustBehavior: Agent is instructed to change its operational behavior based on feedback or new goals. Handler: handleAdjustBehavior (Modifies agent.Parameters or triggers a state change affecting future actions).
-   MessageTypeLogEvent: Agent records an event for future analysis or learning. Handler: handleLogEvent (Stores event details in KnowledgeBase or logs).

Advanced/Abstract Concepts:
-   MessageTypePerformSemanticMatch: Agent receives a query and performs matching against its knowledge base or received data based on meaning/tags. Handler: handlePerformSemanticMatch (Finds relevant info, sends results).
-   MessageTypeSimulateScenario: Agent is asked to run a small internal simulation based on parameters. Handler: handleSimulateScenario (Runs simulation, sends simulation output).
-   MessageTypeVerifyInformation: Agent is asked to verify a piece of information against multiple internal/external (abstracted) sources. Handler: handleVerifyInformation (Checks consistency, sends verification result).
-   MessageTypePrioritizeTask: Agent is asked to re-evaluate and prioritize its current tasks based on new input or state. Handler: handlePrioritizeTask (Reorders internal task queue).
-   MessageTypeAdaptProtocol: Agent is notified or instructed to switch to a different communication pattern or protocol version for interaction with specific agents. Handler: handleAdaptProtocol (Updates internal communication settings for certain recipients).
*/

package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// --- Core MCP Components ---

// MessageType defines the type of message being sent.
type MessageType int

const (
	MessageTypeUnknown MessageType = iota
	// Core Agent Management
	MessageTypeRequestState
	MessageTypeResponseState
	MessageTypeUpdateState
	MessageTypeSpawnAgent      // Handled by MessageBus/Manager
	MessageTypeTerminateAgent  // Handled by MessageBus/Target Agent
	MessageTypeDiscoverAgents  // Handled by MessageBus
	// Information Processing & Analysis
	MessageTypeProcessDataChunk
	MessageTypeSynthesizeReport
	MessageTypeReportSynthesized // Response to SynthesizeReport
	MessageTypePerformCalculation
	MessageTypeCalculationResult // Response to PerformCalculation
	MessageTypeCategorizeData
	MessageTypeDataCategorized // Response to CategorizeData
	MessageTypeSummarizeText
	MessageTypeTextSummarized // Response to SummarizeText
	MessageTypeAnalyzeSentiment
	MessageTypeSentimentResult // Response to AnalyzeSentiment
	// Coordination & Interaction
	MessageTypeRequestCollaboration
	MessageTypeProposeActionPlan
	MessageTypeNegotiateParameter // Initial message
	MessageTypeNegotiationUpdate  // Ongoing negotiation
	MessageTypeNegotiationResult  // Final negotiation outcome
	MessageTypeBroadcastAlert
	MessageTypeRequestResource
	MessageTypeMessageTypeResourceGrant   // Response to RequestResource
	MessageTypeMessageTypeResourceDenial  // Response to RequestResource
	MessageTypeOfferResource
	// Rudimentary Learning & Adaptation
	MessageTypeUpdateInternalModel
	MessageTypeAdjustBehavior
	MessageTypeLogEvent
	// Advanced/Abstract Concepts
	MessageTypePerformSemanticMatch
	MessageTypeSemanticMatchResult // Response to SemanticMatch
	MessageTypeSimulateScenario
	MessageTypeSimulationOutput    // Response to SimulateScenario
	MessageTypeVerifyInformation
	MessageTypeVerificationResult  // Response to VerifyInformation
	MessageTypePrioritizeTask
	MessageTypeAdaptProtocol
)

// Message is the standard structure for communication between agents.
type Message struct {
	SenderID    string      `json:"sender_id"`
	RecipientID string      `json:"recipient_id"` // Can be a specific Agent ID or "all" for broadcast
	Type        MessageType `json:"type"`
	Payload     interface{} `json:"payload"` // Data specific to the message type
	Timestamp   time.Time   `json:"timestamp"`
}

// MessageBus handles message routing between agents.
type MessageBus struct {
	agents     map[string]chan Message // Maps AgentID to its inbound message channel
	agentsMu   sync.RWMutex            // Mutex for concurrent access to the agents map
	control    chan struct{}           // Channel to signal bus shutdown
	wg         sync.WaitGroup          // WaitGroup to wait for bus goroutines to finish
	spawnChan  chan string             // Channel for agents to request spawning
	terminateChan chan string			// Channel for agents to request termination
}

// NewMessageBus creates a new MessageBus instance.
func NewMessageBus() *MessageBus {
	return &MessageBus{
		agents:      make(map[string]chan Message),
		control:     make(chan struct{}),
		spawnChan:   make(chan string),
		terminateChan: make(chan string),
	}
}

// RegisterAgent registers an agent with the bus.
func (mb *MessageBus) RegisterAgent(id string, inboundChan chan Message) {
	mb.agentsMu.Lock()
	defer mb.agentsMu.Unlock()
	if _, exists := mb.agents[id]; exists {
		log.Printf("Warning: Agent %s already registered", id)
	}
	mb.agents[id] = inboundChan
	log.Printf("Agent %s registered with MessageBus", id)
}

// DeregisterAgent removes an agent from the bus.
func (mb *MessageBus) DeregisterAgent(id string) {
	mb.agentsMu.Lock()
	defer mb.agentsMu.Unlock()
	if ch, exists := mb.agents[id]; exists {
		delete(mb.agents, id)
		// Optionally close the channel, but careful not to close while agent is reading
		// close(ch) // Safer if the agent's Run loop breaks cleanly on Deregister or context done
		log.Printf("Agent %s deregistered from MessageBus", id)
	} else {
		log.Printf("Warning: Agent %s not found for deregistration", id)
	}
}

// SendMessage routes a message to the appropriate recipient(s).
func (mb *MessageBus) SendMessage(msg Message) error {
	msg.Timestamp = time.Now() // Set timestamp here
	mb.agentsMu.RLock()
	defer mb.agentsMu.RUnlock()

	if msg.RecipientID == "all" {
		log.Printf("Bus: Broadcasting message type %v from %s to all", msg.Type, msg.SenderID)
		// Broadcast to all registered agents
		for id, ch := range mb.agents {
			// Avoid sending to self in a naive "all" broadcast unless intended
			if id != msg.SenderID {
				select {
				case ch <- msg:
					// Sent
				case <-time.After(50 * time.Millisecond): // Non-blocking send attempt
					log.Printf("Warning: Failed to send broadcast message to agent %s (channel blocked)", id)
				}
			}
		}
		return nil
	}

	// Send to a specific recipient
	if ch, exists := mb.agents[msg.RecipientID]; exists {
		log.Printf("Bus: Routing message type %v from %s to %s", msg.Type, msg.SenderID, msg.RecipientID)
		select {
		case ch <- msg:
			// Sent
			return nil
		case <-time.After(50 * time.Millisecond): // Non-blocking send attempt
			log.Printf("Warning: Failed to send message to agent %s (channel blocked)", msg.RecipientID)
			return fmt.Errorf("failed to send message to agent %s: channel blocked", msg.RecipientID)
		}
	} else {
		log.Printf("Error: Recipient agent %s not found", msg.RecipientID)
		return fmt.Errorf("recipient agent %s not found", msg.RecipientID)
	}
}

// Run starts the MessageBus routing loop (or can be used for internal bus management like spawning).
// For this basic example, routing happens directly in SendMessage.
// This Run method can handle cross-agent requests like Spawn/Terminate if a manager agent isn't used.
func (mb *MessageBus) Run(ctx context.Context) {
	log.Println("MessageBus started")
	defer mb.wg.Done()

	for {
		select {
		case agentID := <-mb.spawnChan:
			// In a real system, a manager agent or the bus would create and register here
			log.Printf("Bus received spawn request for new agent ID: %s (Not implemented fully here)", agentID)
			// Example: Add logic to create a new Agent, start its goroutine, and Register it.
			// newAgent := NewAgent(agentID, mb)
			// go newAgent.Run(ctx)
			// mb.RegisterAgent(agentID, newAgent.InboundMessages)
		case agentID := <-mb.terminateChan:
			log.Printf("Bus received termination request for agent ID: %s", agentID)
			// Find the agent's channel and send a terminate message or signal its context
			mb.agentsMu.RLock()
			ch, exists := mb.agents[agentID]
			mb.agentsMu.RUnlock()

			if exists {
				// Sending a specific terminate message allows the agent to clean up
				terminateMsg := Message{
					SenderID:    "MessageBus",
					RecipientID: agentID,
					Type:        MessageTypeTerminateAgent,
					Payload:     nil, // Or a reason/code
				}
				select {
				case ch <- terminateMsg:
					log.Printf("Bus sent termination signal to agent %s", agentID)
				case <-time.After(100 * time.Millisecond):
					log.Printf("Warning: Failed to send termination signal to agent %s (channel blocked)", agentID)
				}
			} else {
				log.Printf("Warning: Termination requested for non-existent agent %s", agentID)
			}
		case <-ctx.Done():
			log.Println("MessageBus shutting down")
			return
		}
	}
}

// Stop signals the MessageBus to shut down.
func (mb *MessageBus) Stop() {
	log.Println("Stopping MessageBus...")
	close(mb.control)
	mb.wg.Wait() // Wait for the Run goroutine to finish
	log.Println("MessageBus stopped")
}

// --- Agent Structure and Core Logic ---

// Agent represents an individual autonomous entity.
type Agent struct {
	ID string

	InboundMessages chan Message
	OutboundChannel chan Message // Agents send *to* this channel, which is read by the MessageBus

	State         map[string]interface{} // Internal state data
	Parameters    map[string]interface{} // Configuration/behavior parameters
	KnowledgeBase map[string]interface{} // Learned information, models, historical data

	MessageBus *MessageBus // Reference to the bus for sending messages

	EventSubscriptions map[MessageType][]string // Map MessageType to list of AgentIDs to notify

	Ctx    context.Context
	Cancel context.CancelFunc

	wg sync.WaitGroup // WaitGroup for agent's internal goroutines
}

// NewAgent creates a new Agent instance.
func NewAgent(id string, bus *MessageBus) *Agent {
	ctx, cancel := context.WithCancel(context.Background())
	agent := &Agent{
		ID:               id,
		InboundMessages:  make(chan Message, 10), // Buffered channel
		OutboundChannel:  make(chan Message, 10), // Buffered channel for sending
		State:            make(map[string]interface{}),
		Parameters:       make(map[string]interface{}),
		KnowledgeBase:    make(map[string]interface{}),
		MessageBus:       bus,
		EventSubscriptions: make(map[MessageType][]string),
		Ctx:              ctx,
		Cancel:           cancel,
	}

	// Agent needs to forward messages it wants to send to the MessageBus
	// This could be done by having Agent.Run directly call bus.SendMessage,
	// but an explicit outbound channel can sometimes simplify logic if
	// agent has complex sending patterns or retries. Let's stick to direct send for simplicity here.
	// agent.wg.Add(1)
	// go agent.sendLoop() // If using a dedicated outbound channel

	bus.RegisterAgent(id, agent.InboundMessages)

	return agent
}

// sendLoop is a potential goroutine for agents to send messages (alternative to direct SendMessage calls).
// func (a *Agent) sendLoop() {
// 	defer a.wg.Done()
// 	for {
// 		select {
// 		case msg, ok := <-a.OutboundChannel:
// 			if !ok {
// 				log.Printf("Agent %s outbound channel closed", a.ID)
// 				return
// 			}
// 			err := a.MessageBus.SendMessage(msg)
// 			if err != nil {
// 				log.Printf("Agent %s failed to send message %v: %v", a.ID, msg.Type, err)
// 				// Implement retry logic if needed
// 			}
// 		case <-a.Ctx.Done():
// 			log.Printf("Agent %s send loop shutting down", a.ID)
// 			return
// 		}
// 	}
// }


// Run starts the agent's main processing loop.
func (a *Agent) Run() {
	log.Printf("Agent %s started", a.ID)
	defer a.wg.Done()
	defer a.MessageBus.DeregisterAgent(a.ID) // Deregister when done

	for {
		select {
		case msg, ok := <-a.InboundMessages:
			if !ok {
				log.Printf("Agent %s inbound channel closed, shutting down.", a.ID)
				return // Channel closed, shutdown
			}
			a.handleMessage(msg)
		case <-a.Ctx.Done():
			log.Printf("Agent %s received shutdown signal (context done), shutting down.", a.ID)
			return // Context cancelled, shutdown
		}
	}
}

// Stop signals the agent to shut down gracefully.
func (a *Agent) Stop() {
	log.Printf("Agent %s received explicit stop signal, initiating graceful shutdown.", a.ID)
	a.Cancel() // Cancel the context
	// The Run loop will catch the context cancellation and exit.
	// The MessageBus will be deregistered by the defer in Run.
	// If using a separate sendLoop, wait for it: a.wg.Wait()
}

// handleMessage processes incoming messages based on their type.
func (a *Agent) handleMessage(msg Message) {
	log.Printf("Agent %s received message type %v from %s", a.ID, msg.Type, msg.SenderID)

	// Notify subscribers for this message type
	a.notifySubscribers(msg.Type, msg)

	switch msg.Type {
	// --- Core Agent Management Handlers ---
	case MessageTypeRequestState:
		a.handleRequestState(msg)
	case MessageTypeResponseState:
		a.handleResponseState(msg)
	case MessageTypeUpdateState:
		a.handleUpdateState(msg)
	case MessageTypeTerminateAgent: // Explicit termination message
		a.handleTerminateAgent(msg)
	case MessageTypeDiscoverAgents: // Bus sends list of agents
		a.handleDiscoverAgents(msg)

	// --- Information Processing & Analysis Handlers ---
	case MessageTypeProcessDataChunk:
		a.handleProcessDataChunk(msg)
	case MessageTypeSynthesizeReport:
		a.handleSynthesizeReport(msg)
	case MessageTypePerformCalculation:
		a.handlePerformCalculation(msg)
	case MessageTypeCategorizeData:
		a.handleCategorizeData(msg)
	case MessageTypeSummarizeText:
		a.handleSummarizeText(msg)
	case MessageTypeAnalyzeSentiment:
		a.handleAnalyzeSentiment(msg)

	// --- Coordination & Interaction Handlers ---
	case MessageTypeRequestCollaboration:
		a.handleRequestCollaboration(msg)
	case MessageTypeProposeActionPlan:
		a.handleProposeActionPlan(msg)
	case MessageTypeNegotiateParameter:
		a.handleNegotiateParameter(msg)
	case MessageTypeNegotiationUpdate:
		a.handleNegotiationUpdate(msg)
	case MessageTypeNegotiationResult:
		a.handleNegotiationResult(msg)
	case MessageTypeBroadcastAlert: // Handled implicitly by receiving the broadcast
		a.handleBroadcastAlert(msg) // Agent acts on the alert
	case MessageTypeRequestResource:
		a.handleRequestResource(msg)
	case MessageTypeMessageTypeResourceGrant:
		a.handleResourceGrant(msg) // Handle receiving a resource grant
	case MessageTypeMessageTypeResourceDenial:
		a.handleResourceDenial(msg) // Handle receiving a resource denial
	case MessageTypeOfferResource: // Handled implicitly by receiving the offer
		a.handleOfferResource(msg) // Agent considers the offer

	// --- Rudimentary Learning & Adaptation Handlers ---
	case MessageTypeUpdateInternalModel:
		a.handleUpdateInternalModel(msg)
	case MessageTypeAdjustBehavior:
		a.handleAdjustBehavior(msg)
	case MessageTypeLogEvent:
		a.handleLogEvent(msg)

	// --- Advanced/Abstract Concepts Handlers ---
	case MessageTypePerformSemanticMatch:
		a.handlePerformSemanticMatch(msg)
	case MessageTypeSimulateScenario:
		a.handleSimulateScenario(msg)
	case MessageTypeVerifyInformation:
		a.handleVerifyInformation(msg)
	case MessageTypePrioritizeTask:
		a.handlePrioritizeTask(msg)
	case MessageTypeAdaptProtocol:
		a.handleAdaptProtocol(msg)

	// --- Response Handlers (Agents receive these and update state/knowledge) ---
	case MessageTypeReportSynthesized:
		a.handleResponseReport(msg) // Agent receives a synthesized report from another agent
	case MessageTypeCalculationResult:
		a.handleResponseCalculation(msg) // Agent receives a calculation result
	case MessageTypeDataCategorized:
		a.handleResponseCategorized(msg) // Agent receives categorization result
	case MessageTypeTextSummarized:
		a.handleResponseSummarized(msg) // Agent receives text summary
	case MessageTypeSentimentResult:
		a.handleResponseSentiment(msg) // Agent receives sentiment analysis result
	case MessageTypeSemanticMatchResult:
		a.handleResponseSemanticMatch(msg) // Agent receives semantic match results
	case MessageTypeSimulationOutput:
		a.handleResponseSimulation(msg) // Agent receives simulation output
	case MessageTypeVerificationResult:
		a.handleResponseVerification(msg) // Agent receives verification result

	default:
		log.Printf("Agent %s received unknown message type %v", a.ID, msg.Type)
	}
}

// notifySubscribers sends a copy of the message to agents subscribed to this message type.
func (a *Agent) notifySubscribers(msgType MessageType, originalMsg Message) {
	subscribers, ok := a.EventSubscriptions[msgType]
	if !ok {
		return // No subscribers for this type
	}

	for _, subscriberID := range subscribers {
		// Create a copy of the message for the subscriber
		subscriberMsg := Message{
			SenderID:    originalMsg.SenderID,      // Original sender
			RecipientID: subscriberID,              // Subscriber is the new recipient
			Type:        msgType,                   // Original type
			Payload:     originalMsg.Payload,       // Original payload
			Timestamp:   time.Now(),                // New timestamp for the notification
		}
		// Optionally wrap payload to indicate it's a notification of an event?
		// Or add a flag to the message struct?
		// For now, just forward the message.

		err := a.MessageBus.SendMessage(subscriberMsg)
		if err != nil {
			log.Printf("Agent %s failed to notify subscriber %s for message type %v: %v",
				a.ID, subscriberID, msgType, err)
		} else {
			log.Printf("Agent %s notified subscriber %s for message type %v",
				a.ID, subscriberID, msgType)
		}
	}
}

// --- Agent Function Implementations (Message Handlers) ---

// Note: These are simplified handlers. Real AI logic would be significantly more complex.
// Payloads would need type assertions and specific structs.

// handleRequestState processes a request for this agent's state.
func (a *Agent) handleRequestState(msg Message) {
	log.Printf("Agent %s handling RequestState from %s", a.ID, msg.SenderID)
	// Respond with current state
	responsePayload := map[string]interface{}{
		"state": a.State,
		"parameters": a.Parameters,
		"knowledge_summary": fmt.Sprintf("Has %d knowledge entries", len(a.KnowledgeBase)), // Don't send full KB usually
	}
	responseMsg := Message{
		SenderID:    a.ID,
		RecipientID: msg.SenderID,
		Type:        MessageTypeResponseState,
		Payload:     responsePayload,
	}
	err := a.MessageBus.SendMessage(responseMsg)
	if err != nil {
		log.Printf("Agent %s failed to send ResponseState: %v", a.ID, err)
	}
}

// handleResponseState processes a response containing another agent's state.
func (a *Agent) handleResponseState(msg Message) {
	log.Printf("Agent %s handling ResponseState from %s", a.ID, msg.SenderID)
	// Update knowledge base or log information about the other agent
	if payload, ok := msg.Payload.(map[string]interface{}); ok {
		a.KnowledgeBase[fmt.Sprintf("agent_state_%s", msg.SenderID)] = payload
		log.Printf("Agent %s updated knowledge with state from %s", a.ID, msg.SenderID)
	} else {
		log.Printf("Agent %s received invalid payload for ResponseState from %s", a.ID, msg.SenderID)
	}
	// Example: use this state info for planning, collaboration requests, etc.
}

// handleUpdateState processes a request to update this agent's state.
func (a *Agent) handleUpdateState(msg Message) {
	log.Printf("Agent %s handling UpdateState from %s", a.ID, msg.SenderID)
	if updatePayload, ok := msg.Payload.(map[string]interface{}); ok {
		for key, value := range updatePayload {
			a.State[key] = value // Simple merge/overwrite
			log.Printf("Agent %s updated state key '%s'", a.ID, key)
		}
		// Respond with confirmation or updated state?
	} else {
		log.Printf("Agent %s received invalid payload for UpdateState from %s", a.ID, msg.SenderID)
		// Send an error response?
	}
}

// handleTerminateAgent processes a request for this agent to terminate.
func (a *Agent) handleTerminateAgent(msg Message) {
	log.Printf("Agent %s handling TerminateAgent from %s. Initiating shutdown.", a.ID, msg.SenderID)
	// Perform cleanup before stopping
	log.Printf("Agent %s performing shutdown cleanup...", a.ID)
	// ... cleanup logic (saving state, closing connections, etc.) ...
	a.Stop() // Signal graceful shutdown
}

// handleDiscoverAgents processes a response from the bus with a list of agents.
func (a *Agent) handleDiscoverAgents(msg Message) {
	log.Printf("Agent %s handling DiscoverAgents response from %s", a.ID, msg.SenderID)
	if agentList, ok := msg.Payload.([]string); ok {
		log.Printf("Agent %s discovered agents: %v", a.ID, agentList)
		// Update internal knowledge about available agents
		a.KnowledgeBase["discovered_agents"] = agentList
		// Maybe send RequestState to newly discovered agents?
	} else {
		log.Printf("Agent %s received invalid payload for DiscoverAgents from %s", a.ID, msg.SenderID)
	}
}

// handleProcessDataChunk processes a chunk of data.
func (a *Agent) handleProcessDataChunk(msg Message) {
	log.Printf("Agent %s handling ProcessDataChunk from %s", a.ID, msg.SenderID)
	// Payload might be a string, byte slice, map, etc.
	log.Printf("Agent %s processing data: %.50v...", a.ID, msg.Payload)
	// Example processing: count words, find patterns, validate format, etc.
	processedResult := fmt.Sprintf("Processed data from %s (size: %d)", msg.SenderID, len(fmt.Sprintf("%v", msg.Payload)))
	// Send a response or update state
	a.State["last_processed_data"] = processedResult // Update state
	// Or send a result message back:
	// responseMsg := Message{SenderID: a.ID, RecipientID: msg.SenderID, Type: MessageTypeProcessedResult, Payload: processedResult}
	// a.MessageBus.SendMessage(responseMsg)
}

// handleSynthesizeReport processes a request to synthesize a report.
func (a *Agent) handleSynthesizeReport(msg Message) {
	log.Printf("Agent %s handling SynthesizeReport from %s", a.ID, msg.SenderID)
	// Combine data from State, KnowledgeBase, possibly request info from others
	report := fmt.Sprintf("Report from Agent %s:\nState: %v\nKnowledge Summary: %s",
		a.ID, a.State, fmt.Sprintf("Contains %d knowledge entries", len(a.KnowledgeBase)))
	// Send the report back
	responseMsg := Message{
		SenderID:    a.ID,
		RecipientID: msg.SenderID,
		Type:        MessageTypeReportSynthesized,
		Payload:     report,
	}
	err := a.MessageBus.SendMessage(responseMsg)
	if err != nil {
		log.Printf("Agent %s failed to send ReportSynthesized: %v", a.ID, err)
	}
}

// handleResponseReport processes a synthesized report received from another agent.
func (a *Agent) handleResponseReport(msg Message) {
	log.Printf("Agent %s handling ResponseReport from %s", a.ID, msg.SenderID)
	if report, ok := msg.Payload.(string); ok {
		log.Printf("Agent %s received report from %s:\n%s", a.ID, msg.SenderID, report)
		// Process the report: extract info, update knowledge, trigger actions
		a.KnowledgeBase[fmt.Sprintf("report_from_%s_%s", msg.SenderID, msg.Timestamp.Format(time.RFC3339))] = report
	} else {
		log.Printf("Agent %s received invalid payload for ResponseReport from %s", a.ID, msg.SenderID)
	}
}

// handlePerformCalculation processes a calculation request.
func (a *Agent) handlePerformCalculation(msg Message) {
	log.Printf("Agent %s handling PerformCalculation from %s", a.ID, msg.SenderID)
	// Payload might contain parameters for the calculation
	params, ok := msg.Payload.(map[string]float64) // Example payload type
	result := 0.0
	if ok {
		// Example calculation: sum of values
		for _, val := range params {
			result += val
		}
		log.Printf("Agent %s performed calculation: %v", a.ID, result)
	} else {
		log.Printf("Agent %s received invalid payload for PerformCalculation from %s", a.ID, msg.SenderID)
		result = -1 // Indicate error
	}

	// Send the result back
	responseMsg := Message{
		SenderID:    a.ID,
		RecipientID: msg.SenderID,
		Type:        MessageTypeCalculationResult,
		Payload:     result,
	}
	err := a.MessageBus.SendMessage(responseMsg)
	if err != nil {
		log.Printf("Agent %s failed to send CalculationResult: %v", a.ID, err)
	}
}

// handleResponseCalculation processes a calculation result from another agent.
func (a *Agent) handleResponseCalculation(msg Message) {
	log.Printf("Agent %s handling ResponseCalculation from %s", a.ID, msg.SenderID)
	if result, ok := msg.Payload.(float64); ok {
		log.Printf("Agent %s received calculation result from %s: %v", a.ID, msg.SenderID, result)
		// Use the result: update state, make decision, trigger next step
		a.KnowledgeBase[fmt.Sprintf("calc_result_from_%s_%s", msg.SenderID, msg.Timestamp.Format(time.RFC3339))] = result
	} else {
		log.Printf("Agent %s received invalid payload for ResponseCalculation from %s", a.ID, msg.SenderID)
	}
}

// handleCategorizeData processes a request to categorize data.
func (a *Agent) handleCategorizeData(msg Message) {
	log.Printf("Agent %s handling CategorizeData from %s", a.ID, msg.SenderID)
	// Payload is the data to categorize
	data := msg.Payload
	category := "unknown" // Default

	// Example categorization logic (based on a simple parameter)
	if rule, ok := a.Parameters["categorization_rule"].(string); ok {
		dataStr := fmt.Sprintf("%v", data)
		if len(dataStr) > 50 {
			category = rule // E.g., "large_data"
		} else {
			category = "small_data"
		}
		log.Printf("Agent %s categorized data based on rule '%s': %s", a.ID, rule, category)
	} else {
		log.Printf("Agent %s categorizing data with default rule: %s", a.ID, category)
	}

	// Send the categorization result back
	responseMsg := Message{
		SenderID:    a.ID,
		RecipientID: msg.SenderID,
		Type:        MessageTypeDataCategorized,
		Payload:     map[string]interface{}{"data": data, "category": category},
	}
	err := a.MessageBus.SendMessage(responseMsg)
	if err != nil {
		log.Printf("Agent %s failed to send DataCategorized: %v", a.ID, err)
	}
}

// handleResponseCategorized processes categorization result from another agent.
func (a *Agent) handleResponseCategorized(msg Message) {
	log.Printf("Agent %s handling ResponseCategorized from %s", a.ID, msg.SenderID)
	if result, ok := msg.Payload.(map[string]interface{}); ok {
		if category, cok := result["category"].(string); cok {
			log.Printf("Agent %s received categorization '%s' for data from %s", a.ID, category, msg.SenderID)
			// Use the category: update state, make decision based on category
			a.KnowledgeBase[fmt.Sprintf("categorization_from_%s_%s", msg.SenderID, msg.Timestamp.Format(time.RFC3339))] = result
		}
	} else {
		log.Printf("Agent %s received invalid payload for ResponseCategorized from %s", a.ID, msg.SenderID)
	}
}


// handleSummarizeText processes a text summarization request.
func (a *Agent) handleSummarizeText(msg Message) {
	log.Printf("Agent %s handling SummarizeText from %s", a.ID, msg.SenderID)
	// Payload is the text string
	text, ok := msg.Payload.(string)
	summary := "Could not summarize"
	if ok {
		// Simple summarization: take first N characters or words
		summaryLength, pOk := a.Parameters["summary_length"].(int)
		if !pOk || summaryLength <= 0 {
			summaryLength = 50 // Default length
		}
		if len(text) > summaryLength {
			summary = text[:summaryLength] + "..."
		} else {
			summary = text
		}
		log.Printf("Agent %s summarized text", a.ID)
	} else {
		log.Printf("Agent %s received invalid payload for SummarizeText from %s", a.ID, msg.SenderID)
	}

	// Send the summary back
	responseMsg := Message{
		SenderID:    a.ID,
		RecipientID: msg.SenderID,
		Type:        MessageTypeTextSummarized,
		Payload:     summary,
	}
	err := a.MessageBus.SendMessage(responseMsg)
	if err != nil {
		log.Printf("Agent %s failed to send TextSummarized: %v", a.ID, err)
	}
}

// handleResponseSummarized processes text summary from another agent.
func (a *Agent) handleResponseSummarized(msg Message) {
	log.Printf("Agent %s handling ResponseSummarized from %s", a.ID, msg.SenderID)
	if summary, ok := msg.Payload.(string); ok {
		log.Printf("Agent %s received summary from %s: %s", a.ID, msg.SenderID, summary)
		// Use the summary: update knowledge, decide if more detail is needed
		a.KnowledgeBase[fmt.Sprintf("summary_from_%s_%s", msg.SenderID, msg.Timestamp.Format(time.RFC3339))] = summary
	} else {
		log.Printf("Agent %s received invalid payload for ResponseSummarized from %s", a.ID, msg.SenderID)
	}
}


// handleAnalyzeSentiment processes a sentiment analysis request.
func (a *Agent) handleAnalyzeSentiment(msg Message) {
	log.Printf("Agent %s handling AnalyzeSentiment from %s", a.ID, msg.SenderID)
	// Payload is the text string
	text, ok := msg.Payload.(string)
	sentiment := "neutral" // Default

	if ok {
		// Very basic sentiment (example)
		lowerText := string(text) // Simplified, real analysis needs more nuance
		if len(lowerText) > 10 && lowerText[0] == 'p' { // e.g., starts with 'positive'
			sentiment = "positive"
		} else if len(lowerText) > 10 && lowerText[0] == 'n' { // e.g., starts with 'negative'
			sentiment = "negative"
		}
		log.Printf("Agent %s analyzed sentiment: %s", a.ID, sentiment)
	} else {
		log.Printf("Agent %s received invalid payload for AnalyzeSentiment from %s", a.ID, msg.SenderID)
	}

	// Send the result back
	responseMsg := Message{
		SenderID:    a.ID,
		RecipientID: msg.SenderID,
		Type:        MessageTypeSentimentResult,
		Payload:     sentiment,
	}
	err := a.MessageBus.SendMessage(responseMsg)
	if err != nil {
		log.Printf("Agent %s failed to send SentimentResult: %v", a.ID, err)
	}
}

// handleResponseSentiment processes sentiment analysis result from another agent.
func (a *Agent) handleResponseSentiment(msg Message) {
	log.Printf("Agent %s handling ResponseSentiment from %s", a.ID, msg.SenderID)
	if sentiment, ok := msg.Payload.(string); ok {
		log.Printf("Agent %s received sentiment '%s' from %s", a.ID, sentiment, msg.SenderID)
		// Use the sentiment: update state, make decision based on sentiment
		a.KnowledgeBase[fmt.Sprintf("sentiment_from_%s_%s", msg.SenderID, msg.Timestamp.Format(time.RFC3339))] = sentiment
	} else {
		log.Printf("Agent %s received invalid payload for ResponseSentiment from %s", a.ID, msg.SenderID)
	}
}


// handleRequestCollaboration processes a collaboration request.
func (a *Agent) handleRequestCollaboration(msg Message) {
	log.Printf("Agent %s handling RequestCollaboration from %s", a.ID, msg.SenderID)
	// Payload might describe the task, required resources, deadline, etc.
	// Evaluate if this agent is able/willing to collaborate based on State, Parameters, workload
	collaborationDetails, ok := msg.Payload.(map[string]interface{})
	response := "rejected" // Default response
	if ok {
		log.Printf("Agent %s received collaboration request details: %v", a.ID, collaborationDetails)
		// Simple rule: if agent has 'collaboration_mode' parameter set to true
		if mode, pOk := a.Parameters["collaboration_mode"].(bool); pOk && mode {
			response = "accepted"
			log.Printf("Agent %s accepted collaboration request from %s", a.ID, msg.SenderID)
			// Update internal task queue, state to reflect collaboration
			a.State[fmt.Sprintf("collaborating_with_%s", msg.SenderID)] = collaborationDetails
		} else {
			log.Printf("Agent %s rejected collaboration request from %s (collaboration_mode not true)", a.ID, msg.SenderID)
		}
	} else {
		log.Printf("Agent %s received invalid payload for RequestCollaboration from %s", a.ID, msg.SenderID)
	}

	// Send response back
	responseMsg := Message{
		SenderID:    a.ID,
		RecipientID: msg.SenderID,
		Type:        MessageTypeNegotiationResult, // Using negotiation result type for simple accept/reject
		Payload:     map[string]interface{}{"status": response, "request_id": collaborationDetails["request_id"]}, // Include original request ID
	}
	err := a.MessageBus.SendMessage(responseMsg)
	if err != nil {
		log.Printf("Agent %s failed to send collaboration response: %v", a.ID, err)
	}
}


// handleProposeActionPlan processes a proposed action plan.
func (a *Agent) handleProposeActionPlan(msg Message) {
	log.Printf("Agent %s handling ProposeActionPlan from %s", a.ID, msg.SenderID)
	// Payload is likely a list of steps or tasks
	plan, ok := msg.Payload.([]map[string]interface{})
	if ok {
		log.Printf("Agent %s received action plan with %d steps from %s", a.ID, len(plan), msg.SenderID)
		// Evaluate plan: check feasibility, conflicts with current tasks, alignment with goals
		evaluation := "under_review" // Default evaluation
		// Simple rule: if plan has less than 5 steps, accept it
		if len(plan) < 5 {
			evaluation = "accepted"
			log.Printf("Agent %s accepted proposed action plan from %s", a.ID, msg.SenderID)
			// Add tasks from the plan to agent's internal task queue/state
			a.State["current_plan"] = plan // Example: Overwrite current plan
		} else {
			evaluation = "rejected"
			log.Printf("Agent %s rejected proposed action plan from %s (too many steps)", a.ID, msg.SenderID)
		}
		// Send response (e.g., accept, reject, propose modification)
		responseMsg := Message{
			SenderID:    a.ID,
			RecipientID: msg.SenderID,
			Type:        MessageTypeNegotiationResult, // Using negotiation result for plan acceptance
			Payload:     map[string]interface{}{"plan_id": msg.Payload, "status": evaluation}, // Needs a proper plan ID
		}
		err := a.MessageBus.SendMessage(responseMsg)
		if err != nil {
			log.Printf("Agent %s failed to send plan evaluation: %v", a.ID, err)
		}

	} else {
		log.Printf("Agent %s received invalid payload for ProposeActionPlan from %s", a.ID, msg.SenderID)
		// Send error response
	}
}

// handleNegotiateParameter starts or continues a negotiation.
func (a *Agent) handleNegotiateParameter(msg Message) {
	log.Printf("Agent %s handling NegotiateParameter from %s", a.ID, msg.SenderID)
	// Payload contains parameter name, current proposal, context
	// Implement negotiation logic: evaluate proposal against goals/constraints, make counter-proposal or accept/reject
	negotiationState, ok := msg.Payload.(map[string]interface{})
	if !ok {
		log.Printf("Agent %s received invalid payload for NegotiateParameter from %s", a.ID, msg.SenderID)
		return
	}

	paramName, nameOk := negotiationState["parameter"].(string)
	proposal, proposalOk := negotiationState["proposal"] // Could be any type
	if !nameOk || !proposalOk {
		log.Printf("Agent %s invalid negotiation state payload from %s", a.ID, msg.SenderID)
		return
	}

	log.Printf("Agent %s evaluating proposal for '%s': %v from %s", a.ID, paramName, proposal, msg.SenderID)

	// Simple negotiation: accept if proposal is within a certain range based on agent's own parameters
	acceptableRange, rangeOk := a.Parameters[paramName+"_acceptable_range"].([]float64) // Example parameter pattern
	currentValue, valueOk := proposal.(float64) // Assume float negotiation for simplicity

	var responseType MessageType
	var responsePayload interface{}

	if rangeOk && valueOk && currentValue >= acceptableRange[0] && currentValue <= acceptableRange[1] {
		responseType = MessageTypeNegotiationResult
		responsePayload = map[string]interface{}{"parameter": paramName, "status": "accepted", "final_value": currentValue}
		log.Printf("Agent %s accepted proposal for '%s'", a.ID, paramName)
		// Update agent's state/parameter based on accepted value if necessary
		a.State[paramName] = currentValue // Example: update state
	} else {
		// Reject or counter-propose
		responseType = MessageTypeNegotiationUpdate
		// Simple counter: propose the middle of the acceptable range if it exists
		if rangeOk && len(acceptableRange) == 2 {
			counterProposal := (acceptableRange[0] + acceptableRange[1]) / 2
			responsePayload = map[string]interface{}{"parameter": paramName, "status": "counter_proposal", "proposal": counterProposal, "reason": "proposal outside acceptable range"}
			log.Printf("Agent %s counter-proposed %v for '%s'", a.ID, counterProposal, paramName)
		} else {
			responseType = MessageTypeNegotiationResult // End negotiation if no counter-proposal logic
			responsePayload = map[string]interface{}{"parameter": paramName, "status": "rejected", "reason": "proposal outside range or no counter strategy"}
			log.Printf("Agent %s rejected proposal for '%s'", a.ID, paramName)
		}
	}

	responseMsg := Message{
		SenderID:    a.ID,
		RecipientID: msg.SenderID,
		Type:        responseType,
		Payload:     responsePayload,
	}
	err := a.MessageBus.SendMessage(responseMsg)
	if err != nil {
		log.Printf("Agent %s failed to send negotiation response: %v", a.ID, err)
	}
}

// handleNegotiationUpdate processes an ongoing negotiation message (e.g., a counter-proposal).
func (a *Agent) handleNegotiationUpdate(msg Message) {
	log.Printf("Agent %s handling NegotiationUpdate from %s", a.ID, msg.SenderID)
	// Similar logic to handleNegotiateParameter, but specific to ongoing state
	a.handleNegotiateParameter(msg) // Re-use logic for simplicity, real system would track negotiation state
}

// handleNegotiationResult processes the final result of a negotiation.
func (a *Agent) handleNegotiationResult(msg Message) {
	log.Printf("Agent %s handling NegotiationResult from %s", a.ID, msg.SenderID)
	result, ok := msg.Payload.(map[string]interface{})
	if ok {
		paramName, nameOk := result["parameter"].(string)
		status, statusOk := result["status"].(string)
		if nameOk && statusOk {
			log.Printf("Agent %s received final negotiation result for '%s': %s from %s", a.ID, paramName, status, msg.SenderID)
			// Act based on the result: update state/parameter if accepted, find alternative if rejected, proceed with collaborative task etc.
			if status == "accepted" {
				if finalValue, valueOk := result["final_value"]; valueOk {
					a.State[paramName] = finalValue // Update internal state with agreed value
					log.Printf("Agent %s updated state '%s' to %v based on negotiation result", a.ID, paramName, finalValue)
				}
			}
		}
	} else {
		log.Printf("Agent %s received invalid payload for NegotiationResult from %s", a.ID, msg.SenderID)
	}
}


// handleBroadcastAlert processes a broadcast alert message.
func (a *Agent) handleBroadcastAlert(msg Message) {
	log.Printf("Agent %s received BroadcastAlert from %s: %v", a.ID, msg.SenderID, msg.Payload)
	// Act on the alert: change priority, update state, initiate specific behavior, log
	alertInfo, ok := msg.Payload.(map[string]interface{})
	if ok {
		alertType, typeOk := alertInfo["type"].(string)
		if typeOk {
			log.Printf("Agent %s reacting to alert type: %s", a.ID, alertType)
			// Example reaction: if alert is "urgent_task", reprioritize
			if alertType == "urgent_task" {
				a.State["current_task_priority"] = "high"
				log.Printf("Agent %s setting task priority to high due to urgent_task alert", a.ID)
				// Trigger task reprioritization logic
				a.handlePrioritizeTask(Message{SenderID: a.ID, RecipientID: a.ID, Type: MessageTypePrioritizeTask, Payload: "urgent"}) // Self-message to trigger reprioritization
			}
			// Log the alert
			a.handleLogEvent(Message{SenderID: "internal", RecipientID: a.ID, Type: MessageTypeLogEvent, Payload: map[string]interface{}{"event": "alert_received", "alert_details": alertInfo}})
		}
	}
}

// handleRequestResource processes a request for a resource.
func (a *Agent) handleRequestResource(msg Message) {
	log.Printf("Agent %s handling RequestResource from %s", a.ID, msg.SenderID)
	// Payload specifies resource type, amount, context
	resourceRequest, ok := msg.Payload.(map[string]interface{})
	responseType := MessageTypeMessageTypeResourceDenial // Default
	responsePayload := map[string]interface{}{"request_id": resourceRequest["request_id"], "reason": "unknown resource or unavailable"} // Include request ID

	if ok {
		resourceType, typeOk := resourceRequest["resource_type"].(string)
		amount, amountOk := resourceRequest["amount"] // Can be int, float, etc.

		if typeOk && amountOk {
			log.Printf("Agent %s evaluating resource request for type '%s', amount %v from %s", a.ID, resourceType, amount, msg.SenderID)
			// Example: Check if agent has resource and is willing to share
			availableResource, resOk := a.State[resourceType] // Check internal state for resource
			if resOk {
				// Simple logic: If available amount is greater than requested amount
				if availableFloat, isFloat := availableResource.(float64); isFloat {
					if requestedFloat, reqIsFloat := amount.(float64); reqIsFloat && availableFloat >= requestedFloat {
						responseType = MessageTypeMessageTypeResourceGrant
						responsePayload["status"] = "granted"
						responsePayload["allocated_amount"] = requestedFloat // Grant exactly what was requested
						// Update internal state to reflect resource allocation
						a.State[resourceType] = availableFloat - requestedFloat // Decrease available resource
						log.Printf("Agent %s granted resource '%s' amount %v to %s", a.ID, resourceType, requestedFloat, msg.SenderID)
					} else {
						responsePayload["reason"] = "not enough available"
						log.Printf("Agent %s denied resource '%s' request from %s: not enough", a.ID, resourceType, msg.SenderID)
					}
				} else {
					responsePayload["reason"] = "resource type mismatch or unmanageable amount type"
					log.Printf("Agent %s denied resource '%s' request from %s: type mismatch", a.ID, resourceType, msg.SenderID)
				}
			} else {
				responsePayload["reason"] = "resource type not available on this agent"
				log.Printf("Agent %s denied resource '%s' request from %s: not available", a.ID, resourceType, msg.SenderID)
			}
		} else {
			responsePayload["reason"] = "invalid request format"
			log.Printf("Agent %s denied resource request from %s: invalid format", a.ID, msg.SenderID)
		}
	} else {
		responsePayload["reason"] = "invalid payload"
		log.Printf("Agent %s denied resource request from %s: invalid payload", a.ID, msg.SenderID)
	}

	// Send response back
	responseMsg := Message{
		SenderID:    a.ID,
		RecipientID: msg.SenderID,
		Type:        responseType,
		Payload:     responsePayload,
	}
	err := a.MessageBus.SendMessage(responseMsg)
	if err != nil {
		log.Printf("Agent %s failed to send resource response: %v", a.ID, err)
	}
}

// handleResourceGrant processes receiving a resource grant.
func (a *Agent) handleResourceGrant(msg Message) {
	log.Printf("Agent %s handling ResourceGrant from %s", a.ID, msg.SenderID)
	grantDetails, ok := msg.Payload.(map[string]interface{})
	if ok {
		// Use the granted resource: update state, proceed with tasks that needed the resource
		log.Printf("Agent %s received resource grant: %v from %s", a.ID, grantDetails, msg.SenderID)
		// Example: Add granted amount to internal resource state
		if resourceType, typeOk := grantDetails["resource_type"].(string); typeOk {
			if allocatedAmount, amountOk := grantDetails["allocated_amount"]; amountOk {
				currentAmount, currentOk := a.State[resourceType]
				if !currentOk {
					currentAmount = 0.0 // Assume 0 if not exists, handle types
				}
				if currentFloat, isFloat := currentAmount.(float64); isFloat {
					if allocatedFloat, isAllocatedFloat := allocatedAmount.(float64); isAllocatedFloat {
						a.State[resourceType] = currentFloat + allocatedFloat
						log.Printf("Agent %s updated internal resource '%s' to %v", a.ID, resourceType, a.State[resourceType])
						// Trigger tasks waiting for this resource
					}
				} // Handle other types if needed
			}
		}
	} else {
		log.Printf("Agent %s received invalid payload for ResourceGrant from %s", a.ID, msg.SenderID)
	}
}

// handleResourceDenial processes receiving a resource denial.
func (a *Agent) handleResourceDenial(msg Message) {
	log.Printf("Agent %s handling ResourceDenial from %s", a.ID, msg.SenderID)
	denialDetails, ok := msg.Payload.(map[string]interface{})
	if ok {
		// Act on the denial: try another agent, wait, find alternative strategy
		log.Printf("Agent %s received resource denial: %v from %s", a.ID, denialDetails, msg.SenderID)
		// Example: Log and find alternative agent if reason allows
		if reason, reasonOk := denialDetails["reason"].(string); reasonOk {
			log.Printf("Agent %s denied resource due to: %s. Finding alternative...", a.ID, reason)
			// Trigger logic to find another agent or change plan
		}
	} else {
		log.Printf("Agent %s received invalid payload for ResourceDenial from %s", a.ID, msg.SenderID)
	}
}

// handleOfferResource processes receiving a resource offer.
func (a *Agent) handleOfferResource(msg Message) {
	log.Printf("Agent %s received OfferResource from %s: %v", a.ID, msg.SenderID, msg.Payload)
	// Evaluate the offer: check if resource is needed, log for future use
	offerDetails, ok := msg.Payload.(map[string]interface{})
	if ok {
		resourceType, typeOk := offerDetails["resource_type"].(string)
		amount, amountOk := offerDetails["amount"]
		if typeOk && amountOk {
			log.Printf("Agent %s received offer for '%s' amount %v from %s", a.ID, resourceType, amount, msg.SenderID)
			// Example: If agent's state indicates a need for this resource, send a RequestResource message back
			if needAmount, needOk := a.State["needs_"+resourceType].(float64); needOk && needAmount > 0 {
				log.Printf("Agent %s needs resource '%s', requesting part of the offer...", a.ID, resourceType)
				requestMsg := Message{
					SenderID:    a.ID,
					RecipientID: msg.SenderID,
					Type:        MessageTypeRequestResource,
					Payload:     map[string]interface{}{"request_id": "need_"+resourceType+"_"+a.ID, "resource_type": resourceType, "amount": needAmount},
				}
				err := a.MessageBus.SendMessage(requestMsg)
				if err != nil {
					log.Printf("Agent %s failed to respond to offer with request: %v", a.ID, err)
				}
			} else {
				// Log the offer for future reference
				a.KnowledgeBase[fmt.Sprintf("offer_from_%s_%s", msg.SenderID, msg.Timestamp.Format(time.RFC3339))] = offerDetails
			}
		}
	} else {
		log.Printf("Agent %s received invalid payload for OfferResource from %s", a.ID, msg.SenderID)
	}
}


// handleUpdateInternalModel processes a request to update the agent's internal model/parameters.
func (a *Agent) handleUpdateInternalModel(msg Message) {
	log.Printf("Agent %s handling UpdateInternalModel from %s", a.ID, msg.SenderID)
	// Payload contains updates for Parameters or KnowledgeBase (e.g., new rule, weight update, data points)
	updatePayload, ok := msg.Payload.(map[string]interface{})
	if ok {
		log.Printf("Agent %s applying model updates: %v", a.ID, updatePayload)
		// Example: update specific parameters or add knowledge
		if paramsUpdate, pOk := updatePayload["parameters"].(map[string]interface{}); pOk {
			for key, value := range paramsUpdate {
				a.Parameters[key] = value
				log.Printf("Agent %s updated parameter '%s'", a.ID, key)
			}
		}
		if kbUpdate, kbOk := updatePayload["knowledge"].(map[string]interface{}); kbOk {
			for key, value := range kbUpdate {
				a.KnowledgeBase[key] = value
				log.Printf("Agent %s updated knowledge entry '%s'", a.ID, key)
			}
		}
		// Real update logic would be more complex (e.g., gradient descent, Bayesian update)
	} else {
		log.Printf("Agent %s received invalid payload for UpdateInternalModel from %s", a.ID, msg.SenderID)
	}
}

// handleAdjustBehavior processes a request to adjust agent's behavior parameters/rules.
func (a *Agent) handleAdjustBehavior(msg Message) {
	log.Printf("Agent %s handling AdjustBehavior from %s", a.ID, msg.SenderID)
	// Payload contains instructions on how to modify behavior (e.g., change risk tolerance, switch strategy)
	adjustment, ok := msg.Payload.(map[string]interface{})
	if ok {
		log.Printf("Agent %s adjusting behavior based on: %v", a.ID, adjustment)
		// Example: Adjust a risk parameter
		if riskFactor, rOk := adjustment["risk_factor"].(float64); rOk {
			a.Parameters["risk_tolerance"] = riskFactor
			log.Printf("Agent %s set risk tolerance to %v", a.ID, riskFactor)
		}
		// Other adjustments based on message content
		if strategy, sOk := adjustment["strategy"].(string); sOk {
			a.Parameters["current_strategy"] = strategy
			log.Printf("Agent %s switched strategy to '%s'", a.ID, strategy)
			// Potentially clear current tasks or re-plan based on new strategy
		}
	} else {
		log.Printf("Agent %s received invalid payload for AdjustBehavior from %s", a.ID, msg.SenderID)
	}
}

// handleLogEvent processes a request to log an event in the agent's history or knowledge.
func (a *Agent) handleLogEvent(msg Message) {
	log.Printf("Agent %s handling LogEvent from %s", a.ID, msg.SenderID)
	// Payload contains details of the event
	eventDetails := msg.Payload
	log.Printf("Agent %s logging event: %v", a.ID, eventDetails)
	// Store event in knowledge base or a dedicated log structure
	// Using a timestamped key to keep history
	a.KnowledgeBase[fmt.Sprintf("event_%s_%s", time.Now().Format(time.RFC3339Nano), msg.SenderID)] = eventDetails
	// This logged data can later be used by handleUpdateInternalModel or handleAnalyzeData
}

// handlePerformSemanticMatch processes a request to find semantically similar info in KB/State.
func (a *Agent) handlePerformSemanticMatch(msg Message) {
	log.Printf("Agent %s handling PerformSemanticMatch from %s", a.ID, msg.SenderID)
	// Payload contains a query (e.g., string, keywords, vector)
	query, ok := msg.Payload.(string) // Simple string query example
	results := make(map[string]interface{})

	if ok {
		log.Printf("Agent %s performing semantic match for query: '%s'", a.ID, query)
		// Simple match: check if query string appears in knowledge base keys or string values
		queryLower := string(query) // Simplified, real matching needs NLP/embeddings
		for key, value := range a.KnowledgeBase {
			keyLower := string(key)
			valueStr := fmt.Sprintf("%v", value) // Convert value to string
			valueLower := string(valueStr)

			if string(keyLower).Contains(queryLower) || string(valueLower).Contains(queryLower) {
				results[key] = value // Found a match
			}
		}
		log.Printf("Agent %s found %d semantic matches for query '%s'", a.ID, len(results), query)
	} else {
		log.Printf("Agent %s received invalid payload for PerformSemanticMatch from %s", a.ID, msg.SenderID)
	}

	// Send the results back
	responseMsg := Message{
		SenderID:    a.ID,
		RecipientID: msg.SenderID,
		Type:        MessageTypeSemanticMatchResult,
		Payload:     results,
	}
	err := a.MessageBus.SendMessage(responseMsg)
	if err != nil {
		log.Printf("Agent %s failed to send SemanticMatchResult: %v", a.ID, err)
	}
}

// handleResponseSemanticMatch processes semantic match results from another agent.
func (a *Agent) handleResponseSemanticMatch(msg Message) {
	log.Printf("Agent %s handling ResponseSemanticMatch from %s", a.ID, msg.SenderID)
	results, ok := msg.Payload.(map[string]interface{})
	if ok {
		log.Printf("Agent %s received semantic match results from %s: %v", a.ID, msg.SenderID, results)
		// Use the results: integrate into knowledge, decide next action based on relevant info found
		for key, value := range results {
			// Store results, maybe add source agent ID to the key
			a.KnowledgeBase[fmt.Sprintf("sem_match_from_%s_%s_%s", msg.SenderID, msg.Timestamp.Format(time.RFC3339Nano), key)] = value
		}
	} else {
		log.Printf("Agent %s received invalid payload for ResponseSemanticMatch from %s", a.ID, msg.SenderID)
	}
}

// handleSimulateScenario processes a request to run a small internal simulation.
func (a *Agent) handleSimulateScenario(msg Message) {
	log.Printf("Agent %s handling SimulateScenario from %s", a.ID, msg.SenderID)
	// Payload contains scenario parameters/initial state
	scenarioParams, ok := msg.Payload.(map[string]interface{})
	simulationOutput := map[string]interface{}{"status": "failed", "reason": "invalid payload"}

	if ok {
		log.Printf("Agent %s running simulation with parameters: %v", a.ID, scenarioParams)
		// Simple simulation: calculate an outcome based on parameters and internal state/knowledge
		initialValue, vOk := scenarioParams["initial_value"].(float64)
		growthRate, rOk := a.Parameters["simulation_growth_rate"].(float64) // Use internal parameter
		steps, sOk := scenarioParams["steps"].(int)

		if vOk && rOk && sOk && steps > 0 {
			currentValue := initialValue
			simHistory := []float64{}
			for i := 0; i < steps; i++ {
				currentValue *= (1 + growthRate) // Simple exponential growth
				simHistory = append(simHistory, currentValue)
			}
			simulationOutput = map[string]interface{}{
				"status": "success",
				"final_value": currentValue,
				"history": simHistory,
			}
			log.Printf("Agent %s simulation finished. Final value: %v", a.ID, currentValue)
		} else {
			simulationOutput["reason"] = "missing or invalid simulation parameters"
			log.Printf("Agent %s failed simulation: missing/invalid parameters", a.ID)
		}
	} else {
		log.Printf("Agent %s received invalid payload for SimulateScenario from %s", a.ID, msg.SenderID)
	}

	// Send simulation output back
	responseMsg := Message{
		SenderID:    a.ID,
		RecipientID: msg.SenderID,
		Type:        MessageTypeSimulationOutput,
		Payload:     simulationOutput,
	}
	err := a.MessageBus.SendMessage(responseMsg)
	if err != nil {
		log.Printf("Agent %s failed to send SimulationOutput: %v", a.ID, err)
	}
}

// handleResponseSimulation processes simulation output from another agent.
func (a *Agent) handleResponseSimulation(msg Message) {
	log.Printf("Agent %s handling ResponseSimulation from %s", a.ID, msg.SenderID)
	simulationOutput, ok := msg.Payload.(map[string]interface{})
	if ok {
		log.Printf("Agent %s received simulation output from %s: %v", a.ID, msg.SenderID, simulationOutput)
		// Use simulation results: update knowledge, inform decision making, compare scenarios
		a.KnowledgeBase[fmt.Sprintf("simulation_from_%s_%s", msg.SenderID, msg.Timestamp.Format(time.RFC3339))] = simulationOutput
	} else {
		log.Printf("Agent %s received invalid payload for ResponseSimulation from %s", a.ID, msg.SenderID)
	}
}


// handleVerifyInformation processes a request to verify information.
func (a *Agent) handleVerifyInformation(msg Message) {
	log.Printf("Agent %s handling VerifyInformation from %s", a.ID, msg.SenderID)
	// Payload contains the information to verify (e.g., a claim, a data point)
	infoToVerify, ok := msg.Payload.(string) // Simple string claim example
	verificationResult := "undetermined" // Default

	if ok {
		log.Printf("Agent %s verifying information: '%s'", a.ID, infoToVerify)
		// Example verification: check against internal knowledge base or known facts
		if knownFact, factOk := a.KnowledgeBase["known_fact: "+infoToVerify].(bool); factOk {
			if knownFact {
				verificationResult = "verified_true"
			} else {
				verificationResult = "verified_false"
			}
			log.Printf("Agent %s verified info '%s' as %s", a.ID, infoToVerify, verificationResult)
		} else {
			// Could send requests to other agents (MessageTypeVerifyInformation) or external sources (abstracted)
			verificationResult = "needs_external_check" // Indicate need for external verification
			log.Printf("Agent %s cannot verify info '%s' internally, needs external check", a.ID, infoToVerify)
			// Example: Request verification from another agent (circular, but shows coordination)
			// verificationRequest := Message{
			//     SenderID: a.ID, RecipientID: "AgentB", // Assume AgentB is a verifier
			//     Type: MessageTypeVerifyInformation, Payload: infoToVerify,
			// }
			// a.MessageBus.SendMessage(verificationRequest)
		}
	} else {
		log.Printf("Agent %s received invalid payload for VerifyInformation from %s", a.ID, msg.SenderID)
	}

	// Send verification result back
	responseMsg := Message{
		SenderID:    a.ID,
		RecipientID: msg.SenderID,
		Type:        MessageTypeVerificationResult,
		Payload:     map[string]interface{}{"info": infoToVerify, "result": verificationResult},
	}
	err := a.MessageBus.SendMessage(responseMsg)
	if err != nil {
		log.Printf("Agent %s failed to send VerificationResult: %v", a.ID, err)
	}
}

// handleResponseVerification processes verification result from another agent.
func (a *Agent) handleResponseVerification(msg Message) {
	log.Printf("Agent %s handling ResponseVerification from %s", a.ID, msg.SenderID)
	resultPayload, ok := msg.Payload.(map[string]interface{})
	if ok {
		info, infoOk := resultPayload["info"].(string)
		result, resultOk := resultPayload["result"].(string)
		if infoOk && resultOk {
			log.Printf("Agent %s received verification result for '%s': %s from %s", a.ID, info, result, msg.SenderID)
			// Use the verification result: update knowledge, discard or trust information, update state
			a.KnowledgeBase[fmt.Sprintf("verification_result_for_%s_from_%s_%s", info, msg.SenderID, msg.Timestamp.Format(time.RFC3339Nano))] = result
		}
	} else {
		log.Printf("Agent %s received invalid payload for ResponseVerification from %s", a.ID, msg.SenderID)
	}
}


// handlePrioritizeTask processes a request to re-evaluate and prioritize tasks.
func (a *Agent) handlePrioritizeTask(msg Message) {
	log.Printf("Agent %s handling PrioritizeTask from %s", a.ID, msg.SenderID)
	// Payload might indicate criteria (e.g., "urgent", "by_deadline", "resource_needs")
	criteria, ok := msg.Payload.(string) // Simple string criteria
	log.Printf("Agent %s reprioritizing tasks based on criteria: '%s'", a.ID, criteria)
	// Example prioritization: Simple state change
	if criteria == "urgent" {
		a.State["task_order"] = "urgent_first"
	} else if criteria == "by_deadline" {
		a.State["task_order"] = "deadline_nearest"
	} else {
		a.State["task_order"] = "default"
	}
	log.Printf("Agent %s updated task order strategy to: %v", a.ID, a.State["task_order"])
	// Real prioritization involves manipulating an internal task queue based on task properties and criteria.
}


// handleAdaptProtocol processes a request to adapt communication protocol for specific agents.
func (a *Agent) handleAdaptProtocol(msg Message) {
	log.Printf("Agent %s handling AdaptProtocol from %s", a.ID, msg.SenderID)
	// Payload specifies which agent(s) and which protocol/parameters to use
	adaptationDetails, ok := msg.Payload.(map[string]interface{})
	if ok {
		targetAgentID, idOk := adaptationDetails["target_agent"].(string)
		protocolParams, paramsOk := adaptationDetails["protocol_parameters"].(map[string]interface{})

		if idOk && paramsOk {
			log.Printf("Agent %s adapting protocol for %s with parameters: %v", a.ID, targetAgentID, protocolParams)
			// Update internal communication settings for messages sent to targetAgentID
			// This would involve storing protocol details indexed by recipient ID
			if _, exists := a.KnowledgeBase["communication_protocols"]; !exists {
				a.KnowledgeBase["communication_protocols"] = make(map[string]map[string]interface{})
			}
			protocols, _ := a.KnowledgeBase["communication_protocols"].(map[string]map[string]interface{})
			protocols[targetAgentID] = protocolParams
			a.KnowledgeBase["communication_protocols"] = protocols // Ensure map is updated back
			log.Printf("Agent %s updated protocol settings for %s", a.ID, targetAgentID)
		} else {
			log.Printf("Agent %s received invalid payload for AdaptProtocol from %s", a.ID, msg.SenderID)
		}
	} else {
		log.Printf("Agent %s received invalid payload for AdaptProtocol from %s", a.ID, msg.SenderID)
	}
}

// SubscribeToEvents allows an agent to register interest in message types from others.
// In this simple model, the *receiving* agent (the one whose event is occurring) notifies the *subscribing* agent.
// A more robust model might have a dedicated EventManager agent.
// For this implementation, we'll add a dummy handler on the receiving side to show how it *could* work.
// Agent A wants to subscribe to Agent B's MessageTypeReportSynthesized. A sends MessageTypeSubscribeEvents to B.
// B receives it, adds A to its EventSubscriptions list for that type.
// When B later handles MessageTypeSynthesizeReport and sends MessageTypeReportSynthesized,
// its notifySubscribers method sends a copy to A.

// handleSubscribeEvents processes a subscription request from another agent.
func (a *Agent) handleSubscribeEvents(msg Message) {
	log.Printf("Agent %s handling SubscribeEvents from %s", a.ID, msg.SenderID)
	// Payload contains the message types the sender wants to subscribe to
	typesToSubscribe, ok := msg.Payload.([]MessageType)
	if ok {
		log.Printf("Agent %s received subscription request for types %v from %s", a.ID, typesToSubscribe, msg.SenderID)
		for _, msgType := range typesToSubscribe {
			// Add senderID to the list of subscribers for this message type
			subscribers := a.EventSubscriptions[msgType]
			// Check if already subscribed
			isSubscribed := false
			for _, subID := range subscribers {
				if subID == msg.SenderID {
					isSubscribed = true
					break
				}
			}
			if !isSubscribed {
				a.EventSubscriptions[msgType] = append(subscribers, msg.SenderID)
				log.Printf("Agent %s added %s as subscriber for type %v", a.ID, msg.SenderID, msgType)
			} else {
				log.Printf("Agent %s: %s is already subscribed to type %v", a.ID, msg.SenderID, msgType)
			}
		}
		// Optionally send a confirmation response
	} else {
		log.Printf("Agent %s received invalid payload for SubscribeEvents from %s", a.ID, msg.SenderID)
	}
}

// handleUnsubscribeEvents processes an unsubscription request from another agent.
func (a *Agent) handleUnsubscribeEvents(msg Message) {
	log.Printf("Agent %s handling UnsubscribeEvents from %s", a.ID, msg.SenderID)
	// Payload contains the message types the sender wants to unsubscribe from
	typesToUnsubscribe, ok := msg.Payload.([]MessageType)
	if ok {
		log.Printf("Agent %s received unsubscription request for types %v from %s", a.ID, typesToUnsubscribe, msg.SenderID)
		for _, msgType := range typesToUnsubscribe {
			subscribers := a.EventSubscriptions[msgType]
			newSubscribers := []string{}
			removed := false
			for _, subID := range subscribers {
				if subID != msg.SenderID {
					newSubscribers = append(newSubscribers, subID)
				} else {
					removed = true
				}
			}
			a.EventSubscriptions[msgType] = newSubscribers
			if removed {
				log.Printf("Agent %s removed %s as subscriber for type %v", a.ID, msg.SenderID, msgType)
			} else {
				log.Printf("Agent %s: %s was not subscribed to type %v", a.ID, msg.SenderID, msgType)
			}
		}
		// Optionally send a confirmation response
	} else {
		log.Printf("Agent %s received invalid payload for UnsubscribeEvents from %s", a.ID, msg.SenderID)
	}
}

// --- Example Usage ---

func main() {
	log.Println("Starting AI Agent System example")

	// Set up root context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())

	// 1. Create Message Bus
	messageBus := NewMessageBus()
	messageBus.wg.Add(1)
	go messageBus.Run(ctx) // Run the bus management loop

	// 2. Create Agents
	agentA := NewAgent("AgentA", messageBus)
	agentB := NewAgent("AgentB", messageBus)
	agentC := NewAgent("AgentC", messageBus)

	// Initialize some state/parameters/knowledge for agents
	agentA.State["task"] = "idle"
	agentA.Parameters["collaboration_mode"] = true
	agentA.Parameters["summary_length"] = 30
	agentA.KnowledgeBase["known_fact: Go is fun"] = true

	agentB.State["data_queue"] = []string{}
	agentB.Parameters["categorization_rule"] = "long_text"
	agentB.Parameters["simulation_growth_rate"] = 0.1
	agentB.State["needs_resourceX"] = 5.0 // AgentB needs 5 units of resourceX

	agentC.State["resourceX"] = 100.0 // AgentC has 100 units of resourceX
	agentC.Parameters["risk_tolerance"] = 0.5

	// Add agents to the bus (Done in NewAgent constructor)

	// 3. Start Agent Goroutines
	agentA.wg.Add(1)
	go agentA.Run()

	agentB.wg.Add(1)
	go agentB.Run()

	agentC.wg.Add(1)
	go agentC.Run()

	// Give agents a moment to start
	time.Sleep(100 * time.Millisecond)

	// 4. Send some initial messages to demonstrate functions
	log.Println("\n--- Sending initial messages ---")

	// AgentA requests state of AgentB
	err := messageBus.SendMessage(Message{
		SenderID:    "AgentA",
		RecipientID: "AgentB",
		Type:        MessageTypeRequestState,
		Payload:     nil,
	})
	if err != nil { log.Println(err) }

	// AgentB processes some data
	err = messageBus.SendMessage(Message{
		SenderID:    "ExternalSource", // Message can be from non-agents too, if bus supports it
		RecipientID: "AgentB",
		Type:        MessageTypeProcessDataChunk,
		Payload:     "This is a sample text data chunk for AgentB to process. It is quite long.",
	})
	if err != nil { log.Println(err) }

	// AgentC offers a resource (broadcast)
	err = messageBus.SendMessage(Message{
		SenderID:    "AgentC",
		RecipientID: "all", // Broadcast
		Type:        MessageTypeOfferResource,
		Payload:     map[string]interface{}{"resource_type": "resourceX", "amount": 50.0},
	})
	if err != nil { log.Println(err) }

	// AgentB requests resource from AgentC (reacting to offer)
	// This message would likely be sent *after* AgentB processes the OfferResource message.
	// For demonstration, simulate AgentB sending it now.
	err = messageBus.SendMessage(Message{
		SenderID:    "AgentB",
		RecipientID: "AgentC",
		Type:        MessageTypeRequestResource,
		Payload:     map[string]interface{}{"request_id": "b_needs_x_1", "resource_type": "resourceX", "amount": 5.0},
	})
	if err != nil { log.Println(err) }


	// AgentA requests AgentB to categorize data
	err = messageBus.SendMessage(Message{
		SenderID:    "AgentA",
		RecipientID: "AgentB",
		Type:        MessageTypeCategorizeData,
		Payload:     "Short text.",
	})
	if err != nil { log.Println(err) }

	err = messageBus.SendMessage(Message{
		SenderID:    "AgentA",
		RecipientID: "AgentB",
		Type:        MessageTypeCategorizeData,
		Payload:     "This is a moderately long piece of text that might be categorized differently.",
	})
	if err != nil { log.Println(err) }


	// AgentA requests AgentC to negotiate a parameter
	err = messageBus.SendMessage(Message{
		SenderID:    "AgentA",
		RecipientID: "AgentC",
		Type:        MessageTypeNegotiateParameter,
		Payload:     map[string]interface{}{"parameter": "risk_tolerance", "proposal": 0.6}, // AgentA proposes 0.6
	})
	if err != nil { log.Println(err) }

	// AgentA requests AgentB to simulate a scenario
	err = messageBus.SendMessage(Message{
		SenderID:    "AgentA",
		RecipientID: "AgentB",
		Type:        MessageTypeSimulateScenario,
		Payload:     map[string]interface{}{"initial_value": 100.0, "steps": 5},
	})
	if err != nil { log.Println(err) }

	// AgentA requests AgentA to prioritize its own tasks (self-message example)
	err = messageBus.SendMessage(Message{
		SenderID:    "AgentA",
		RecipientID: "AgentA",
		Type:        MessageTypePrioritizeTask,
		Payload:     "by_deadline",
	})
	if err != nil { log.Println(err) }


	// AgentB subscribes to AgentA's synthesized reports
	err = messageBus.SendMessage(Message{
		SenderID:    "AgentB",
		RecipientID: "AgentA",
		Type:        MessageTypeSubscribeEvents,
		Payload:     []MessageType{MessageTypeReportSynthesized},
	})
	if err != nil { log.Println(err) }

	// AgentA synthesizes a report (which should trigger a notification to AgentB now)
	err = messageBus.SendMessage(Message{
		SenderID:    "ExternalTrigger",
		RecipientID: "AgentA",
		Type:        MessageTypeSynthesizeReport,
		Payload:     nil,
	})
	if err != nil { log.Println(err) }


	// Let the system run for a bit to process messages
	log.Println("\n--- Letting system run for a few seconds ---")
	time.Sleep(3 * time.Second)

	log.Println("\n--- Sending Terminate message to AgentB ---")
	err = messageBus.SendMessage(Message{
		SenderID: "AgentA", // AgentA initiates termination of B
		RecipientID: "AgentB",
		Type: MessageTypeTerminateAgent,
		Payload: "Task completed",
	})
	if err != nil { log.Println(err) }

	// Give AgentB a moment to process termination
	time.Sleep(500 * time.Millisecond)


	// 5. Graceful Shutdown
	log.Println("\n--- Initiating graceful shutdown ---")

	// Signal all agents to stop via context
	cancel()

	// Wait for all agents to finish their Run loops
	log.Println("Waiting for agents to stop...")
	agentA.wg.Wait()
	agentB.wg.Wait() // This might already be done if termination message worked
	agentC.wg.Wait()
	log.Println("All agents stopped.")

	// Stop the Message Bus
	messageBus.Stop()

	log.Println("AI Agent System example finished.")
}
```

---

**Explanation:**

1.  **Message Structure:** The `Message` struct is the core of the MCP. It defines sender, recipient, message type, and a flexible payload.
2.  **Message Types:** The `MessageType` iota provides a clear enumeration for all possible interactions between agents, directly mapping to the "functions" the agents can perform.
3.  **MessageBus:** The `MessageBus` acts as the central router. Agents register themselves (`RegisterAgent`) and send messages to it (`SendMessage`). The bus looks up the recipient's channel and delivers the message. It supports direct agent-to-agent messages and broadcasts (`"all"`). Its `Run` method is minimal here but could handle system-level tasks like agent spawning or monitoring.
4.  **Agent Structure:** The `Agent` struct holds the agent's identity, its input channel (`InboundMessages`), internal state, configurable parameters, a knowledge base (simple maps for demonstration), a reference to the `MessageBus`, event subscriptions, and a context for cancellation.
5.  **Agent `Run` Loop:** The `Run` method is where an agent's life happens. It listens on its `InboundMessages` channel using a `select`. It also listens on the `Ctx.Done()` channel for a shutdown signal. When a message arrives, it calls `handleMessage`.
6.  **`handleMessage`:** This method is the central dispatch for an agent. It takes an incoming `Message`, uses a `switch` statement on `msg.Type` to determine what function is being requested, and calls the corresponding internal `handle...` method. It also includes a basic event notification mechanism for subscribers.
7.  **Agent Function Handlers (`handle...` methods):** Each `handle...` method implements the logic for a specific `MessageType`.
    *   They access the message payload (`msg.Payload`).
    *   They read and potentially modify the agent's internal state (`a.State`), parameters (`a.Parameters`), or knowledge base (`a.KnowledgeBase`).
    *   They perform the "AI" or processing task (simplified in this example).
    *   They often send a response message back to the sender using `a.MessageBus.SendMessage`.
    *   Some handlers include basic logic demonstrating the concept (e.g., simple categorization based on length, checking a parameter for negotiation).
8.  **Event Subscription:** A simple mechanism is included where an agent (AgentA) can request another agent (AgentB) to notify it of specific events (messages AgentB sends). AgentB stores these subscriptions and uses `notifySubscribers` in its `handleMessage` to send copies of relevant messages. This avoids a central event manager.
9.  **Graceful Shutdown:** The `context` package is used for graceful shutdown. The main function cancels the root context, which propagates to agent contexts, causing their `Run` loops to exit cleanly after processing any remaining messages or finishing current tasks. The `sync.WaitGroup` ensures the main function waits for goroutines to finish.
10. **Example Usage (`main`):** Sets up the bus and a few agents, initializes some state, starts their goroutines, sends a sequence of messages to trigger various functions, pauses to allow processing, and then initiates shutdown.

This system provides a robust foundation for building complex multi-agent systems in Go where agents communicate and coordinate exclusively through well-defined messages. The "AI" aspect is represented by the *types* of sophisticated tasks the agents are designed to handle via these messages, even if the internal logic of the handlers is simplified for demonstration.