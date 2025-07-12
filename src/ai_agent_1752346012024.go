Okay, here is a design and implementation for an AI Agent in Go with an MCP (Master Control Program) style interface. The goal is to create functions that feel distinct, conceptually interesting, and are implemented using core Go capabilities rather than relying on standard external AI/ML libraries, thus avoiding direct duplication of existing open source models or frameworks. The "AI" aspect comes from state management, rule application, pattern detection, simple simulation, and internal reasoning/introspection concepts, all implemented with basic logic.

**Outline and Function Summary:**

```go
/*
AI Agent with MCP Interface

Outline:
1.  Agent State Definition: Structure to hold the agent's internal state (knowledge, metrics, parameters, history, etc.).
2.  Agent Core Structure: The main AIAgent struct holding the state.
3.  MCP Interface: The central ProcessCommand function responsible for parsing and dispatching commands.
4.  Core Agent Functions: Initialization, state management (get/set).
5.  Processing Functions: Rule application, simple pattern matching, data analysis (basic).
6.  Simulation & Prediction Functions: Simple state projection, scenario testing.
7.  Knowledge & Memory Functions: Fact storage, retrieval, association, decay.
8.  Self-Management Functions: Monitoring, prioritization (internal), adaptation (simple parameter adjustment), logging.
9.  Interaction & Communication Functions: Response generation (rule-based), intent identification (simple), clarification.
10. Advanced/Creative Functions: Conceptual blending (rule-based), analogy (simple), constraint negotiation (priority), introspection report, simulated emotion update, anticipation, abstract entity modeling, simple sequence learning, hypothetical generation, context drift detection, preference learning, simulated delegation.

Function Summary (Approx. 30 functions):

1.  Initialize(): Sets up the initial state of the agent.
2.  ProcessCommand(command string): The core MCP interface. Parses command string and dispatches to appropriate internal function. Returns result string and error.
3.  GetState(key string): Retrieves a specific value or sub-structure from the agent's state.
4.  SetStateValue(key string, value interface{}): Sets or updates a specific value in the agent's state.
5.  ApplyRule(ruleName string, context map[string]string): Applies a predefined simple rule based on context, potentially modifying state or returning a value.
6.  FindKeywordPattern(text string, pattern string): Checks if a simple keyword pattern exists in text.
7.  CheckInternalCondition(conditionName string, params map[string]interface{}): Evaluates a predefined internal condition based on state and parameters.
8.  LogEvent(eventType string, message string, data map[string]interface{}): Records an event in the agent's history/logs.
9.  RetrieveHistory(filter map[string]string): Retrieves logged events or command history.
10. StoreKnowledgeFact(category string, key string, value string, tags []string): Stores a piece of information in the knowledge base.
11. RetrieveKnowledgeFact(category string, key string, tags []string): Retrieves information from the knowledge base.
12. AssociateKnowledge(key1 string, key2 string, relationship string): Creates a simple link between two knowledge facts.
13. DecayKnowledge(category string, key string): Marks knowledge for decay (e.g., reduces retrieval priority or removes).
14. PredictStateTransition(input map[string]interface{}, rules map[string]string): Predicts a simple next state based on current input and defined transition rules.
15. RunSimulatedScenario(scenarioName string, initialConditions map[string]interface{}): Runs a simple simulation based on internal rules and initial conditions, reporting outcome.
16. PrioritizeTaskQueue(taskID string, priority int): Modifies the priority of a task in an internal queue representation.
17. GetNextTask(): Retrieves the highest priority task from the internal queue.
18. GenerateRuleBasedResponse(context map[string]interface{}): Generates a response string based on current state and context using simple predefined rules.
19. IdentifySimpleIntent(input string): Maps input text to a predefined simple intent based on keywords/phrases.
20. RequestClarification(uncertaintyKey string): Indicates uncertainty and requests more specific input related to a key concept.
21. MonitorInternalMetric(metricName string): Reports the current value of a simulated internal metric (e.g., 'energy', 'load').
22. AdjustParameterRule(parameterName string, adjustmentRule string): Modifies an internal operational parameter based on a rule (e.g., increase processing speed if 'load' is low).
23. BlendSimpleConcepts(conceptA string, conceptB string, blendRules map[string]string): Combines two stored conceptual representations based on simple rules to form a new concept representation.
24. GenerateSimpleAnalogy(sourceConcept string, targetDomain string, mappingRules map[string]string): Finds a simple analogy mapping attributes from a source concept to a target domain based on rules.
25. NegotiateBasicConstraints(constraints []map[string]string): Resolves conflicting constraints (e.g., time vs. resource) based on a predefined priority hierarchy or simple scoring.
26. ReportLastActionRationale(): Provides a simple trace or explanation of the steps taken during the last command processing.
27. UpdateEmotionalState(eventImpact map[string]float64): Adjusts a simulated internal 'emotional' metric based on the perceived impact of an event.
28. AnticipateNextInput(historyWindow int): Based on recent command history, predicts a likely next command or topic.
29. ModelAbstractEntityState(entityID string, stateUpdate map[string]interface{}): Updates the internal simulated state of an abstract entity the agent is tracking.
30. LearnSimpleSequence(sequence []string, outcome map[string]interface{}): Records a sequence of commands/events and associates it with a resulting state change or outcome for future reference/prediction.

(Note: Implementations are simplified to demonstrate concepts without relying on complex external libraries.)
*/
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// --- 1. Agent State Definition ---

// AgentState holds all internal data for the AI Agent.
type AgentState struct {
	sync.RWMutex // For thread-safe access if used concurrently

	KnowledgeBase map[string]map[string]string   // category -> key -> value
	KnowledgeTags map[string][]string            // key -> tags
	KnowledgeLinks map[string]map[string]string // key1 -> key2 -> relationship

	InternalMetrics map[string]float64 // e.g., "energy", "load", "confidence", "simulated_mood"
	Parameters      map[string]string  // Operational parameters
	TaskQueue       []string           // Simplified task queue (just IDs)
	TaskPriorities  map[string]int     // Task ID -> Priority

	CommandHistory []string             // Recent commands received
	EventLog       []map[string]interface{} // Log of internal events

	AbstractEntityStates map[string]map[string]interface{} // id -> state

	SimpleSequences map[string]map[string]interface{} // Joined sequence string -> outcome/metadata
	Preferences map[string]string // Simple key-value preferences

	LastAction struct {
		Command string
		Result  string
		Error   string
		Time    time.Time
		Rationale string // Simple explanation
	}
}

// --- 2. Agent Core Structure ---

// AIAgent represents the AI agent with its state and methods.
type AIAgent struct {
	State *AgentState
}

// NewAIAgent creates and initializes a new agent.
func NewAIAgent() *AIAgent {
	agent := &AIAgent{
		State: &AgentState{
			KnowledgeBase:        make(map[string]map[string]string),
			KnowledgeTags:        make(map[string][]string),
			KnowledgeLinks:       make(map[string]map[string]string),
			InternalMetrics:      make(map[string]float64),
			Parameters:           make(map[string]string),
			TaskPriorities:       make(map[string]int),
			AbstractEntityStates: make(map[string]map[string]interface{}),
			SimpleSequences:      make(map[string]map[string]interface{}),
			Preferences:          make(map[string]string),
		},
	}
	agent.Initialize() // Call the initialization function
	return agent
}

// --- Core Agent Functions ---

// 1. Initialize(): Sets up the initial state of the agent.
func (a *AIAgent) Initialize() {
	a.State.Lock()
	defer a.State.Unlock()

	// Set initial metrics
	a.State.InternalMetrics["energy"] = 100.0
	a.State.InternalMetrics["load"] = 0.0
	a.State.InternalMetrics["confidence"] = 0.8
	a.State.InternalMetrics["simulated_mood"] = 0.5 // 0 to 1 scale

	// Set default parameters
	a.State.Parameters["processing_speed"] = "normal"
	a.State.Parameters["log_level"] = "info"

	// Initialize internal structures if needed (maps/slices already done in NewAIAgent)

	a.LogEvent("system", "Agent initialized", nil)
}

// --- 3. MCP Interface ---

// Command structure for parsing
type Command struct {
	Name string
	Params map[string]string
}

// ProcessCommand(command string): The core MCP interface.
// Parses command string and dispatches to appropriate internal function.
// Command format: COMMAND_NAME param1=value1 param2="value 2" ...
func (a *AIAgent) ProcessCommand(commandStr string) (string, error) {
	a.State.Lock() // Lock state during command processing
	defer a.State.Unlock()

	a.State.CommandHistory = append(a.State.CommandHistory, commandStr) // Log command history

	// Simple parsing: Split command name and parameters
	parts := strings.Fields(commandStr)
	if len(parts) == 0 {
		a.State.LastAction = struct { Command string; Result string; Error string; Time time.Time; Rationale string }{commandStr, "", "Empty command", time.Now(), "No input received."}
		return "", fmt.Errorf("empty command")
	}

	cmd := Command{
		Name: parts[0],
		Params: make(map[string]string),
	}

	// Parse parameters (basic key=value or key="value with spaces")
	paramRegex := regexp.MustCompile(`(\w+)=(?:"([^"]*)"|([^ ]+))`)
	paramsStr := strings.Join(parts[1:], " ")
	matches := paramRegex.FindAllStringSubmatch(paramsStr, -1)

	for _, match := range matches {
		key := match[1]
		var value string
		if match[2] != "" {
			value = match[2] // Quoted value
		} else {
			value = match[3] // Unquoted value
		}
		cmd.Params[key] = value
	}

	var result string
	var err error
	var rationale = "Dispatched command."

	// Dispatch based on command name (using a switch for simplicity)
	switch cmd.Name {
	case "GET_STATE":
		key := cmd.Params["key"]
		result, err = a.handleGetState(key)
		rationale = fmt.Sprintf("Retrieved state key '%s'.", key)
	case "SET_STATE_VALUE":
		key := cmd.Params["key"]
		valueStr := cmd.Params["value"] // Note: value is always a string from parsing
		err = a.handleSetStateValue(key, valueStr) // Needs conversion inside handler
		if err == nil {
			result = "State updated."
		}
		rationale = fmt.Sprintf("Attempted to set state key '%s'.", key)
	case "APPLY_RULE":
		ruleName := cmd.Params["ruleName"]
		// Need a way to pass complex context, simple map[string]string for now
		contextJSON := cmd.Params["context"] // Expecting JSON string for context
		context := make(map[string]string)
		if contextJSON != "" {
			if jsonErr := json.Unmarshal([]byte(contextJSON), &context); jsonErr != nil {
				err = fmt.Errorf("invalid context JSON: %w", jsonErr)
				break // Exit switch on parsing error
			}
		}
		result, err = a.handleApplyRule(ruleName, context)
		rationale = fmt.Sprintf("Applied rule '%s'.", ruleName)
	case "FIND_KEYWORD_PATTERN":
		text := cmd.Params["text"]
		pattern := cmd.Params["pattern"]
		result, err = a.handleFindKeywordPattern(text, pattern)
		rationale = fmt.Sprintf("Searched for pattern '%s' in text.", pattern)
	case "CHECK_INTERNAL_CONDITION":
		conditionName := cmd.Params["conditionName"]
		// Similar to APPLY_RULE, need context handling
		paramsJSON := cmd.Params["params"]
		params := make(map[string]interface{}) // Can handle different types inside handler
		if paramsJSON != "" {
			if jsonErr := json.Unmarshal([]byte(paramsJSON), &params); jsonErr != nil {
				err = fmt.Errorf("invalid params JSON: %w", jsonErr)
				break
			}
		}
		boolResult, checkErr := a.handleCheckInternalCondition(conditionName, params)
		if checkErr != nil {
			err = checkErr
		} else {
			result = fmt.Sprintf("%v", boolResult)
		}
		rationale = fmt.Sprintf("Checked condition '%s'.", conditionName)

	case "LOG_EVENT":
		eventType := cmd.Params["eventType"]
		message := cmd.Params["message"]
		dataJSON := cmd.Params["data"]
		data := make(map[string]interface{})
		if dataJSON != "" {
			if jsonErr := json.Unmarshal([]byte(dataJSON), &data); jsonErr != nil {
				err = fmt.Errorf("invalid data JSON: %w", jsonErr)
				break
			}
		}
		a.LogEvent(eventType, message, data) // LogEvent doesn't return error typically
		result = "Event logged."
		rationale = "Logged a new event."

	case "RETRIEVE_HISTORY":
		filterJSON := cmd.Params["filter"]
		filter := make(map[string]string)
		if filterJSON != "" {
			if jsonErr := json.Unmarshal([]byte(filterJSON), &filter); jsonErr != nil {
				err = fmt.Errorf("invalid filter JSON: %w", jsonErr)
				break
			}
		}
		history, histErr := a.RetrieveHistory(filter)
		if histErr != nil {
			err = histErr
		} else {
			historyBytes, _ := json.Marshal(history) // Convert result to JSON string
			result = string(historyBytes)
		}
		rationale = "Retrieved history."

	case "STORE_KNOWLEDGE_FACT":
		category := cmd.Params["category"]
		key := cmd.Params["key"]
		value := cmd.Params["value"]
		tagsStr := cmd.Params["tags"] // Comma-separated tags
		tags := []string{}
		if tagsStr != "" {
			tags = strings.Split(tagsStr, ",")
		}
		err = a.StoreKnowledgeFact(category, key, value, tags)
		if err == nil {
			result = "Fact stored."
		}
		rationale = fmt.Sprintf("Stored fact '%s' in '%s'.", key, category)

	case "RETRIEVE_KNOWLEDGE_FACT":
		category := cmd.Params["category"]
		key := cmd.Params["key"]
		tagsStr := cmd.Params["tags"]
		tags := []string{}
		if tagsStr != "" {
			tags = strings.Split(tagsStr, ",")
		}
		value, retrieveErr := a.RetrieveKnowledgeFact(category, key, tags)
		if retrieveErr != nil {
			err = retrieveErr
		} else {
			result = value
		}
		rationale = fmt.Sprintf("Retrieved fact '%s' from '%s'.", key, category)

	case "ASSOCIATE_KNOWLEDGE":
		key1 := cmd.Params["key1"]
		key2 := cmd.Params["key2"]
		relationship := cmd.Params["relationship"]
		err = a.AssociateKnowledge(key1, key2, relationship)
		if err == nil {
			result = "Knowledge associated."
		}
		rationale = fmt.Sprintf("Associated '%s' and '%s' with relation '%s'.", key1, key2, relationship)

	case "DECAY_KNOWLEDGE":
		category := cmd.Params["category"]
		key := cmd.Params["key"]
		err = a.DecayKnowledge(category, key)
		if err == nil {
			result = "Knowledge marked for decay."
		}
		rationale = fmt.Sprintf("Marked fact '%s' from '%s' for decay.", key, category)

	case "PREDICT_STATE_TRANSITION":
		inputJSON := cmd.Params["input"]
		input := make(map[string]interface{})
		if inputJSON != "" {
			if jsonErr := json.Unmarshal([]byte(inputJSON), &input); jsonErr != nil {
				err = fmt.Errorf("invalid input JSON: %w", jsonErr)
				break
			}
		}
		rulesJSON := cmd.Params["rules"] // Simple rules map? Or rule name? Let's assume rule name for simplicity
		ruleName := rulesJSON // Treat rulesJSON as rule name
		predictedState, predictErr := a.PredictStateTransition(input, map[string]string{"ruleName": ruleName}) // Pass ruleName in a map
		if predictErr != nil {
			err = predictErr
		} else {
			stateBytes, _ := json.Marshal(predictedState)
			result = string(stateBytes)
		}
		rationale = "Predicted next state."

	case "RUN_SIMULATED_SCENARIO":
		scenarioName := cmd.Params["scenarioName"]
		initialConditionsJSON := cmd.Params["initialConditions"]
		initialConditions := make(map[string]interface{})
		if initialConditionsJSON != "" {
			if jsonErr := json.Unmarshal([]byte(initialConditionsJSON), &initialConditions); jsonErr != nil {
				err = fmt.Errorf("invalid initialConditions JSON: %w", jsonErr)
				break
			}
		}
		outcome, scenarioErr := a.RunSimulatedScenario(scenarioName, initialConditions)
		if scenarioErr != nil {
			err = scenarioErr
		} else {
			outcomeBytes, _ := json.Marshal(outcome)
			result = string(outcomeBytes)
		}
		rationale = fmt.Sprintf("Ran scenario '%s'.", scenarioName)

	case "PRIORITIZE_TASK_QUEUE":
		taskID := cmd.Params["taskID"]
		priorityStr := cmd.Params["priority"]
		priority, pErr := strconv.Atoi(priorityStr)
		if pErr != nil {
			err = fmt.Errorf("invalid priority: %w", pErr)
			break
		}
		err = a.PrioritizeTaskQueue(taskID, priority)
		if err == nil {
			result = "Task priority updated."
		}
		rationale = fmt.Sprintf("Set priority for task '%s'.", taskID)

	case "GET_NEXT_TASK":
		taskID, taskErr := a.GetNextTask()
		if taskErr != nil {
			err = taskErr
		} else {
			result = taskID
		}
		rationale = "Retrieved next task from queue."

	case "GENERATE_RULE_BASED_RESPONSE":
		contextJSON := cmd.Params["context"]
		context := make(map[string]interface{})
		if contextJSON != "" {
			if jsonErr := json.Unmarshal([]byte(contextJSON), &context); jsonErr != nil {
				err = fmt.Errorf("invalid context JSON: %w", jsonErr)
				break
			}
		}
		response, respErr := a.GenerateRuleBasedResponse(context)
		if respErr != nil {
			err = respErr
		} else {
			result = response
		}
		rationale = "Generated rule-based response."

	case "IDENTIFY_SIMPLE_INTENT":
		input := cmd.Params["input"]
		intent, intentErr := a.IdentifySimpleIntent(input)
		if intentErr != nil {
			err = intentErr
		} else {
			result = intent
		}
		rationale = fmt.Sprintf("Identified simple intent from input '%s'.", input)

	case "REQUEST_CLARIFICATION":
		uncertaintyKey := cmd.Params["uncertaintyKey"]
		response, clarifyErr := a.RequestClarification(uncertaintyKey)
		if clarifyErr != nil {
			err = clarifyErr
		} else {
			result = response
		}
		rationale = fmt.Sprintf("Requested clarification for key '%s'.", uncertaintyKey)

	case "MONITOR_INTERNAL_METRIC":
		metricName := cmd.Params["metricName"]
		value, metricErr := a.MonitorInternalMetric(metricName)
		if metricErr != nil {
			err = metricErr
		} else {
			result = fmt.Sprintf("%f", value)
		}
		rationale = fmt.Sprintf("Monitored metric '%s'.", metricName)

	case "ADJUST_PARAMETER_RULE":
		parameterName := cmd.Params["parameterName"]
		adjustmentRule := cmd.Params["adjustmentRule"]
		err = a.AdjustParameterRule(parameterName, adjustmentRule)
		if err == nil {
			result = "Parameter adjustment rule applied."
		}
		rationale = fmt.Sprintf("Applied adjustment rule '%s' to parameter '%s'.", adjustmentRule, parameterName)

	case "BLEND_SIMPLE_CONCEPTS":
		conceptA := cmd.Params["conceptA"]
		conceptB := cmd.Params["conceptB"]
		rulesJSON := cmd.Params["blendRules"] // Rules for blending
		blendRules := make(map[string]string)
		if rulesJSON != "" {
			if jsonErr := json.Unmarshal([]byte(rulesJSON), &blendRules); jsonErr != nil {
				err = fmt.Errorf("invalid blendRules JSON: %w", jsonErr)
				break
			}
		}
		blendedConcept, blendErr := a.BlendSimpleConcepts(conceptA, conceptB, blendRules)
		if blendErr != nil {
			err = blendErr
		} else {
			conceptBytes, _ := json.Marshal(blendedConcept)
			result = string(conceptBytes)
		}
		rationale = fmt.Sprintf("Blended concepts '%s' and '%s'.", conceptA, conceptB)

	case "GENERATE_SIMPLE_ANALOGY":
		sourceConcept := cmd.Params["sourceConcept"]
		targetDomain := cmd.Params["targetDomain"]
		rulesJSON := cmd.Params["mappingRules"] // Rules for mapping
		mappingRules := make(map[string]string)
		if rulesJSON != "" {
			if jsonErr := json.Unmarshal([]byte(rulesJSON), &mappingRules); jsonErr != nil {
				err = fmt.Errorf("invalid mappingRules JSON: %w", jsonErr)
				break
			}
		}
		analogy, analogyErr := a.GenerateSimpleAnalogy(sourceConcept, targetDomain, mappingRules)
		if analogyErr != nil {
			err = analogyErr
		} else {
			analogyBytes, _ := json.Marshal(analogy)
			result = string(analogyBytes)
		}
		rationale = fmt.Sprintf("Generated analogy from '%s' to '%s'.", sourceConcept, targetDomain)

	case "NEGOTIATE_BASIC_CONSTRAINTS":
		constraintsJSON := cmd.Params["constraints"]
		constraints := []map[string]string{}
		if constraintsJSON != "" {
			if jsonErr := json.Unmarshal([]byte(constraintsJSON), &constraints); jsonErr != nil {
				err = fmt.Errorf("invalid constraints JSON: %w", jsonErr)
				break
			}
		}
		negotiatedOutcome, negotiateErr := a.NegotiateBasicConstraints(constraints)
		if negotiateErr != nil {
			err = negotiateErr
		} else {
			outcomeBytes, _ := json.Marshal(negotiatedOutcome)
			result = string(outcomeBytes)
		}
		rationale = "Negotiated constraints."

	case "REPORT_LAST_ACTION_RATIONALE":
		result = a.ReportLastActionRationale()
		rationale = "Reported on the last action's rationale."

	case "UPDATE_EMOTIONAL_STATE":
		eventImpactJSON := cmd.Params["eventImpact"]
		eventImpact := make(map[string]float64)
		if eventImpactJSON != "" {
			if jsonErr := json.Unmarshal([]byte(eventImpactJSON), &eventImpact); jsonErr != nil {
				err = fmt.Errorf("invalid eventImpact JSON: %w", jsonErr)
				break
			}
		}
		a.UpdateEmotionalState(eventImpact) // This function updates state directly
		result = "Emotional state updated."
		rationale = "Updated simulated emotional state."

	case "ANTICIPATE_NEXT_INPUT":
		historyWindowStr := cmd.Params["historyWindow"]
		historyWindow, wErr := strconv.Atoi(historyWindowStr)
		if wErr != nil {
			err = fmt.Errorf("invalid historyWindow: %w", wErr)
			break
		}
		predicted, anticipateErr := a.AnticipateNextInput(historyWindow)
		if anticipateErr != nil {
			err = anticipateErr
		} else {
			result = predicted
		}
		rationale = "Anticipated next input."

	case "MODEL_ABSTRACT_ENTITY_STATE":
		entityID := cmd.Params["entityID"]
		stateUpdateJSON := cmd.Params["stateUpdate"]
		stateUpdate := make(map[string]interface{})
		if stateUpdateJSON != "" {
			if jsonErr := json.Unmarshal([]byte(stateUpdateJSON), &stateUpdate); jsonErr != nil {
				err = fmt.Errorf("invalid stateUpdate JSON: %w", jsonErr)
				break
			}
		}
		err = a.ModelAbstractEntityState(entityID, stateUpdate)
		if err == nil {
			result = fmt.Sprintf("Entity '%s' state updated.", entityID)
		}
		rationale = fmt.Sprintf("Updated model for entity '%s'.", entityID)

	case "LEARN_SIMPLE_SEQUENCE":
		sequenceStr := cmd.Params["sequence"] // Comma-separated sequence steps
		sequence := strings.Split(sequenceStr, ",")
		outcomeJSON := cmd.Params["outcome"]
		outcome := make(map[string]interface{})
		if outcomeJSON != "" {
			if jsonErr := json.Unmarshal([]byte(outcomeJSON), &outcome); jsonErr != nil {
				err = fmt.Errorf("invalid outcome JSON: %w", jsonErr)
				break
			}
		}
		a.LearnSimpleSequence(sequence, outcome) // This function just records
		result = "Sequence learned."
		rationale = "Recorded a simple sequence."

	// --- Add more cases for other functions ---

	default:
		err = fmt.Errorf("unknown command: %s", cmd.Name)
		rationale = "Failed to find command handler."
	}

	// Store result and error in LastAction history
	a.State.LastAction = struct { Command string; Result string; Error string; Time time.Time; Rationale string }{
		Command: commandStr,
		Result:  result,
		Error:   fmt.Sprintf("%v", err), // Store error as string
		Time:    time.Now(),
		Rationale: rationale,
	}

	// Basic logging of process result
	if err != nil {
		log.Printf("Command failed: %s, Error: %v", commandStr, err)
	} else {
		log.Printf("Command processed: %s, Result: %s", commandStr, result)
	}

	return result, err
}

// --- 4. Core Agent Functions (Handlers for ProcessCommand) ---

// 3. GetState(key string): Retrieves a specific value or sub-structure from the agent's state.
func (a *AIAgent) handleGetState(key string) (string, error) {
	// Accessing state requires the lock held by ProcessCommand
	switch key {
	case "knowledge":
		data, _ := json.Marshal(a.State.KnowledgeBase)
		return string(data), nil
	case "metrics":
		data, _ := json.Marshal(a.State.InternalMetrics)
		return string(data), nil
	case "parameters":
		data, _ := json.Marshal(a.State.Parameters)
		return string(data), nil
	case "task_queue":
		data, _ := json.Marshal(a.State.TaskQueue)
		return string(data), nil
	case "task_priorities":
		data, _ := json.Marshal(a.State.TaskPriorities)
		return string(data), nil
	case "command_history":
		data, _ := json.Marshal(a.State.CommandHistory)
		return string(data), nil
	case "event_log":
		data, _ := json.Marshal(a.State.EventLog)
		return string(data), nil
	case "entity_states":
		data, _ := json.Marshal(a.State.AbstractEntityStates)
		return string(data), nil
	case "sequences":
		data, _ := json.Marshal(a.State.SimpleSequences)
		return string(data), nil
	case "preferences":
		data, _ := json.Marshal(a.State.Preferences)
		return string(data), nil
	case "last_action":
		data, _ := json.Marshal(a.State.LastAction)
		return string(data), nil
	default:
		// Attempt to get a specific metric/parameter
		if metric, ok := a.State.InternalMetrics[key]; ok {
			return fmt.Sprintf("%f", metric), nil
		}
		if param, ok := a.State.Parameters[key]; ok {
			return param, nil
		}
		return "", fmt.Errorf("state key '%s' not found", key)
	}
}

// 4. SetStateValue(key string, value interface{}): Sets or updates a specific value in the agent's state.
// Value comes in as string from ProcessCommand, needs type handling here.
func (a *AIAgent) handleSetStateValue(key string, valueStr string) error {
	// Accessing state requires the lock held by ProcessCommand

	// Simple type inference/casting based on key or format
	switch key {
	// Handle metrics (expect float)
	case "energy", "load", "confidence", "simulated_mood":
		val, err := strconv.ParseFloat(valueStr, 64)
		if err != nil {
			return fmt.Errorf("invalid float value for metric '%s': %w", key, err)
		}
		a.State.InternalMetrics[key] = val
		a.LogEvent("state_change", fmt.Sprintf("Metric '%s' set", key), map[string]interface{}{key: val})
		return nil
	// Handle parameters (expect string)
	case "processing_speed", "log_level":
		a.State.Parameters[key] = valueStr
		a.LogEvent("state_change", fmt.Sprintf("Parameter '%s' set", key), map[string]interface{}{key: valueStr})
		return nil
	// Add cases for other top-level keys if simple set is allowed (e.g., replacing whole maps - use with caution)
	default:
		// Could potentially support setting sub-keys with a more complex key format (e.g., "metrics.energy")
		// For now, only top-level simple values/maps are handled explicitly.
		return fmt.Errorf("setting state key '%s' not supported directly", key)
	}
}

// --- 5. Processing Functions ---

// 5. ApplyRule(ruleName string, context map[string]string): Applies a predefined simple rule based on context.
func (a *AIAgent) handleApplyRule(ruleName string, context map[string]string) (string, error) {
	// Accessing state requires the lock held by ProcessCommand
	// This is where rule logic lives. Rules are hardcoded here for simplicity.
	// In a real system, rules might be loaded from config/KB.

	switch ruleName {
	case "greet":
		name, ok := context["name"]
		if !ok {
			name = "user"
		}
		return fmt.Sprintf("Hello, %s!", name), nil
	case "check_load_alert":
		thresholdStr, ok := context["threshold"]
		if !ok {
			thresholdStr = "0.7" // Default threshold
		}
		threshold, err := strconv.ParseFloat(thresholdStr, 64)
		if err != nil {
			return "", fmt.Errorf("invalid threshold format: %w", err)
		}
		currentLoad := a.State.InternalMetrics["load"]
		if currentLoad > threshold {
			a.LogEvent("alert", "Load high", map[string]interface{}{"load": currentLoad, "threshold": threshold})
			return "ALERT: System load is high.", nil
		} else {
			return "System load is normal.", nil
		}
	// Add more rules here
	default:
		return "", fmt.Errorf("unknown rule: %s", ruleName)
	}
}

// 6. FindKeywordPattern(text string, pattern string): Checks if a simple keyword pattern exists in text.
// Pattern could be comma-separated keywords, order-sensitive phrase, etc.
func (a *AIAgent) handleFindKeywordPattern(text string, pattern string) (string, error) {
	// Accessing state requires the lock held by ProcessCommand
	// Simple implementation: check for all keywords in pattern (case-insensitive, any order)
	keywords := strings.Split(pattern, ",")
	textLower := strings.ToLower(text)
	foundAll := true
	missing := []string{}
	for _, keyword := range keywords {
		if !strings.Contains(textLower, strings.ToLower(strings.TrimSpace(keyword))) {
			foundAll = false
			missing = append(missing, strings.TrimSpace(keyword))
		}
	}

	if foundAll {
		return "Pattern found.", nil
	} else {
		return fmt.Sprintf("Pattern not fully matched. Missing keywords: %s", strings.Join(missing, ", ")), nil
	}
}

// 7. CheckInternalCondition(conditionName string, params map[string]interface{}): Evaluates a predefined internal condition.
func (a *AIAgent) handleCheckInternalCondition(conditionName string, params map[string]interface{}) (bool, error) {
	// Accessing state requires the lock held by ProcessCommand
	switch conditionName {
	case "is_energy_low":
		threshold, ok := params["threshold"].(float64)
		if !ok {
			threshold = 20.0 // Default low energy threshold
		}
		return a.State.InternalMetrics["energy"] < threshold, nil
	case "has_knowledge":
		category, ok := params["category"].(string)
		if !ok {
			return false, fmt.Errorf("missing 'category' parameter for condition '%s'", conditionName)
		}
		key, ok := params["key"].(string)
		if !ok {
			return false, fmt.Errorf("missing 'key' parameter for condition '%s'", conditionName)
		}
		_, err := a.RetrieveKnowledgeFact(category, key, nil) // Use the existing method, nil tags means any
		return err == nil, nil // True if retrieval successful
	// Add more conditions
	default:
		return false, fmt.Errorf("unknown condition: %s", conditionName)
	}
}

// --- 8. Self-Management Functions ---

// 8. LogEvent(eventType string, message string, data map[string]interface{}): Records an event.
// Called internally, doesn't typically return an error or result.
func (a *AIAgent) LogEvent(eventType string, message string, data map[string]interface{}) {
	// Accessing state requires the lock held by ProcessCommand
	event := map[string]interface{}{
		"timestamp": time.Now().Format(time.RFC3339),
		"type":      eventType,
		"message":   message,
		"data":      data,
	}
	a.State.EventLog = append(a.State.EventLog, event)
	// Keep log size manageable? Add limit logic here if needed.
	// log.Printf("EVENT [%s]: %s", eventType, message) // Optional: log to console too
}

// 9. RetrieveHistory(filter map[string]string): Retrieves logged events or command history based on filter.
func (a *AIAgent) RetrieveHistory(filter map[string]string) ([]map[string]interface{}, error) {
	// Accessing state requires the lock held by ProcessCommand
	// Simple filtering for now: match event type or message substring
	var filteredLog []map[string]interface{}
	filterType, typeOK := filter["type"]
	filterMessage, msgOK := filter["message_contains"]

	for _, event := range a.State.EventLog {
		match := true
		if typeOK {
			if eventType, ok := event["type"].(string); !ok || eventType != filterType {
				match = false
			}
		}
		if msgOK && match { // Only check message if type matched (or no type filter)
			if message, ok := event["message"].(string); !ok || !strings.Contains(message, filterMessage) {
				match = false
			}
		}
		// Add more filter criteria here (e.g., time range, data content)

		if match {
			filteredLog = append(filteredLog, event)
		}
	}

	// Could also retrieve command history based on filter
	// For simplicity, just returning filtered event log
	return filteredLog, nil
}

// 21. MonitorInternalMetric(metricName string): Reports the current value of a simulated internal metric.
func (a *AIAgent) MonitorInternalMetric(metricName string) (float64, error) {
	// Accessing state requires the lock held by ProcessCommand
	if value, ok := a.State.InternalMetrics[metricName]; ok {
		return value, nil
	}
	return 0, fmt.Errorf("metric '%s' not found", metricName)
}

// 22. AdjustParameterRule(parameterName string, adjustmentRule string): Modifies an internal operational parameter based on a rule.
func (a *AIAgent) AdjustParameterRule(parameterName string, adjustmentRule string) error {
	// Accessing state requires the lock held by ProcessCommand
	currentValue, ok := a.State.Parameters[parameterName]
	if !ok {
		return fmt.Errorf("parameter '%s' not found", parameterName)
	}

	// Hardcoded simple adjustment rules for simplicity
	switch adjustmentRule {
	case "toggle_speed":
		if currentValue == "normal" {
			a.State.Parameters[parameterName] = "high"
		} else {
			a.State.Parameters[parameterName] = "normal"
		}
		a.LogEvent("parameter_adjustment", fmt.Sprintf("Parameter '%s' toggled", parameterName), map[string]interface{}{"new_value": a.State.Parameters[parameterName]})
		return nil
	case "set_low_log_level":
		a.State.Parameters["log_level"] = "warn"
		a.LogEvent("parameter_adjustment", fmt.Sprintf("Parameter '%s' set to low", parameterName), map[string]interface{}{"new_value": a.State.Parameters["log_level"]})
		return nil
	// Add more rules
	default:
		return fmt.Errorf("unknown adjustment rule: %s", adjustmentRule)
	}
}

// 27. UpdateEmotionalState(eventImpact map[string]float64): Adjusts a simulated internal 'emotional' metric.
// Simplistic model: average of impacts scaled to 0-1, with decay towards 0.5 (neutral).
func (a *AIAgent) UpdateEmotionalState(eventImpact map[string]float64) {
	// Accessing state requires the lock held by ProcessCommand
	currentMood := a.State.InternalMetrics["simulated_mood"]
	decayRate := 0.05 // Tendency to return to neutral

	// Apply decay
	if currentMood > 0.5 {
		currentMood = currentMood - decayRate
		if currentMood < 0.5 {
			currentMood = 0.5
		}
	} else if currentMood < 0.5 {
		currentMood = currentMood + decayRate
		if currentMood > 0.5 {
			currentMood = 0.5
		}
	}

	// Apply impacts (average of positive/negative impacts)
	totalImpact := 0.0
	impactCount := 0
	for _, impact := range eventImpact {
		totalImpact += impact
		impactCount++
	}

	if impactCount > 0 {
		avgImpact := totalImpact / float64(impactCount)
		// Simple mapping: +1 impact moves mood towards 1, -1 impact moves mood towards 0
		currentMood += avgImpact * 0.1 // Scale impact effect

		// Clamp mood between 0 and 1
		if currentMood < 0 {
			currentMood = 0
		} else if currentMood > 1 {
			currentMood = 1
		}
	}

	a.State.InternalMetrics["simulated_mood"] = currentMood
	a.LogEvent("emotional_state_update", "Simulated mood updated", map[string]interface{}{"mood": currentMood, "impact": eventImpact})
}


// --- 6. Simulation & Prediction Functions ---

// 14. PredictStateTransition(input map[string]interface{}, rules map[string]string): Predicts a simple next state based on input and rules.
// Rules structure: map[currentStateProperty]=nextStateValue
func (a *AIAgent) PredictStateTransition(input map[string]interface{}, rules map[string]string) (map[string]interface{}, error) {
	// Accessing state requires the lock held by ProcessCommand
	predictedState := make(map[string]interface{})

	ruleName, ok := rules["ruleName"]
	if !ok {
		return nil, fmt.Errorf("ruleName not provided for prediction")
	}

	// Hardcoded simple prediction rules
	switch ruleName {
	case "energy_decay_predict":
		// Predict energy decay over timeSteps based on load
		timeSteps, ok := input["timeSteps"].(float64)
		if !ok || timeSteps <= 0 {
			timeSteps = 1.0 // Default 1 step
		}
		currentEnergy := a.State.InternalMetrics["energy"]
		currentLoad := a.State.InternalMetrics["load"]
		decayRatePerStep := 1.0 + (currentLoad * 0.5) // Higher load means faster decay

		predictedEnergy := currentEnergy - (decayRatePerStep * timeSteps)
		if predictedEnergy < 0 {
			predictedEnergy = 0
		}
		predictedState["energy"] = predictedEnergy
		return predictedState, nil
	case "task_completion_predict":
		// Predict if a task finishes based on energy and task complexity
		taskID, taskOK := input["taskID"].(string)
		complexity, compOK := input["complexity"].(float64)
		if !taskOK || !compOK || complexity <= 0 {
			return nil, fmt.Errorf("missing taskID or complexity for prediction")
		}
		currentEnergy := a.State.InternalMetrics["energy"]
		// Simple rule: Task completes if energy > complexity * 10
		predictedOutcome := "incomplete"
		if currentEnergy > complexity*10.0 {
			predictedOutcome = "complete"
		}
		predictedState["taskOutcome"] = predictedOutcome
		predictedState["taskID"] = taskID
		return predictedState, nil
	default:
		return nil, fmt.Errorf("unknown prediction rule: %s", ruleName)
	}
}

// 15. RunSimulatedScenario(scenarioName string, initialConditions map[string]interface{}): Runs a simple simulation.
func (a *AIAgent) RunSimulatedScenario(scenarioName string, initialConditions map[string]interface{}) (map[string]interface{}, error) {
	// Accessing state requires the lock held by ProcessCommand
	outcome := make(map[string]interface{})

	// Create a temporary state copy for simulation (important: shallow copy!)
	// For deep copy, more complex logic needed. Simple values are fine.
	simState := *a.State // Shallow copy
	simState.InternalMetrics = make(map[string]float64)
	for k, v := range a.State.InternalMetrics { simState.InternalMetrics[k] = v }
	simState.Parameters = make(map[string]string)
	for k, v := range a.State.Parameters { simState.Parameters[k] = v }
	// ... copy other state parts as needed for the simulation

	// Apply initial conditions to simState
	for key, value := range initialConditions {
		// This requires logic to apply initial conditions to specific state parts
		// Simple: map string keys to metric names for now
		if _, ok := simState.InternalMetrics[key]; ok {
			if val, ok := value.(float64); ok {
				simState.InternalMetrics[key] = val
			}
		}
		// Add more mapping logic as needed for parameters, etc.
	}


	// Hardcoded simulation logic based on scenario name
	switch scenarioName {
	case "energy_depletion_over_time":
		timeSteps, ok := initialConditions["timeSteps"].(float64)
		if !ok || timeSteps <= 0 {
			timeSteps = 5.0 // Default 5 steps
		}
		loadFactor, ok := initialConditions["loadFactor"].(float64)
		if !ok {
			loadFactor = 1.0 // Default load factor
		}

		currentEnergy := simState.InternalMetrics["energy"]
		simState.InternalMetrics["load"] = loadFactor // Set sim load

		// Simulate energy decay over steps
		for i := 0; i < int(timeSteps); i++ {
			decayRatePerStep := 1.0 + (simState.InternalMetrics["load"] * 0.5)
			currentEnergy -= decayRatePerStep
			if currentEnergy < 0 {
				currentEnergy = 0
				break // Depleted
			}
		}
		outcome["final_energy"] = currentEnergy
		outcome["depleted"] = currentEnergy == 0
		return outcome, nil

	// Add more scenarios
	default:
		return nil, fmt.Errorf("unknown scenario: %s", scenarioName)
	}
}

// --- 7. Knowledge & Memory Functions ---

// 10. StoreKnowledgeFact(category string, key string, value string, tags []string): Stores a piece of information.
func (a *AIAgent) StoreKnowledgeFact(category string, key string, value string, tags []string) error {
	// Accessing state requires the lock held by ProcessCommand
	if a.State.KnowledgeBase[category] == nil {
		a.State.KnowledgeBase[category] = make(map[string]string)
	}
	a.State.KnowledgeBase[category][key] = value
	a.State.KnowledgeTags[key] = append(a.State.KnowledgeTags[key], tags...) // Append new tags
	a.LogEvent("knowledge", "Fact stored", map[string]interface{}{"category": category, "key": key})
	return nil
}

// 11. RetrieveKnowledgeFact(category string, key string, tags []string): Retrieves information.
func (a *AIAgent) RetrieveKnowledgeFact(category string, key string, tags []string) (string, error) {
	// Accessing state requires the lock held by ProcessCommand
	catMap, ok := a.State.KnowledgeBase[category]
	if !ok {
		return "", fmt.Errorf("knowledge category '%s' not found", category)
	}
	value, ok := catMap[key]
	if !ok {
		return "", fmt.Errorf("knowledge key '%s' not found in category '%s'", key, category)
	}

	// Simple tag check (AND logic): all provided tags must be present
	if len(tags) > 0 {
		storedTags, tagsOK := a.State.KnowledgeTags[key]
		if !tagsOK {
			return "", fmt.Errorf("knowledge key '%s' has no tags, cannot match filter", key)
		}
		tagMatch := true
		for _, tag := range tags {
			foundTag := false
			for _, storedTag := range storedTags {
				if strings.EqualFold(strings.TrimSpace(tag), strings.TrimSpace(storedTag)) {
					foundTag = true
					break
				}
			}
			if !foundTag {
				tagMatch = false
				break
			}
		}
		if !tagMatch {
			return "", fmt.Errorf("knowledge key '%s' found, but does not match all tags", key)
		}
	}


	a.LogEvent("knowledge", "Fact retrieved", map[string]interface{}{"category": category, "key": key})
	return value, nil
}

// 12. AssociateKnowledge(key1 string, key2 string, relationship string): Creates a simple link.
// Keys here are assumed to be unique across categories for simplicity in linking.
// A real KB might need category in link.
func (a *AIAgent) AssociateKnowledge(key1 string, key2 string, relationship string) error {
	// Accessing state requires the lock held by ProcessCommand
	// Check if keys exist (optional but good practice)
	key1Found := false
	for _, catMap := range a.State.KnowledgeBase {
		if _, ok := catMap[key1]; ok {
			key1Found = true
			break
		}
	}
	key2Found := false
	for _, catMap := range a.State.KnowledgeBase {
		if _, ok := catMap[key2]; ok {
			key2Found = true
			break
		}
	}

	if !key1Found || !key2Found {
		return fmt.Errorf("one or both keys ('%s', '%s') not found for association", key1, key2)
	}


	if a.State.KnowledgeLinks[key1] == nil {
		a.State.KnowledgeLinks[key1] = make(map[string]string)
	}
	a.State.KnowledgeLinks[key1][key2] = relationship

	// Optional: store reciprocal link
	if a.State.KnowledgeLinks[key2] == nil {
		a.State.KnowledgeLinks[key2] = make(map[string]string)
	}
	a.State.KnowledgeLinks[key2][key1] = "related_via_" + relationship // Simple reciprocal label

	a.LogEvent("knowledge", "Knowledge associated", map[string]interface{}{"key1": key1, "key2": key2, "relationship": relationship})
	return nil
}

// 13. DecayKnowledge(category string, key string): Marks knowledge for decay (simple removal for this implementation).
func (a *AIAgent) DecayKnowledge(category string, key string) error {
	// Accessing state requires the lock held by ProcessCommand
	catMap, ok := a.State.KnowledgeBase[category]
	if !ok {
		return fmt.Errorf("knowledge category '%s' not found for decay", category)
	}
	if _, ok := catMap[key]; !ok {
		return fmt.Errorf("knowledge key '%s' not found in category '%s' for decay", key, category)
	}

	// In a real system, this might reduce a "recency" or "confidence" score.
	// Here, we simply delete it.
	delete(catMap, key)
	delete(a.State.KnowledgeTags, key) // Remove associated tags
	// Links are harder to clean up completely without iterating all links.
	// Simple approach: remove links *from* the key, not necessarily *to* it.
	delete(a.State.KnowledgeLinks, key)

	a.LogEvent("knowledge", "Knowledge decayed/removed", map[string]interface{}{"category": category, "key": key})
	return nil
}

// --- 9. Interaction & Communication Functions ---

// 18. GenerateRuleBasedResponse(context map[string]interface{}): Generates a response string based on current state and context.
func (a *AIAgent) GenerateRuleBasedResponse(context map[string]interface{}) (string, error) {
	// Accessing state requires the lock held by ProcessCommand
	// Example: Respond based on simulated mood and presence of certain context keys

	mood := a.State.InternalMetrics["simulated_mood"] // 0-1 scale
	subject, subjectOK := context["subject"].(string)
	status, statusOK := context["status"].(string)

	baseResponse := "Acknowledged."

	if subjectOK {
		baseResponse = fmt.Sprintf("Regarding %s: ", subject)
		if statusOK {
			baseResponse += fmt.Sprintf("Current status is %s. ", status)
		}
	} else if statusOK {
		baseResponse = fmt.Sprintf("Current status: %s. ", status)
	}


	// Add mood inflection
	if mood > 0.75 {
		baseResponse += "All systems nominal."
	} else if mood < 0.25 {
		baseResponse += "Warning: Internal state requires attention."
	} else {
		// Neutral or slight variation
		if !subjectOK && !statusOK {
			baseResponse = "Ready." // Simple default if no context
		}
	}

	a.LogEvent("communication", "Response generated", map[string]interface{}{"context": context, "response": baseResponse})
	return baseResponse, nil
}

// 19. IdentifySimpleIntent(input string): Maps input text to a predefined simple intent.
// Basic keyword matching for predefined intents.
func (a *AIAgent) IdentifySimpleIntent(input string) (string, error) {
	// Accessing state requires the lock held by ProcessCommand
	inputLower := strings.ToLower(input)

	// Simple intent mapping rules (hardcoded)
	if strings.Contains(inputLower, "status") || strings.Contains(inputLower, "how are you") {
		return "query_status", nil
	}
	if strings.Contains(inputLower, "learn") || strings.Contains(inputLower, "store") || strings.Contains(inputLower, "remember") {
		return "request_learn_fact", nil
	}
	if strings.Contains(inputLower, "retrieve") || strings.Contains(inputLower, "what do you know") || strings.Contains(inputLower, "recall") {
		return "request_retrieve_fact", nil
	}
	if strings.Contains(inputLower, "run") || strings.Contains(inputLower, "simulate") || strings.Contains(inputLower, "scenario") {
		return "request_run_scenario", nil
	}
	if strings.Contains(inputLower, "prioritize") || strings.Contains(inputLower, "task") || strings.Contains(inputLower, "queue") {
		return "request_prioritize_task", nil
	}

	a.LogEvent("communication", "Intent identified", map[string]interface{}{"input": input, "intent": "unknown"})
	return "unknown", nil // Default unknown intent
}

// 20. RequestClarification(uncertaintyKey string): Indicates uncertainty and requests more specific input.
func (a *AIAgent) RequestClarification(uncertaintyKey string) (string, error) {
	// Accessing state requires the lock held by ProcessCommand
	// Based on the 'uncertaintyKey', generate a specific clarification request.
	response := fmt.Sprintf("I require more information. Please specify the '%s'.", uncertaintyKey)
	a.LogEvent("communication", "Clarification requested", map[string]interface{}{"uncertaintyKey": uncertaintyKey, "response": response})
	return response, nil
}

// --- 10. Advanced/Creative Functions ---

// 23. BlendSimpleConcepts(conceptA string, conceptB string, blendRules map[string]string): Combines two stored concepts.
// Concepts are assumed to be keys in the KnowledgeBase, and blendRules map property names.
// Very simplistic: combines property values if rules define how.
func (a *AIAgent) BlendSimpleConcepts(conceptA string, conceptB string, blendRules map[string]string) (map[string]string, error) {
	// Accessing state requires the lock held by ProcessCommand
	// Retrieve representations (assuming they are categories for simplicity)
	conceptAMap, okA := a.State.KnowledgeBase[conceptA]
	conceptBMap, okB := a.State.KnowledgeBase[conceptB]

	if !okA || !okB {
		return nil, fmt.Errorf("one or both concept categories ('%s', '%s') not found", conceptA, conceptB)
	}

	blended := make(map[string]string)

	// Apply blend rules: map a property from A or B to a new property in blended.
	// e.g., blendRules = {"color": "conceptA.color", "material": "conceptB.material", "new_prop": "combine(conceptA.prop1, conceptB.prop2)"}
	for blendedProp, sourceRule := range blendRules {
		if strings.HasPrefix(sourceRule, "conceptA.") {
			prop := strings.TrimPrefix(sourceRule, "conceptA.")
			if val, ok := conceptAMap[prop]; ok {
				blended[blendedProp] = val
			}
		} else if strings.HasPrefix(sourceRule, "conceptB.") {
			prop := strings.TrimPrefix(sourceRule, "conceptB.")
			if val, ok := conceptBMap[prop]; ok {
				blended[blendedProp] = val
			}
		} else if strings.HasPrefix(sourceRule, "combine(") && strings.HasSuffix(sourceRule, ")") {
			// Very basic combine example: just concatenate properties
			combineParts := strings.TrimSuffix(strings.TrimPrefix(sourceRule, "combine("), ")")
			propsToCombine := strings.Split(combineParts, ",") // e.g., "conceptA.prop1,conceptB.prop2"
			combinedValue := ""
			for _, p := range propsToCombine {
				p = strings.TrimSpace(p)
				if strings.HasPrefix(p, "conceptA.") {
					prop := strings.TrimPrefix(p, "conceptA.")
					if val, ok := conceptAMap[prop]; ok {
						combinedValue += val + " "
					}
				} else if strings.HasPrefix(p, "conceptB.") {
					prop := strings.TrimPrefix(p, "conceptB.")
					if val, ok := conceptBMap[prop]; ok {
						combinedValue += val + " "
					}
				}
			}
			blended[blendedProp] = strings.TrimSpace(combinedValue)
		}
		// Add more complex blending logic here (e.g., average numbers, choose based on rule)
	}

	a.LogEvent("conceptual_processing", "Concepts blended", map[string]interface{}{"conceptA": conceptA, "conceptB": conceptB, "blended": blended})
	return blended, nil
}

// 24. GenerateSimpleAnalogy(sourceConcept string, targetDomain string, mappingRules map[string]string): Finds a simple analogy.
// Maps properties/relationships from a source concept (KB category) to a target domain (another KB category or conceptual space) based on rules.
// Simplistic: map properties by name or rule.
func (a *AIAgent) GenerateSimpleAnalogy(sourceConcept string, targetDomain string, mappingRules map[string]string) (map[string]string, error) {
	// Accessing state requires the lock held by ProcessCommand
	sourceMap, okSource := a.State.KnowledgeBase[sourceConcept]
	targetMap, okTarget := a.State.KnowledgeBase[targetDomain] // Target domain might also be a concept category

	if !okSource {
		return nil, fmt.Errorf("source concept category '%s' not found", sourceConcept)
	}
	if !okTarget {
		return nil, fmt.Errorf("target domain category '%s' not found", targetDomain)
	}

	analogy := make(map[string]string) // Maps source property -> target property/value

	// Apply mapping rules: maps source property name/value to a target property name/value
	// e.g., mappingRules = {"source_prop1": "target_prop_equivalent", "source_prop2_value": "target_result_value"}
	for sourceKey, targetMapping := range mappingRules {
		// Simple mapping: sourceKey is a property name from the sourceConcept
		if sourceValue, ok := sourceMap[sourceKey]; ok {
			// Check if targetMapping is a property name in the target domain
			if targetValue, ok := targetMap[targetMapping]; ok {
				analogy[sourceKey] = fmt.Sprintf("is to %s (value: %s)", targetMapping, targetValue)
			} else {
				// If not a property, maybe it's a literal value or a rule result
				analogy[sourceKey] = fmt.Sprintf("maps to '%s'", targetMapping)
			}
			// Could add more complex mapping logic here
		}
	}

	// Also consider structural mapping via KnowledgeLinks if concepts are linked
	// e.g., If A -> B (relation X) and C -> D (relation Y), and relation X is analogous to relation Y,
	// then A is analogous to C and B is analogous to D. (Too complex for this simple implementation).

	a.LogEvent("conceptual_processing", "Analogy generated", map[string]interface{}{"source": sourceConcept, "target": targetDomain, "analogy": analogy})
	return analogy, nil
}


// 25. NegotiateBasicConstraints(constraints []map[string]string): Resolves conflicting constraints.
// Constraints are maps like {"type": "resource", "name": "cpu", "value": "high", "priority": "low"}
// Simple: resolve based on a predefined priority order of constraint types or explicit priorities.
func (a *AIAgent) NegotiateBasicConstraints(constraints []map[string]string) (map[string]string, error) {
	// Accessing state requires the lock held by ProcessCommand
	resolved := make(map[string]string)
	conflicts := []map[string]string{} // Track unresolved conflicts

	// Define a simple priority order for constraint types (higher index = higher priority)
	priorityOrder := map[string]int{
		"safety":    3,
		"critical":  2, // e.g., critical resource limits
		"resource":  1, // e.g., CPU, memory limits
		"time":      0, // e.g., deadlines
		"preference": -1, // e.g., user preference
	}

	// Sort constraints by priority (descending)
	// Note: This is a simplified sort. Real negotiation is much more complex.
	// Here, we just iterate, prioritizing based on the *first* constraint encountered for a key,
	// or maybe just using the highest explicit 'priority' value if present.
	// Let's use explicit priority first, falling back to type priority.

	sortedConstraints := make([]map[string]string, len(constraints))
	copy(sortedConstraints, constraints)

	// Sort by explicit priority (high to low) then type priority (high to low)
	// Requires a proper sort function or manually building based on priority.
	// Simple approach: Use a map to track the highest priority constraint seen for each key (e.g., "resource:cpu").
	constraintDecisions := make(map[string]map[string]string) // key (e.g. "resource:cpu") -> winning constraint

	for _, c := range constraints {
		cType, typeOK := c["type"]
		cName, nameOK := c["name"]
		cValue, valueOK := c["value"]
		cPriorityStr, priorityOK := c["priority"]

		if !typeOK || !nameOK || !valueOK {
			// Skip malformed constraints
			a.LogEvent("warning", "Skipped malformed constraint", map[string]interface{}{"constraint": c})
			continue
		}

		constraintKey := cType + ":" + cName
		currentWinner, hasWinner := constraintDecisions[constraintKey]

		currentConstraintPriority := -100 // Default low priority
		if priorityOK {
			p, err := strconv.Atoi(cPriorityStr)
			if err == nil {
				currentConstraintPriority = p // Use explicit priority if valid
			}
		}
		if currentConstraintPriority == -100 { // If no valid explicit priority, use type priority
			currentConstraintPriority, _ = priorityOrder[cType] // 0 if type not in map
		}


		if !hasWinner {
			// First constraint for this key wins for now
			constraintDecisions[constraintKey] = c
			resolved[constraintKey] = cValue // Record the resolved value
		} else {
			// Compare with existing winner
			winnerPriority := -100
			winnerPriorityStr, winnerPriorityOK := currentWinner["priority"]
			if winnerPriorityOK {
				p, err := strconv.Atoi(winnerPriorityStr)
				if err == nil {
					winnerPriority = p
				}
			}
			if winnerPriority == -100 {
				winnerType, _ := currentWinner["type"]
				winnerPriority, _ = priorityOrder[winnerType]
			}

			if currentConstraintPriority > winnerPriority {
				// This constraint is higher priority, it becomes the winner
				constraintDecisions[constraintKey] = c
				resolved[constraintKey] = cValue
				// Add previous winner to conflicts?
			} else if currentConstraintPriority == winnerPriority {
				// Same priority - this is an unresolved conflict
				conflicts = append(conflicts, c)
				// Mark the key as conflicted? For now, the first one processed wins the 'resolved' slot
				a.LogEvent("warning", "Detected equal priority conflict", map[string]interface{}{"key": constraintKey, "constraints": []map[string]string{currentWinner, c}})
			} else {
				// This constraint is lower priority, it's ignored for resolution but potentially logged as conflict
				conflicts = append(conflicts, c)
			}
		}
	}

	// Report conflicts if any
	if len(conflicts) > 0 {
		conflictsJSON, _ := json.Marshal(conflicts)
		a.LogEvent("conflict", "Unresolved constraints detected", map[string]interface{}{"conflicts": string(conflictsJSON)})
	}

	a.LogEvent("negotiation", "Constraints negotiated", map[string]interface{}{"resolved": resolved, "conflicts": conflicts})
	return resolved, nil
}

// 26. ReportLastActionRationale(): Provides a simple trace or explanation of the last command processing.
func (a *AIAgent) ReportLastActionRationale() string {
	// Accessing state requires the lock held by ProcessCommand
	// This function just reports the stored rationale.
	return a.State.LastAction.Rationale
}


// 28. AnticipateNextInput(historyWindow int): Based on recent command history, predicts a likely next command.
// Simple: Look for common sequences in the last `historyWindow` commands.
func (a *AIAgent) AnticipateNextInput(historyWindow int) (string, error) {
	// Accessing state requires the lock held by ProcessCommand
	if historyWindow <= 0 {
		return "", fmt.Errorf("historyWindow must be positive")
	}

	history := a.State.CommandHistory
	if len(history) < 2 {
		return "No sufficient history for anticipation.", nil
	}

	// Consider only the last `historyWindow` commands
	start := 0
	if len(history) > historyWindow {
		start = len(history) - historyWindow
	}
	recentHistory := history[start:]

	// Simple frequency analysis of commands following recent commands
	// Map: command -> map[nextCommand] -> count
	sequenceCounts := make(map[string]map[string]int)

	for i := 0; i < len(recentHistory)-1; i++ {
		currentCmd := recentHistory[i]
		nextCmd := recentHistory[i+1]

		if sequenceCounts[currentCmd] == nil {
			sequenceCounts[currentCmd] = make(map[string]int)
		}
		sequenceCounts[currentCmd][nextCmd]++
	}

	// Find the command that most frequently followed the *last* command
	lastCmd := recentHistory[len(recentHistory)-1]
	possibleNextCommands, ok := sequenceCounts[lastCmd]

	if !ok || len(possibleNextCommands) == 0 {
		return "Cannot anticipate based on recent history.", nil
	}

	// Find the most frequent next command
	mostFrequentNext := ""
	maxCount := 0
	for nextCmd, count := range possibleNextCommands {
		if count > maxCount {
			maxCount = count
			mostFrequentNext = nextCmd
		}
	}

	if mostFrequentNext != "" {
		a.LogEvent("prediction", "Anticipated next input", map[string]interface{}{"based_on": lastCmd, "predicted": mostFrequentNext})
		return mostFrequentNext, nil
	}

	return "Cannot anticipate based on recent history.", nil
}

// 29. ModelAbstractEntityState(entityID string, stateUpdate map[string]interface{}): Updates the internal simulated state of an abstract entity.
func (a *AIAgent) ModelAbstractEntityState(entityID string, stateUpdate map[string]interface{}) error {
	// Accessing state requires the lock held by ProcessCommand
	if a.State.AbstractEntityStates[entityID] == nil {
		a.State.AbstractEntityStates[entityID] = make(map[string]interface{})
	}

	// Apply updates to the entity's state
	for key, value := range stateUpdate {
		a.State.AbstractEntityStates[entityID][key] = value
	}

	a.LogEvent("entity_modeling", "Entity state updated", map[string]interface{}{"entityID": entityID, "update": stateUpdate, "current_state": a.State.AbstractEntityStates[entityID]})
	return nil
}

// 30. LearnSimpleSequence(sequence []string, outcome map[string]interface{}): Records a sequence of commands/events.
// Stores a sequence and its associated outcome for potential future recognition or use in prediction/automation.
func (a *AIAgent) LearnSimpleSequence(sequence []string, outcome map[string]interface{}) {
	// Accessing state requires the lock held by ProcessCommand
	if len(sequence) == 0 {
		a.LogEvent("warning", "Attempted to learn empty sequence", nil)
		return
	}

	sequenceKey := strings.Join(sequence, " -> ") // Use a string key for the map

	// Store or update the sequence and its outcome/metadata
	// Could add logic here to merge outcomes if the same sequence is learned multiple times
	a.State.SimpleSequences[sequenceKey] = outcome // Overwrite or merge

	a.LogEvent("learning", "Sequence learned", map[string]interface{}{"sequence": sequenceKey, "outcome": outcome})
}

// --- Task Queue Functions (Simple Implementation) ---

// 16. PrioritizeTaskQueue(taskID string, priority int): Modifies the priority of a task.
// Simple: just store the priority in the map. Adding/removing tasks from the queue is separate.
func (a *AIAgent) PrioritizeTaskQueue(taskID string, priority int) error {
	// Accessing state requires the lock held by ProcessCommand
	// Check if task exists in the queue (simplified - assume any ID can be prioritized)
	// A real queue would need task objects.
	a.State.TaskPriorities[taskID] = priority
	a.LogEvent("task_management", "Task priority set", map[string]interface{}{"taskID": taskID, "priority": priority})
	return nil
}

// 17. GetNextTask(): Retrieves the highest priority task from the internal queue.
// Simple: finds the task ID with the highest priority in TaskPriorities map and assumes it's 'in' the queue.
// Removes it from TaskPriorities after retrieving (simulating processing).
func (a *AIAgent) GetNextTask() (string, error) {
	// Accessing state requires the lock held by ProcessCommand
	if len(a.State.TaskPriorities) == 0 {
		return "", fmt.Errorf("task queue is empty")
	}

	highestPriority := -1 // Assuming priorities can be 0 or positive
	var nextTaskID string
	found := false

	for taskID, priority := range a.State.TaskPriorities {
		if !found || priority > highestPriority {
			highestPriority = priority
			nextTaskID = taskID
			found = true
		}
	}

	if found {
		delete(a.State.TaskPriorities, nextTaskID) // Remove from map after retrieval
		a.LogEvent("task_management", "Next task retrieved", map[string]interface{}{"taskID": nextTaskID, "priority": highestPriority})
		return nextTaskID, nil
	}

	return "", fmt.Errorf("error determining next task") // Should not happen if map is not empty
}

// --- Preference Learning (Simple) ---

// 35. LearnPreference(key string, value string): Stores a simple preference.
// Not in the initial list count, adding to reach/exceed 20 easily.
func (a *AIAgent) LearnPreference(key string, value string) {
    // Accessing state requires the lock held by ProcessCommand
	a.State.Preferences[key] = value
	a.LogEvent("learning", "Preference learned", map[string]interface{}{"key": key, "value": value})
}

// 36. RetrievePreference(key string): Retrieves a simple preference.
func (a *AIAgent) RetrievePreference(key string) (string, error) {
    // Accessing state requires the lock held by ProcessCommand
	value, ok := a.State.Preferences[key]
	if !ok {
		return "", fmt.Errorf("preference '%s' not found", key)
	}
	a.LogEvent("knowledge", "Preference retrieved", map[string]interface{}{"key": key, "value": value})
	return value, nil
}

// --- Additional Function (Simulate Delegation) ---

// 34. SimulateDelegation(subTaskCommand string): Simulate delegating a task to an internal sub-process or sub-agent.
// Implemented by simply processing another command internally.
func (a *AIAgent) SimulateDelegation(subTaskCommand string) (string, error) {
    // Accessing state requires the lock held by ProcessCommand
	a.LogEvent("task_management", "Simulating delegation", map[string]interface{}{"subTaskCommand": subTaskCommand})
	// Recursively call ProcessCommand for the sub-task.
	// Note: This is simplified; a real delegation might use separate concurrency or state.
	// This implementation shares the same lock and state.
	result, err := a.ProcessCommand(subTaskCommand)
	a.LogEvent("task_management", "Delegated task completed", map[string]interface{}{"subTaskCommand": subTaskCommand, "result": result, "error": fmt.Sprintf("%v", err)})
	return result, err
}


// Main function for demonstration
func main() {
	agent := NewAIAgent()

	fmt.Println("--- Agent Initialized ---")

	// Example Commands
	commands := []string{
		`GET_STATE key="metrics"`,
		`SET_STATE_VALUE key="simulated_mood" value="0.9"`,
		`GET_STATE key="simulated_mood"`,
		`APPLY_RULE ruleName="greet" context="{\"name\":\"Master\"}"`,
		`STORE_KNOWLEDGE_FACT category="concepts" key="apple" value="fruit, red/green, sweet" tags="food, organic"`,
		`STORE_KNOWLEDGE_FACT category="concepts" key="banana" value="fruit, yellow, sweet" tags="food, organic"`,
		`STORE_KNOWLEDGE_FACT category="properties" key="color:red" value="indicates ripeness"`,
		`RETRIEVE_KNOWLEDGE_FACT category="concepts" key="apple"`,
		`RETRIEVE_KNOWLEDGE_FACT category="concepts" key="banana" tags="organic"`,
		`ASSOCIATE_KNOWLEDGE key1="apple" key2="banana" relationship="both_fruit"`,
		`GET_STATE key="knowledge"`, // Check stored knowledge and associations
		`IDENTIFY_SIMPLE_INTENT input="What do you know about apples?"`,
		`GENERATE_RULE_BASED_RESPONSE context="{\"subject\":\"task status\",\"status\":\"pending\"}"`,
		`UPDATE_EMOTIONAL_STATE eventImpact="{\"success\":0.3,\"difficulty\":-0.1}"`,
		`GET_STATE key="simulated_mood"`,
		`MONITOR_INTERNAL_METRIC metricName="energy"`,
		`CHECK_INTERNAL_CONDITION conditionName="is_energy_low" params="{\"threshold\":10.0}"`,
		`PREDICT_STATE_TRANSITION input="{\"timeSteps\":2.0,\"load\":0.6}" rules="{\"ruleName\":\"energy_decay_predict\"}"`,
		`RUN_SIMULATED_SCENARIO scenarioName="energy_depletion_over_time" initialConditions="{\"timeSteps\":3.0,\"loadFactor\":1.2}"`,
		`PRIORITIZE_TASK_QUEUE taskID="task_001" priority=10`,
		`PRIORITIZE_TASK_QUEUE taskID="task_002" priority=5`,
		`GET_NEXT_TASK`, // task_001 should be retrieved
		`GET_NEXT_TASK`, // task_002 should be retrieved
		`BLEND_SIMPLE_CONCEPTS conceptA="concepts" conceptB="properties" blendRules="{\"color_desc\":\"combine(conceptA.color, conceptB.color:red)\"}"`, // This example needs 'color' property on 'concepts' to work well, KB structure is simple string value, need refinement for structured concepts. Let's retry with simpler blend rule.
		`STORE_KNOWLEDGE_FACT category="shapes" key="circle" value="round, one edge"`,
		`STORE_KNOWLEDGE_FACT category="colors" key="red" value="primary, hot"`,
		`BLEND_SIMPLE_CONCEPTS conceptA="shapes" conceptB="colors" blendRules="{\"feeling\":\"conceptB.value\",\"shape_info\":\"conceptA.value\"}"`, // Blend 'colors' (category) with 'shapes' (category), map properties
		`GENERATE_SIMPLE_ANALOGY sourceConcept="shapes" targetDomain="colors" mappingRules="{\"round\":\"primary\"}"`, // Analogy: round is to primary color? (Needs meaningful rules)
		`NEGOTIATE_BASIC_CONSTRAINTS constraints="[{\"type\":\"resource\",\"name\":\"cpu\",\"value\":\"high\",\"priority\":\"1\"}, {\"type\":\"resource\",\"name\":\"cpu\",\"value\":\"low\",\"priority\":\"0\"}, {\"type\":\"time\",\"name\":\"deadline\",\"value\":\"urgent\",\"priority\":\"2\"}]"`,
		`REPORT_LAST_ACTION_RATIONALE`,
		`ANTICIPATE_NEXT_INPUT historyWindow=5`,
		`MODEL_ABSTRACT_ENTITY_STATE entityID="drone_001" stateUpdate="{\"position\":\"(10, 20, 5)\",\"battery\":0.85}"`,
		`LEARN_SIMPLE_SEQUENCE sequence="IDENTIFY_SIMPLE_INTENT,GENERATE_RULE_BASED_RESPONSE" outcome="{\"description\":\"standard interaction loop\"}"`,
		`GET_STATE key="sequences"`,
		`LEARN_PREFERENCE key="favorite_color" value="blue"`,
		`RETRIEVE_PREFERENCE key="favorite_color"`,
		`SIMULATE_DELEGATION subTaskCommand="GET_STATE key=\\"preferences\\""`, // Delegating a GET_STATE call

	}

	for _, cmd := range commands {
		fmt.Printf("\n>>> Processing: %s\n", cmd)
		result, err := agent.ProcessCommand(cmd)
		if err != nil {
			fmt.Printf("!!! Error: %v\n", err)
		} else {
			fmt.Printf("<<< Result: %s\n", result)
		}
	}

	fmt.Println("\n--- Final State Metrics ---")
	metricsJSON, _ := agent.handleGetState("metrics") // Access state directly for final report
	fmt.Println(metricsJSON)
}
```

**Explanation and How it Meets Requirements:**

1.  **Go Language:** Written entirely in Go.
2.  **MCP Interface:** The `ProcessCommand` function serves as the central Message/Command Processing interface. It takes a string command, parses it, and dispatches it to the appropriate internal method. This isolates the core logic from the input/output method (which is just `main` here, but could be network, file, etc.).
3.  **AI-Agent:** The "AI" aspect is achieved through:
    *   **State Management:** `AgentState` holds internal variables, simulating memory and internal status.
    *   **Rule-Based Processing:** Functions like `ApplyRule`, `GenerateRuleBasedResponse`, `PredictStateTransition`, `CheckInternalCondition`, `AdjustParameterRule` use hardcoded or simple rule structures to guide behavior and derive outcomes based on state and input. This avoids external ML models but provides deterministic, rule-driven "intelligence."
    *   **Knowledge Representation:** `KnowledgeBase`, `KnowledgeTags`, `KnowledgeLinks` provide a simple symbolic representation capability.
    *   **Simple Simulation:** `RunSimulatedScenario` allows projecting outcomes based on simplified internal models and rules.
    *   **Pattern Recognition:** `FindKeywordPattern`, `IdentifySimpleIntent`, `LearnSimpleSequence`, `AnticipateNextInput` use basic pattern matching and history analysis.
    *   **Conceptual Operations:** `BlendSimpleConcepts`, `GenerateSimpleAnalogy` implement highly simplified versions of cognitive tasks based on manipulating the symbolic knowledge representation.
    *   **Self-Management Simulation:** `MonitorInternalMetric`, `PrioritizeTaskQueue`, `UpdateEmotionalState` (simulated emotion) represent internal monitoring and control.
    *   **Introspection:** `ReportLastActionRationale` provides a basic level of reporting on its own process.
    *   **Negotiation:** `NegotiateBasicConstraints` uses priority rules to resolve conflicts.
    *   **Learning (Simple):** `LearnSimpleSequence`, `LearnPreference` allow recording information for future use.
    *   **Modeling:** `ModelAbstractEntityState` allows tracking external (abstract) entities.
    *   **Delegation (Simulated):** `SimulateDelegation` shows how complex tasks could be broken down internally.
4.  **Interesting, Advanced-Concept, Creative, Trendy:**
    *   **Interesting/Creative:** `BlendSimpleConcepts`, `GenerateSimpleAnalogy`, `NegotiateBasicConstraints`, `ReportLastActionRationale`, `UpdateEmotionalState`, `SimulateDelegation`, `ModelAbstractEntityState` implement concepts often associated with more advanced AI but are done here with simple, custom, non-ML logic.
    *   **Advanced-Concept (Simplified):** Functions like state prediction, scenario simulation, task prioritization, knowledge association, and intent identification touch upon areas relevant to intelligent agents.
    *   **Trendy:** The overall "AI Agent" paradigm is trendy. The specific functions cover areas like cognitive simulation, symbolic AI remnants (knowledge base), and basic self-management.
5.  **At Least 20 Functions:** The summary lists over 30 functions, and the code implements handlers for most of them via the `ProcessCommand` switch statement.
6.  **No Duplication of Open Source:** The core logic for each function is implemented using standard Go features (maps, slices, string manipulation, simple loops, conditions). It *does not* rely on external libraries for machine learning, neural networks, natural language processing parsers (beyond simple string checks), complex optimization solvers, or other typical AI framework components. The "intelligence" is handcrafted rules and state transitions.
7.  **Outline and Summary:** Provided at the top as requested.

**Limitations and Further Development:**

*   **Parsing:** The command parsing is very basic. A real application might use a more robust parser, potentially with a grammar.
*   **Rule Complexity:** Rules are hardcoded within the handler functions. A more flexible design would involve loading rules from configuration or a knowledge base.
*   **Knowledge Representation:** The knowledge base is simple string key-value pairs. A more advanced agent might use a graph database or a more structured semantic representation.
*   **Learning:** The "learning" functions are currently just recording. Real learning would involve adjusting rules or internal parameters based on experience/data.
*   **Concurrency:** While `sync.RWMutex` is included, complex concurrent command processing would require careful design to avoid deadlocks or race conditions, especially when multiple commands might modify overlapping parts of the state simultaneously.
*   **Error Handling:** Error handling is basic; production code would need more detailed error types and recovery mechanisms.
*   **Scalability:** The current in-memory state would not scale to vast amounts of data or complex knowledge.
*   **The "AI" Depth:** The intelligence is explicitly *not* based on complex statistical models or learning from large datasets, which is the dominant paradigm in modern AI. This agent simulates AI-like *behaviors* using deterministic rules and state.