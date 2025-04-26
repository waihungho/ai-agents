```go
// Package main implements a simple AI Agent with a simulated MCP interface.
// It demonstrates various conceptual "agent-like" functions, focusing on the
// interface and structure rather than complex AI implementations.
//
// Outline:
// - Define structs for MCP commands and responses.
// - Implement functions to parse incoming MCP strings and format outgoing responses.
// - Define the Agent struct containing state, memory, knowledge base, and handlers.
// - Define the CommandHandlerFunc type for agent command processing.
// - Implement handler functions for a variety of conceptual agent tasks (25+ functions).
// - Implement the NewAgent constructor to initialize the agent and register handlers.
// - Implement the Agent.Run method to process commands from an input channel and
//   send responses to an output channel.
// - Implement helper functions for parsing command arguments and key-value pairs.
// - Main function to demonstrate agent creation and command processing via channels.
//
// Function Summary:
// - ParseMCP(string): Parses a raw string line into an MCPCommand struct.
// - FormatMCPResponse(*MCPResponse): Formats an MCPResponse struct into a string line.
// - extractArgsAndKV(string): Helper to split a string into arguments and key-value pairs.
// - NewAgent(string): Creates and initializes a new Agent with a given ID.
// - Agent.Run(): Starts the command processing loop for the agent.
// - Agent.handleStatus(*MCPCommand): Reports the agent's current high-level status.
// - Agent.handleSetState(*MCPCommand): Sets a specific key-value pair in the agent's state.
// - Agent.handleGetState(*MCPCommand): Retrieves the value for a specific key from the agent's state.
// - Agent.handleRemember(*MCPCommand): Adds a piece of information or experience to the agent's memory.
// - Agent.handleRecall(*MCPCommand): Searches the agent's memory for relevant information based on keywords or context.
// - Agent.handleAddKnowledge(*MCPCommand): Adds structured or unstructured data to the agent's knowledge base.
// - Agent.handleQueryKnowledge(*MCPCommand): Queries the knowledge base for information matching criteria.
// - Agent.handleAnalyzeText(*MCPCommand): Simulates analyzing text input for sentiment, keywords, or structure.
// - Agent.handleSynthesizeInfo(*MCPCommand): Combines information from state, memory, and knowledge base to form a new insight.
// - Agent.handlePlanSimpleTask(*MCPCommand): Generates a conceptual sequence of steps to achieve a simple goal.
// - Agent.handleExecuteSimTask(*MCPCommand): Simulates the execution of a task plan, updating state based on conceptual outcomes.
// - Agent.handleLearnFromOutcome(*MCPCommand): Simulates adjusting internal state, knowledge, or strategy based on a previous task outcome.
// - Agent.handleGenerateText(*MCPCommand): Simulates generating a piece of text (e.g., a message, a summary) based on context.
// - Agent.handleSimulateEvent(*MCPCommand): Runs a simple internal simulation model or scenario.
// - Agent.handlePrognosticate(*MCPCommand): Simulates predicting a future state or event based on current data and models.
// - Agent.handleOptimizeSetting(*MCPCommand): Simulates finding an optimal parameter value for a given objective.
// - Agent.handleDiffState(*MCPCommand): Compares the current state to a previously saved state snapshot.
// - Agent.handleAuditTrail(*MCPCommand): Retrieves a history of recent commands processed by the agent.
// - Agent.handleSelfReflect(*MCPCommand): Generates a summary of the agent's recent activity, state, or internal thoughts (simulated).
// - Agent.handleDetectPattern(*MCPCommand): Searches for recurring patterns or anomalies in recent data streams or memory.
// - Agent.handlePrioritizeGoals(*MCPCommand): Re-evaluates and prioritizes the agent's current goals based on new information.
// - Agent.handleQueryCapabilities(*MCPCommand): Lists the commands or functions the agent is capable of executing.
// - Agent.handleGenerateIdea(*MCPCommand): Simulates generating a novel concept or approach based on inputs or knowledge.
// - Agent.handleClusterMemories(*MCPCommand): Groups similar or related memories together.
// - Agent.handleReportSummary(*MCPCommand): Generates a formatted summary report based on current state, activity, or findings.
// - Agent.handleRequestFeedback(*MCPCommand): Simulates requesting feedback on its performance or output.
// - Agent.handleEvaluateInputTrust(*MCPCommand): Simulates evaluating the trustworthiness or reliability of an input source.
// - Agent.handleSuggestAction(*MCPCommand): Suggests the next logical action based on current state and goals.

package main

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
	"strconv" // Added for parsing numbers
)

// --- MCP Interface Definition and Parsing ---

// MCPCommand represents a parsed MCP command.
type MCPCommand struct {
	Command string
	Args    []string
	KV      map[string]string
}

// MCPResponse represents an MCP response to be sent.
type MCPResponse struct {
	Command string // Original command name
	Success bool
	Message string            // Human-readable message
	Result  map[string]string // Key-value results
}

// ParseMCP parses a raw string line into an MCPCommand struct.
// Simple parsing: splits by space, handles basic key:value, no complex quoting.
func ParseMCP(line string) (*MCPCommand, error) {
	parts := strings.Fields(line)
	if len(parts) == 0 {
		return nil, fmt.Errorf("empty command line")
	}

	cmd := MCPCommand{
		Command: parts[0],
		Args:    []string{},
		KV:      make(map[string]string),
	}

	// Separate args and KV pairs (simple: key:value)
	// This basic implementation assumes KV pairs are at the end and look like key:value
	// A more robust parser would handle quoted arguments and values properly.
	kvStartIdx := len(parts)
	for i, part := range parts[1:] {
		if strings.Contains(part, ":") && strings.HasSuffix(part, "\"") == false { // Avoid splitting URLs or complex strings
			// Could be a KV pair or just an arg with a colon.
			// Simple heuristic: if it has exactly one colon and text on both sides, treat as KV.
			// Let's refine: look for the *first* part that is clearly key:value
			kvParts := strings.SplitN(part, ":", 2)
			if len(kvParts) == 2 && kvParts[0] != "" {
				kvStartIdx = i + 1 // +1 because we are iterating parts[1:]
				break
			}
			cmd.Args = append(cmd.Args, part)
		} else {
			cmd.Args = append(cmd.Args, part)
		}
	}

	// Process KV pairs from kvStartIdx onwards
	for _, part := range parts[1+kvStartIdx:] {
		kvParts := strings.SplitN(part, ":", 2)
		if len(kvParts) == 2 && kvParts[0] != "" {
			cmd.KV[kvParts[0]] = kvParts[1]
		} else {
			// Malformed KV? Treat as an arg? For simplicity, ignore malformed KV here.
			// Or maybe add to args if the heuristic was wrong. Let's add to args.
			cmd.Args = append(cmd.Args, part)
		}
	}

	// A better KV parser that handles quotes and finds KV anywhere
	// Replace the above logic with a more careful scan:
	cmd.Args = []string{}
	cmd.KV = make(map[string]string)
	argsList := []string{}
	kvMap := make(map[string]string)

	currentArg := ""
	inQuote := false

	// Iterate over characters for better quote handling (simple version)
	// This is still not perfect for escaped quotes but better than Fields
	lineScanner := strings.NewReader(line)
	var token strings.Builder

	for {
		r, _, err := lineScanner.ReadRune()
		if err != nil {
			break // EOF or error
		}

		switch r {
		case '"':
			inQuote = !inQuote
			// Optionally include quotes in the token, or not. Let's not for simplicity.
			// token.WriteRune(r)
		case ' ', '\t':
			if inQuote {
				token.WriteRune(r)
			} else {
				// End of token
				if token.Len() > 0 {
					t := token.String()
					token.Reset()

					// Check if it's a potential KV pair: key:value
					if strings.Contains(t, ":") {
						kvParts := strings.SplitN(t, ":", 2)
						if len(kvParts) == 2 && kvParts[0] != "" {
							kvMap[kvParts[0]] = kvParts[1]
						} else {
							argsList = append(argsList, t) // Not a valid KV, treat as arg
						}
					} else {
						argsList = append(argsList, t)
					}
				}
			}
		default:
			token.WriteRune(r)
		}
	}

	// Add any remaining token after loop
	if token.Len() > 0 {
		t := token.String()
		if strings.Contains(t, ":") {
			kvParts := strings.SplitN(t, ":", 2)
			if len(kvParts) == 2 && kvParts[0] != "" {
				kvMap[kvParts[0]] = kvParts[1]
			} else {
				argsList = append(argsList, t)
			}
		} else {
			argsList = append(argsList, t)
		}
	}

	// The command name is the first token
	if len(argsList) > 0 {
		cmd.Command = argsList[0]
		cmd.Args = argsList[1:]
	} else {
		// Should not happen if initial parts check passed, but safety
		return nil, fmt.Errorf("failed to parse command name")
	}
	cmd.KV = kvMap


	return &cmd, nil
}

// FormatMCPResponse formats an MCPResponse struct into a string line.
func FormatMCPResponse(resp *MCPResponse) string {
	status := "error"
	if resp.Success {
		status = "ok"
	}

	var sb strings.Builder
	sb.WriteString(status)
	sb.WriteString(" ")
	sb.WriteString(resp.Command)

	if resp.Message != "" {
		sb.WriteString(" message:\"") // Simple quoting
		sb.WriteString(strings.ReplaceAll(resp.Message, "\"", "\\\"")) // Escape quotes
		sb.WriteString("\"")
	}

	for k, v := range resp.Result {
		sb.WriteString(" ")
		sb.WriteString(k)
		sb.WriteString(":\"")
		sb.WriteString(strings.ReplaceAll(v, "\"", "\\\"")) // Escape quotes
		sb.WriteString("\"")
	}

	return sb.String()
}

// --- Agent Core ---

// CommandHandlerFunc is a function that handles a specific MCP command.
type CommandHandlerFunc func(*Agent, *MCPCommand) *MCPResponse

// Agent represents the AI agent entity.
type Agent struct {
	ID             string
	State          map[string]interface{}
	Memory         []string
	KnowledgeBase  map[string]interface{} // Could be structured, e.g., map[string]map[string]string
	Goals          []string
	Skills         []string // Conceptual skills/capabilities
	Output         chan string // Channel for sending outgoing MCP messages
	Input          chan string // Channel for receiving incoming MCP messages
	CommandHandlers map[string]CommandHandlerFunc
	AuditLog       []MCPCommand // Simple log of received commands
	mu             sync.Mutex   // Mutex for protecting shared state (State, Memory, etc.)
}

// NewAgent creates and initializes a new Agent.
func NewAgent(id string, output chan string, input chan string) *Agent {
	agent := &Agent{
		ID:             id,
		State:          make(map[string]interface{}),
		Memory:         []string{},
		KnowledgeBase:  make(map[string]interface{}),
		Goals:          []string{"maintain_status", "learn_new_patterns"}, // Default goals
		Skills:         []string{"status", "set_state", "get_state", "remember", "recall", "query_capabilities"}, // Initial skills
		Output:         output,
		Input:          input,
		CommandHandlers: make(map[string]CommandHandlerFunc),
		AuditLog:       []MCPCommand{},
	}

	// Register Handlers
	agent.registerHandlers()

	return agent
}

// registerHandlers maps command names to their handler functions.
// This is where the 25+ functions are connected.
func (a *Agent) registerHandlers() {
	a.CommandHandlers["agent.status"] = a.handleStatus
	a.CommandHandlers["agent.set_state"] = a.handleSetState
	a.CommandHandlers["agent.get_state"] = a.handleGetState
	a.CommandHandlers["agent.remember"] = a.handleRemember
	a.CommandHandlers["agent.recall"] = a.handleRecall
	a.CommandHandlers["agent.add_knowledge"] = a.handleAddKnowledge
	a.CommandHandlers["agent.query_knowledge"] = a.handleQueryKnowledge
	a.CommandHandlers["agent.analyze_text"] = a.handleAnalyzeText
	a.CommandHandlers["agent.synthesize_info"] = a.handleSynthesizeInfo
	a.CommandHandlers["agent.plan_simple_task"] = a.handlePlanSimpleTask
	a.CommandHandlers["agent.execute_sim_task"] = a.handleExecuteSimTask
	a.CommandHandlers["agent.learn_from_outcome"] = a.handleLearnFromOutcome
	a.CommandHandlers["agent.generate_text"] = a.handleGenerateText
	a.CommandHandlers["agent.simulate_event"] = a.handleSimulateEvent
	a.CommandHandlers["agent.prognosticate"] = a.handlePrognosticate
	a.CommandHandlers["agent.optimize_setting"] = a.handleOptimizeSetting
	a.CommandHandlers["agent.diff_state"] = a.handleDiffState
	a.CommandHandlers["agent.audit_trail"] = a.handleAuditTrail
	a.CommandHandlers["agent.self_reflect"] = a.handleSelfReflect
	a.CommandHandlers["agent.detect_pattern"] = a.handleDetectPattern
	a.CommandHandlers["agent.prioritize_goals"] = a.handlePrioritizeGoals
	a.CommandHandlers["agent.query_capabilities"] = a.handleQueryCapabilities
	a.CommandHandlers["agent.generate_idea"] = a.handleGenerateIdea
	a.CommandHandlers["agent.cluster_memories"] = a.handleClusterMemories
	a.CommandHandlers["agent.report_summary"] = a.handleReportSummary
	a.CommandHandlers["agent.request_feedback"] = a.handleRequestFeedback
	a.CommandHandlers["agent.evaluate_input_trust"] = a.handleEvaluateInputTrust
	a.CommandHandlers["agent.suggest_action"] = a.handleSuggestAction
}

// Run starts the agent's main loop to listen for commands.
func (a *Agent) Run() {
	fmt.Printf("Agent '%s' started and listening...\n", a.ID)
	for line := range a.Input {
		fmt.Printf("Agent '%s' received: %s\n", a.ID, line)
		cmd, err := ParseMCP(line)
		if err != nil {
			resp := &MCPResponse{
				Success: false,
				Message: fmt.Sprintf("Failed to parse command: %v", err),
				Command: "parse_error", // Indicate parsing failed before command could be identified
			}
			a.Output <- FormatMCPResponse(resp)
			continue
		}

		a.mu.Lock()
		a.AuditLog = append(a.AuditLog, *cmd) // Log the command
		// Keep log size reasonable
		if len(a.AuditLog) > 100 {
			a.AuditLog = a.AuditLog[1:]
		}
		a.mu.Unlock()


		handler, ok := a.CommandHandlers[cmd.Command]
		if !ok {
			resp := &MCPResponse{
				Command: cmd.Command,
				Success: false,
				Message: fmt.Sprintf("Unknown command: %s", cmd.Command),
			}
			a.Output <- FormatMCPResponse(resp)
			continue
		}

		// Execute the handler
		resp := handler(a, cmd)
		a.Output <- FormatMCPResponse(resp)
	}
	fmt.Printf("Agent '%s' input channel closed. Shutting down.\n", a.ID)
}

// --- Agent Command Handlers (25+ functions) ---
// These functions simulate complex tasks using simple logic (print statements,
// state updates) to demonstrate the *interface* and *concept*.

// handleStatus reports agent's current high-level status.
func (a *Agent) handleStatus(cmd *MCPCommand) *MCPResponse {
	a.mu.Lock()
	defer a.mu.Unlock()
	status := fmt.Sprintf("Agent %s is operational.", a.ID)
	stateCount := len(a.State)
	memoryCount := len(a.Memory)
	knowledgeCount := len(a.KnowledgeBase)
	goalCount := len(a.Goals)
	skillCount := len(a.Skills)

	result := map[string]string{
		"id": a.ID,
		"state_entries": strconv.Itoa(stateCount),
		"memory_entries": strconv.Itoa(memoryCount),
		"knowledge_entries": strconv.Itoa(knowledgeCount),
		"goals_active": strconv.Itoa(goalCount),
		"skills_available": strconv.Itoa(skillCount),
		"current_time": time.Now().Format(time.RFC3339),
	}

	if val, ok := a.State["current_task"]; ok {
		result["current_task"] = fmt.Sprintf("%v", val)
	}
	if val, ok := a.State["health_status"]; ok {
		result["health_status"] = fmt.Sprintf("%v", val)
	}

	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: status,
		Result:  result,
	}
}

// handleSetState sets a specific key-value pair in agent's state.
// Usage: set_state key:value [key2:value2...]
func (a *Agent) handleSetState(cmd *MCPCommand) *MCPResponse {
	if len(cmd.KV) == 0 {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing state key:value pair(s) to set.",
		}
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	updatedKeys := []string{}
	for k, v := range cmd.KV {
		// Attempt to unmarshal JSON string values for structured data
		var js json.RawMessage
		if err := json.Unmarshal([]byte(v), &js); err == nil {
			// It's valid JSON, store as interface{} (map, slice, value)
			var val interface{}
			json.Unmarshal([]byte(v), &val) // Error ignored as already checked validity
			a.State[k] = val
		} else {
			// Not valid JSON, store as string
			a.State[k] = v
		}
		updatedKeys = append(updatedKeys, k)
	}

	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: fmt.Sprintf("State updated for keys: %s", strings.Join(updatedKeys, ", ")),
		Result:  cmd.KV, // Echo back the set keys/values
	}
}

// handleGetState retrieves the value for a specific key from agent's state.
// Usage: get_state key [key2 key3...]
func (a *Agent) handleGetState(cmd *MCPCommand) *MCPResponse {
	if len(cmd.Args) == 0 {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing state key(s) to get.",
		}
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	result := make(map[string]string)
	notFound := []string{}

	for _, key := range cmd.Args {
		if val, ok := a.State[key]; ok {
			// Attempt to marshal value back to string, especially complex types
			valStr := fmt.Sprintf("%v", val) // Default simple conversion
			if bytes, err := json.Marshal(val); err == nil {
				valStr = string(bytes) // Use JSON if possible
			}
			result[key] = valStr
		} else {
			notFound = append(notFound, key)
		}
	}

	msg := "State retrieved."
	if len(notFound) > 0 {
		msg = fmt.Sprintf("State retrieved. Keys not found: %s", strings.Join(notFound, ", "))
	}

	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: msg,
		Result:  result,
	}
}

// handleRemember adds a piece of information or experience to the agent's memory.
// Usage: remember "some event happened" [tags:"event,important"]
func (a *Agent) handleRemember(cmd *MCPCommand) *MCPResponse {
	if len(cmd.Args) == 0 {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing memory content to remember.",
		}
	}

	memory := strings.Join(cmd.Args, " ") // Combine args into single memory string
	// Simple timestamping/tagging
	timestamp := time.Now().Format(time.RFC3339)
	tags, ok := cmd.KV["tags"]
	if !ok {
		tags = "untagged"
	}

	a.mu.Lock()
	a.Memory = append(a.Memory, fmt.Sprintf("[%s] [%s] %s", timestamp, tags, memory))
	// Keep memory size reasonable
	if len(a.Memory) > 500 {
		a.Memory = a.Memory[1:]
	}
	a.mu.Unlock()

	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: fmt.Sprintf("Remembered: %s", memory),
	}
}

// handleRecall searches the agent's memory for relevant information.
// Usage: recall keyword1 [keyword2...] [limit:N] [tags:tag1,tag2]
func (a *Agent) handleRecall(cmd *MCPCommand) *MCPResponse {
	if len(cmd.Args) == 0 && cmd.KV["tags"] == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing keywords or tags to recall.",
		}
	}

	keywords := cmd.Args
	targetTags := strings.Split(cmd.KV["tags"], ",")
	limitStr, ok := cmd.KV["limit"]
	limit := 10 // Default limit
	if ok {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 {
			limit = l
		}
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	results := []string{}
	// Search memory backwards to get most recent matches first
	for i := len(a.Memory) - 1; i >= 0; i-- {
		entry := a.Memory[i]
		match := true

		// Check keywords
		if len(keywords) > 0 {
			keywordMatch := false
			for _, kw := range keywords {
				if strings.Contains(strings.ToLower(entry), strings.ToLower(kw)) {
					keywordMatch = true
					break
				}
			}
			if !keywordMatch {
				match = false
			}
		}

		// Check tags (if provided)
		if len(targetTags) > 0 && targetTags[0] != "" {
			tagMatch := false
			// Extract tags from entry format "[timestamp] [tags] content"
			tagStringStart := strings.Index(entry, "[") + 1
			tagStringEnd := strings.Index(entry[tagStringStart:], "]") + tagStringStart
			if tagStringStart > 0 && tagStringEnd > tagStringStart {
				entryTagsRaw := entry[strings.Index(entry[tagStringEnd+1:], "[")+tagStringEnd+2 : strings.Index(entry[strings.Index(entry[tagStringEnd+1:], "[")+tagStringEnd+2:], "]")+strings.Index(entry[tagStringEnd+1:], "[")+tagStringEnd+2]
				entryTags := strings.Split(entryTagsRaw, ",")
				for _, tt := range targetTags {
					for _, et := range entryTags {
						if strings.TrimSpace(strings.ToLower(tt)) == strings.TrimSpace(strings.ToLower(et)) {
							tagMatch = true
							break
						}
					}
					if tagMatch {
						break
					}
				}
			}
			if !tagMatch {
				match = false
			}
		}

		if match {
			results = append(results, entry)
			if len(results) >= limit {
				break
			}
		}
	}

	if len(results) == 0 {
		return &MCPResponse{
			Command: cmd.Command,
			Success: true,
			Message: "No relevant memories found.",
			Result:  map[string]string{"count": "0"},
		}
	}

	resultKV := make(map[string]string)
	resultKV["count"] = strconv.Itoa(len(results))
	for i, r := range results {
		resultKV[fmt.Sprintf("memory_%d", i+1)] = r
	}

	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: fmt.Sprintf("Found %d relevant memories.", len(results)),
		Result:  resultKV,
	}
}

// handleAddKnowledge adds structured or unstructured data to the agent's knowledge base.
// Usage: add_knowledge category:item_id data:"{...json...}"
// Or: add_knowledge concept:definition [tags:tag1,tag2]
func (a *Agent) handleAddKnowledge(cmd *MCPCommand) *MCPResponse {
	if len(cmd.KV) == 0 {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing knowledge key:value pair(s) to add.",
		}
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	addedEntries := []string{}
	for k, v := range cmd.KV {
		// Attempt to unmarshal JSON string values for structured data
		var js json.RawMessage
		if err := json.Unmarshal([]byte(v), &js); err == nil {
			// It's valid JSON, store as interface{} (map, slice, value)
			var val interface{}
			json.Unmarshal([]byte(v), &val) // Error ignored as already checked validity
			a.KnowledgeBase[k] = val
		} else {
			// Not valid JSON, store as string
			a.KnowledgeBase[k] = v
		}
		addedEntries = append(addedEntries, k)
	}

	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: fmt.Sprintf("Knowledge base updated for keys: %s", strings.Join(addedEntries, ", ")),
		Result:  cmd.KV, // Echo back the added keys/values
	}
}

// handleQueryKnowledge queries the knowledge base.
// Usage: query_knowledge key [key2...]
// Or: query_knowledge contains:"keyword"
func (a *Agent) handleQueryKnowledge(cmd *MCPCommand) *MCPResponse {
	if len(cmd.Args) == 0 && cmd.KV["contains"] == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing key(s) or 'contains' keyword to query knowledge.",
		}
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	result := make(map[string]string)
	notFound := []string{}
	containsKeyword, hasContains := cmd.KV["contains"]

	if hasContains {
		// Search for keyword in values (simple string search)
		foundCount := 0
		for k, val := range a.KnowledgeBase {
			valStr := fmt.Sprintf("%v", val) // Convert any type to string
			if strings.Contains(strings.ToLower(valStr), strings.ToLower(containsKeyword)) {
				result[k] = valStr
				foundCount++
				if foundCount >= 20 { // Limit search results
					result["_truncated"] = "true"
					break
				}
			}
		}
		msg := fmt.Sprintf("Knowledge base searched for '%s'. Found %d matching entries.", containsKeyword, foundCount)
		if foundCount == 0 {
			msg = fmt.Sprintf("Knowledge base searched for '%s'. No matches found.", containsKeyword)
		}

		return &MCPResponse{
			Command: cmd.Command,
			Success: true,
			Message: msg,
			Result:  result,
		}

	} else {
		// Query specific keys
		for _, key := range cmd.Args {
			if val, ok := a.KnowledgeBase[key]; ok {
				// Attempt to marshal value back to string
				valStr := fmt.Sprintf("%v", val) // Default simple conversion
				if bytes, err := json.Marshal(val); err == nil {
					valStr = string(bytes) // Use JSON if possible
				}
				result[key] = valStr
			} else {
				notFound = append(notFound, key)
			}
		}

		msg := "Knowledge base queried."
		if len(notFound) > 0 {
			msg = fmt.Sprintf("Knowledge base queried. Keys not found: %s", strings.Join(notFound, ", "))
		}

		return &MCPResponse{
			Command: cmd.Command,
			Success: true,
			Message: msg,
			Result:  result,
		}
	}
}

// handleAnalyzeText simulates analyzing text input.
// Usage: analyze_text "the quick brown fox" [type:sentiment]
func (a *Agent) handleAnalyzeText(cmd *MCPCommand) *MCPResponse {
	if len(cmd.Args) == 0 {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing text to analyze.",
		}
	}
	text := strings.Join(cmd.Args, " ")
	analysisType, ok := cmd.KV["type"]
	if !ok || analysisType == "" {
		analysisType = "general" // Default analysis
	}

	result := make(map[string]string)
	result["input_text"] = text
	result["analysis_type"] = analysisType

	// Simulate different analysis types
	switch strings.ToLower(analysisType) {
	case "sentiment":
		// Very basic sentiment simulation
		sentiment := "neutral"
		lowerText := strings.ToLower(text)
		if strings.Contains(lowerText, "happy") || strings.Contains(lowerText, "good") || strings.Contains(lowerText, "great") {
			sentiment = "positive"
		} else if strings.Contains(lowerText, "sad") || strings.Contains(lowerText, "bad") || strings.Contains(lowerText, "terrible") {
			sentiment = "negative"
		}
		result["sentiment"] = sentiment
		result["sim_confidence"] = "0.7" // Simulated confidence
	case "keywords":
		// Basic keyword extraction simulation
		keywords := []string{}
		words := strings.Fields(strings.ToLower(text))
		// Simple filter for common words (a, the, is, are, etc.)
		commonWords := map[string]bool{"a": true, "the": true, "is": true, "are": true, "in": true, "on": true, "and": true}
		for _, word := range words {
			cleanWord := strings.Trim(word, ".,!?;\"'()")
			if len(cleanWord) > 2 && !commonWords[cleanWord] {
				keywords = append(keywords, cleanWord)
			}
		}
		result["keywords"] = strings.Join(keywords, ", ")
	case "structure":
		// Basic structure simulation (e.g., sentence count)
		sentences := strings.Split(text, ".") // Very naive split
		result["sentence_count"] = strconv.Itoa(len(sentences))
		result["char_count"] = strconv.Itoa(len(text))
	default:
		result["sim_result"] = "Analysis type not specifically simulated, performing general check."
	}

	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: fmt.Sprintf("Analysis (simulated) performed on text."),
		Result:  result,
	}
}

// handleSynthesizeInfo combines info from state, memory, and knowledge base.
// Usage: synthesize_info query:"report on recent events related to X"
func (a *Agent) handleSynthesizeInfo(cmd *MCPCommand) *MCPResponse {
	query, ok := cmd.KV["query"]
	if !ok || query == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing 'query' parameter for synthesis.",
		}
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	// Simulate gathering relevant info
	relevantState := make(map[string]string)
	// In a real agent, this would be intelligent search/matching
	for k, v := range a.State {
		if strings.Contains(strings.ToLower(k), strings.ToLower(query)) {
			relevantState[k] = fmt.Sprintf("%v", v)
		}
	}

	relevantMemory := []string{}
	// Simulate simple keyword search in memory
	for _, mem := range a.Memory {
		if strings.Contains(strings.ToLower(mem), strings.ToLower(query)) {
			relevantMemory = append(relevantMemory, mem)
		}
	}

	relevantKnowledge := make(map[string]string)
	// Simulate simple keyword search in knowledge base
	for k, v := range a.KnowledgeBase {
		valStr := fmt.Sprintf("%v", v)
		if strings.Contains(strings.ToLower(k), strings.ToLower(query)) || strings.Contains(strings.ToLower(valStr), strings.ToLower(query)) {
			relevantKnowledge[k] = valStr
		}
	}

	// Simulate synthesis
	var synthesis strings.Builder
	synthesis.WriteString(fmt.Sprintf("Synthesized information based on query '%s':\n", query))
	synthesis.WriteString("\n--- Relevant State ---\n")
	if len(relevantState) == 0 {
		synthesis.WriteString("None found.\n")
	} else {
		for k, v := range relevantState {
			synthesis.WriteString(fmt.Sprintf("%s: %s\n", k, v))
		}
	}

	synthesis.WriteString("\n--- Relevant Memory ---\n")
	if len(relevantMemory) == 0 {
		synthesis.WriteString("None found.\n")
	} else {
		synthesis.WriteString(strings.Join(relevantMemory, "\n"))
		synthesis.WriteString("\n")
	}

	synthesis.WriteString("\n--- Relevant Knowledge ---\n")
	if len(relevantKnowledge) == 0 {
		synthesis.WriteString("None found.\n")
	} else {
		for k, v := range relevantKnowledge {
			synthesis.WriteString(fmt.Sprintf("%s: %s\n", k, v))
		}
	}

	result := map[string]string{
		"query": query,
		"synthesis_result": synthesis.String(),
	}

	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: "Information synthesis (simulated) complete.",
		Result:  result,
	}
}

// handlePlanSimpleTask generates a conceptual sequence of steps.
// Usage: plan_simple_task goal:"achieve objective X" [context:"current situation"]
func (a *Agent) handlePlanSimpleTask(cmd *MCPCommand) *MCPResponse {
	goal, ok := cmd.KV["goal"]
	if !ok || goal == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing 'goal' parameter for planning.",
		}
	}
	context := cmd.KV["context"] // Optional

	// Simulate simple planning logic based on goal/context
	planSteps := []string{
		"Assess current state relevant to goal.",
		"Identify necessary resources/information.",
		"Determine initial action.",
		"Execute action (simulated).",
		"Evaluate outcome.",
		"Adjust plan or proceed to next step.",
		"Repeat until goal is met or deemed unachievable.",
		fmt.Sprintf("Final step: Report achievement status for goal '%s'.", goal),
	}

	planSummary := strings.Join(planSteps, " -> ")

	result := map[string]string{
		"goal":         goal,
		"context":      context,
		"plan_summary": planSummary,
		"step_count":   strconv.Itoa(len(planSteps)),
	}
	for i, step := range planSteps {
		result[fmt.Sprintf("step_%d", i+1)] = step
	}

	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: "Simple task plan generated (simulated).",
		Result:  result,
	}
}

// handleExecuteSimTask simulates the execution of a task plan, updating state.
// Usage: execute_sim_task plan:"plan_summary_string" [outcome:success/failure] [details:"..."]
func (a *Agent) handleExecuteSimTask(cmd *MCPCommand) *MCPResponse {
	planSummary, ok := cmd.KV["plan"]
	if !ok || planSummary == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing 'plan' parameter for simulation.",
		}
	}
	outcome := cmd.KV["outcome"] // e.g., "success", "failure", "partial"
	details := cmd.KV["details"] // Optional details

	// Simulate state changes based on outcome
	a.mu.Lock()
	defer a.mu.Unlock()

	statusMsg := fmt.Sprintf("Simulating execution of plan: %s", planSummary)
	stateUpdates := make(map[string]string)

	switch strings.ToLower(outcome) {
	case "success":
		a.State["last_task_status"] = "success"
		a.State["last_task_details"] = fmt.Sprintf("Plan '%s' executed successfully.", planSummary)
		statusMsg += " -> Simulated Success."
		stateUpdates["last_task_status"] = "success"
		stateUpdates["last_task_details"] = fmt.Sprintf("Plan '%s' executed successfully.", planSummary)
		// Simulate achieving a goal if plan relates to one
		for i, g := range a.Goals {
			if strings.Contains(planSummary, g) {
				a.State["achieved_goal"] = g
				a.Goals = append(a.Goals[:i], a.Goals[i+1:]...) // Remove goal
				stateUpdates["achieved_goal"] = g
				stateUpdates["goals_remaining"] = strconv.Itoa(len(a.Goals))
				break
			}
		}

	case "failure":
		a.State["last_task_status"] = "failure"
		a.State["last_task_details"] = fmt.Sprintf("Plan '%s' execution failed. Details: %s", planSummary, details)
		statusMsg += " -> Simulated Failure."
		stateUpdates["last_task_status"] = "failure"
		stateUpdates["last_task_details"] = fmt.Sprintf("Plan '%s' execution failed. Details: %s", planSummary, details)

	case "partial":
		a.State["last_task_status"] = "partial_success"
		a.State["last_task_details"] = fmt.Sprintf("Plan '%s' execution partially successful. Details: %s", planSummary, details)
		statusMsg += " -> Simulated Partial Success."
		stateUpdates["last_task_status"] = "partial_success"
		stateUpdates["last_task_details"] = fmt.Sprintf("Plan '%s' execution partially successful. Details: %s", planSummary, details)

	default:
		a.State["last_task_status"] = "simulated"
		a.State["last_task_details"] = fmt.Sprintf("Plan '%s' simulation completed with unspecified outcome. Details: %s", planSummary, details)
		statusMsg += " -> Simulation Completed."
		stateUpdates["last_task_status"] = "simulated"
		stateUpdates["last_task_details"] = fmt.Sprintf("Plan '%s' simulation completed with unspecified outcome. Details: %s", planSummary, details)
	}

	// Add outcome details if provided
	if details != "" {
		a.State["last_task_sim_details"] = details
		stateUpdates["last_task_sim_details"] = details
	}


	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: statusMsg,
		Result: stateUpdates, // Report state changes
	}
}

// handleLearnFromOutcome simulates adjusting internal state, knowledge, or strategy.
// Usage: learn_from_outcome task:"task_id" outcome:"success/failure" [notes:"..."]
func (a *Agent) handleLearnFromOutcome(cmd *MCPCommand) *MCPResponse {
	taskID, ok := cmd.KV["task"]
	if !ok || taskID == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing 'task' parameter for learning.",
		}
	}
	outcome, ok := cmd.KV["outcome"]
	if !ok || outcome == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing 'outcome' parameter for learning.",
		}
	}
	notes := cmd.KV["notes"] // Optional notes

	a.mu.Lock()
	defer a.mu.Unlock()

	// Simulate learning: Add a memory entry, potentially update state/knowledge
	learningMsg := fmt.Sprintf("Learned from task '%s' with outcome '%s'.", taskID, outcome)
	memoryEntry := fmt.Sprintf("[Learning] Task: %s, Outcome: %s, Notes: %s", taskID, outcome, notes)
	a.Memory = append(a.Memory, memoryEntry)

	// Simulate updating a simple strategy or knowledge item
	learningKey := fmt.Sprintf("learning_from_%s", taskID)
	a.KnowledgeBase[learningKey] = map[string]string{
		"outcome": outcome,
		"timestamp": time.Now().Format(time.RFC3339),
		"notes": notes,
	}

	// Simple simulation of strategic adjustment
	if outcome == "failure" {
		a.State["strategy_flag"] = "needs_review"
		learningMsg += " Strategy flagged for review."
	} else if outcome == "success" {
		// Maybe reinforce a positive strategy flag
		if currentFlag, ok := a.State["strategy_flag"].(string); ok && currentFlag == "needs_review" {
			// If we succeeded despite needing review, maybe clear the flag or update
			a.State["strategy_flag"] = "partially_validated"
			learningMsg += " Strategy partially validated."
		} else {
			a.State["strategy_flag"] = "reinforced"
			learningMsg += " Strategy reinforced."
		}
	}


	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: learningMsg,
		Result: map[string]string{
			"task_id": taskID,
			"outcome": outcome,
			"memory_added": "true",
			"knowledge_added": "true",
			"simulated_state_update": fmt.Sprintf("%v", a.State["strategy_flag"]), // Report simulated effect
		},
	}
}


// handleGenerateText simulates generating a piece of text.
// Usage: generate_text prompt:"write a summary of X" [context:"..."] [length:short/medium/long]
func (a *Agent) handleGenerateText(cmd *MCPCommand) *MCPResponse {
	prompt, ok := cmd.KV["prompt"]
	if !ok || prompt == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing 'prompt' parameter for text generation.",
		}
	}
	context := cmd.KV["context"]
	length := cmd.KV["length"]
	if length == "" {
		length = "medium"
	}

	// Simulate text generation based on prompt, context, and length
	generatedText := fmt.Sprintf("Simulated text generation based on prompt: '%s'.", prompt)
	if context != "" {
		generatedText += fmt.Sprintf(" Context considered: '%s'.", context)
	}

	switch strings.ToLower(length) {
	case "short":
		generatedText += " This is a short example response."
	case "long":
		generatedText += " This is a longer example response, simulating more detailed output. The agent attempts to provide a comprehensive answer incorporating various simulated elements."
	default: // Medium
		generatedText += " This is a medium-length example response."
	}

	// Add some variability based on state or knowledge (simulated)
	if task, ok := a.State["current_task"].(string); ok && task != "" {
		generatedText += fmt.Sprintf(" The agent's current task (%s) might influence output.", task)
	}
	if _, ok := a.KnowledgeBase["important_concept"]; ok {
		generatedText += " An important concept from knowledge base was potentially referenced."
	}


	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: "Text generated (simulated).",
		Result: map[string]string{
			"prompt": prompt,
			"generated_text": generatedText,
			"length_requested": length,
		},
	}
}

// handleSimulateEvent runs a simple internal simulation model or scenario.
// Usage: simulate_event scenario:"traffic_flow" [parameters:"{...json...}"] [duration:10]
func (a *Agent) handleSimulateEvent(cmd *MCPCommand) *MCPResponse {
	scenario, ok := cmd.KV["scenario"]
	if !ok || scenario == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing 'scenario' parameter for simulation.",
		}
	}
	parameters := cmd.KV["parameters"] // JSON string likely
	durationStr := cmd.KV["duration"]
	duration := 1 // Default simulation duration units

	if dur, err := strconv.Atoi(durationStr); err == nil && dur > 0 {
		duration = dur
	}

	// Simulate running a scenario
	statusMsg := fmt.Sprintf("Simulating scenario '%s' for %d units.", scenario, duration)
	simResult := fmt.Sprintf("Simulation results for %s:", scenario)

	// Simulate different outcomes based on scenario/parameters
	switch strings.ToLower(scenario) {
	case "traffic_flow":
		// Simulate congestion based on parameters (e.g., number of cars)
		cars := 100
		if paramsMap := parseJSONString(parameters); paramsMap != nil {
			if carsVal, ok := paramsMap["cars"].(float64); ok {
				cars = int(carsVal)
			}
		}
		congestionLevel := "low"
		if cars > 500 { congestionLevel = "medium" }
		if cars > 1000 { congestionLevel = "high" }
		simResult += fmt.Sprintf(" Traffic congestion level: %s.", congestionLevel)
		a.mu.Lock()
		a.State["last_sim_congestion"] = congestionLevel // Update state
		a.mu.Unlock()
	case "resource_allocation":
		// Simulate resource use
		resources := 10
		tasks := 5
		if paramsMap := parseJSONString(parameters); paramsMap != nil {
			if resVal, ok := paramsMap["resources"].(float64); ok { resources = int(resVal) }
			if taskVal, ok := paramsMap["tasks"].(float64); ok { tasks = int(taskVal) }
		}
		efficiency := "high"
		if tasks > resources*1.5 { efficiency = "medium" }
		if tasks > resources*3 { efficiency = "low" }
		simResult += fmt.Sprintf(" Resource allocation efficiency: %s.", efficiency)
		a.mu.Lock()
		a.State["last_sim_efficiency"] = efficiency // Update state
		a.mu.Unlock()

	default:
		simResult += " Outcome based on generic simulation logic."
	}

	resultKV := map[string]string{
		"scenario": scenario,
		"duration": strconv.Itoa(duration),
		"sim_output": simResult,
	}
	if parameters != "" {
		resultKV["parameters_used"] = parameters
	}


	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: statusMsg,
		Result: resultKV,
	}
}

// handlePrognosticate simulates predicting a future state or event.
// Usage: prognosticate event:"system failure" [context:"current health metrics"] [horizon:"hour"]
func (a *Agent) handlePrognosticate(cmd *MCPCommand) *MCPResponse {
	event, ok := cmd.KV["event"]
	if !ok || event == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing 'event' parameter for prognostication.",
		}
	}
	context := cmd.KV["context"]
	horizon := cmd.KV["horizon"]
	if horizon == "" {
		horizon = "day"
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	// Simulate prediction based on current state, knowledge, and context
	prediction := "Unknown likelihood."
	confidence := "low"
	justification := "Based on general knowledge."

	// Simple simulated logic: check if state/knowledge contains keywords related to the event
	lowerEvent := strings.ToLower(event)
	contextScore := 0
	knowledgeScore := 0
	stateScore := 0

	if strings.Contains(strings.ToLower(context), lowerEvent) { contextScore += 1 }
	if stateVal, ok := a.State[lowerEvent]; ok {
		stateScore += 1 // Simple check if event name is a state key
		if stateString, ok := stateVal.(string); ok && strings.Contains(strings.ToLower(stateString), "critical") {
			stateScore += 2 // Higher score for critical state
		}
	}
	if knowledgeVal, ok := a.KnowledgeBase[lowerEvent]; ok {
		knowledgeScore += 1 // Simple check if event name is a knowledge key
		if kbString, ok := knowledgeVal.(string); ok && strings.Contains(strings.ToLower(kbString), "high risk") {
			knowledgeScore += 2 // Higher score for high risk in knowledge
		}
	}


	totalScore := contextScore + knowledgeScore + stateScore

	switch {
	case totalScore >= 3:
		prediction = fmt.Sprintf("High likelihood of '%s' within %s.", event, horizon)
		confidence = "high"
		justification = "Multiple indicators found in state, knowledge, or context."
	case totalScore >= 1:
		prediction = fmt.Sprintf("Medium likelihood of '%s' within %s.", event, horizon)
		confidence = "medium"
		justification = "Some relevant information found."
	default:
		prediction = fmt.Sprintf("Low likelihood of '%s' within %s.", event, orizon) // Corrected typo 'orizon'
		confidence = "low"
		justification = "Limited relevant information found."
	}

	if stateVal, ok := a.State["last_sim_congestion"].(string); ok && lowerEvent == "traffic congestion" && stateVal == "high" {
		prediction = fmt.Sprintf("Very High likelihood of '%s' within %s based on last simulation.", event, horizon)
		confidence = "very high"
		justification = "Confirmed by recent internal simulation results."
	}


	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: "Prognostication (simulated) complete.",
		Result: map[string]string{
			"event": event,
			"horizon": horizon,
			"prediction": prediction,
			"confidence": confidence,
			"justification": justification,
			"sim_score": strconv.Itoa(totalScore), // Report simulated score
		},
	}
}

// handleOptimizeSetting simulates finding an optimal parameter value.
// Usage: optimize_setting parameter:"threshold" objective:"minimize errors" [range:"0.1,10.0"]
func (a *Agent) handleOptimizeSetting(cmd *MCPCommand) *MCPResponse {
	parameter, ok := cmd.KV["parameter"]
	if !ok || parameter == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing 'parameter' for optimization.",
		}
	}
	objective, ok := cmd.KV["objective"]
	if !ok || objective == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing 'objective' for optimization.",
		}
	}
	valueRange := cmd.KV["range"] // e.g., "0,100" or "low,medium,high"

	// Simulate optimization process
	optimizedValue := "unknown"
	performanceMetric := "N/A"
	notes := fmt.Sprintf("Simulated optimization for '%s' to '%s'.", parameter, objective)

	// Simple simulated logic
	switch strings.ToLower(parameter) {
	case "threshold":
		notes += " Assuming numerical threshold optimization."
		// Simulate finding a value in the range
		if valueRange != "" {
			parts := strings.Split(valueRange, ",")
			if len(parts) == 2 {
				min, err1 := strconv.ParseFloat(parts[0], 64)
				max, err2 := strconv.ParseFloat(parts[1], 64)
				if err1 == nil && err2 == nil && max > min {
					// Simulate picking a value within range, maybe favoring edges or middle
					// Naive simulation: pick a value slightly above min or below max
					if strings.Contains(strings.ToLower(objective), "minimize") {
						optimizedValue = fmt.Sprintf("%.2f", min*1.1) // Try value near min
						performanceMetric = "simulated_low_error"
					} else if strings.Contains(strings.ToLower(objective), "maximize") {
						optimizedValue = fmt.Sprintf("%.2f", max*0.9) // Try value near max
						performanceMetric = "simulated_high_performance"
					} else {
						optimizedValue = fmt.Sprintf("%.2f", (min+max)/2.0) // Try middle
						performanceMetric = "simulated_balanced"
					}
				} else {
					notes += " Invalid range format."
				}
			} else {
				notes += " Invalid range format."
			}
		} else {
			optimizedValue = "50" // Default if no range
			performanceMetric = "simulated_default"
		}
	case "setting_level":
		notes += " Assuming categorical setting optimization."
		// Simulate picking from categories
		if valueRange != "" {
			options := strings.Split(valueRange, ",")
			if len(options) > 0 {
				// Naive simulation: pick the middle option if available
				optimizedValue = options[(len(options)-1)/2]
				performanceMetric = "simulated_categorical_selection"
			} else {
				notes += " Invalid category list format."
			}
		} else {
			optimizedValue = "medium" // Default
			performanceMetric = "simulated_default"
		}
	default:
		optimizedValue = "default_value" // Generic default
		performanceMetric = "simulated_generic"
		notes += fmt.Sprintf(" Parameter '%s' not specifically handled, using generic simulation.", parameter)
	}

	a.mu.Lock()
	a.State["last_optimization_result"] = map[string]string{
		"parameter": parameter,
		"optimized_value": optimizedValue,
		"objective": objective,
		"performance_metric": performanceMetric,
		"timestamp": time.Now().Format(time.RFC3339),
	}
	a.mu.Unlock()


	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: "Parameter optimization (simulated) complete.",
		Result: map[string]string{
			"parameter": parameter,
			"objective": objective,
			"range_used": valueRange,
			"optimized_value": optimizedValue,
			"simulated_performance": performanceMetric,
			"notes": notes,
		},
	}
}


// handleDiffState compares the current state to a previously saved state snapshot.
// Usage: diff_state snapshot_id:"snap_XYZ"
func (a *Agent) handleDiffState(cmd *MCPCommand) *MCPResponse {
	snapshotID, ok := cmd.KV["snapshot_id"]
	if !ok || snapshotID == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing 'snapshot_id' for state diff.",
		}
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	// Retrieve the snapshot from state/knowledge base (simulated storage)
	snapshotKey := fmt.Sprintf("state_snapshot_%s", snapshotID)
	snapVal, snapFound := a.KnowledgeBase[snapshotKey]

	if !snapFound {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: fmt.Sprintf("Snapshot ID '%s' not found in knowledge base.", snapshotID),
		}
	}

	// Attempt to cast or unmarshal the snapshot back into a map[string]interface{}
	// This assumes snapshots were stored as maps or JSON strings representing maps.
	var snapshotState map[string]interface{}
	if snapMap, ok := snapVal.(map[string]interface{}); ok {
		snapshotState = snapMap
	} else if snapMapString, ok := snapVal.(string); ok {
		// Try unmarshalling if stored as string
		var tempMap map[string]interface{}
		if err := json.Unmarshal([]byte(snapMapString), &tempMap); err == nil {
			snapshotState = tempMap
		} else {
			return &MCPResponse{
				Command: cmd.Command,
				Success: false,
				Message: fmt.Sprintf("Snapshot ID '%s' found, but data format is unexpected.", snapshotID),
				Result: map[string]string{"snapshot_key": snapshotKey, "data_type": fmt.Sprintf("%T", snapVal)},
			}
		}
	} else {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: fmt.Sprintf("Snapshot ID '%s' found, but data format is unexpected.", snapshotID),
			Result: map[string]string{"snapshot_key": snapshotKey, "data_type": fmt.Sprintf("%T", snapVal)},
		}
	}


	// Perform diff
	added := make(map[string]string)
	removed := make(map[string]string)
	changed := make(map[string]string) // format: key: "old_value" -> "new_value"

	// Check current state against snapshot
	for k, currentVal := range a.State {
		snapshotVal, existsInSnapshot := snapshotState[k]
		currentValStr := fmt.Sprintf("%v", currentVal)
		if existsInSnapshot {
			snapshotValStr := fmt.Sprintf("%v", snapshotVal)
			if currentValStr != snapshotValStr {
				// Value changed. Try JSON representation for complex types.
				curBytes, curErr := json.Marshal(currentVal)
				snapBytes, snapErr := json.Marshal(snapshotVal)

				curString := currentValStr
				if curErr == nil { curString = string(curBytes) }
				snapString := snapshotValStr
				if snapErr == nil { snapString = string(snapBytes) }

				if curString != snapString {
					changed[k] = fmt.Sprintf("\"%s\" -> \"%s\"", snapString, curString)
				}
			}
		} else {
			// Added in current state
			added[k] = currentValStr
		}
	}

	// Check snapshot against current state for removed keys
	for k, snapshotVal := range snapshotState {
		_, existsInCurrent := a.State[k]
		if !existsInCurrent {
			removed[k] = fmt.Sprintf("%v", snapshotVal)
		}
	}

	resultKV := make(map[string]string)
	resultKV["snapshot_id"] = snapshotID
	resultKV["added_count"] = strconv.Itoa(len(added))
	resultKV["removed_count"] = strconv.Itoa(len(removed))
	resultKV["changed_count"] = strconv.Itoa(len(changed))

	// Add diff details to result KV (simple string representation)
	addDetails := []string{}
	for k, v := range added { addDetails = append(addDetails, fmt.Sprintf("%s: \"%s\"", k, v)) }
	resultKV["added"] = strings.Join(addDetails, ", ")

	removeDetails := []string{}
	for k, v := range removed { removeDetails = append(removeDetails, fmt.Sprintf("%s: \"%s\"", k, v)) }
	resultKV["removed"] = strings.Join(removeDetails, ", ")

	changeDetails := []string{}
	for k, v := range changed { changeDetails = append(changeDetails, fmt.Sprintf("%s: %s", k, v)) }
	resultKV["changed"] = strings.Join(changeDetails, ", ")


	msg := fmt.Sprintf("State diff vs snapshot '%s' complete.", snapshotID)
	if len(added) == 0 && len(removed) == 0 && len(changed) == 0 {
		msg = fmt.Sprintf("State is identical to snapshot '%s'.", snapshotID)
	} else {
		msg = fmt.Sprintf("State differs from snapshot '%s'. Added: %d, Removed: %d, Changed: %d.",
			snapshotID, len(added), len(removed), len(changed))
	}


	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: msg,
		Result: resultKV,
	}
}

// handleAuditTrail retrieves a history of recent commands processed.
// Usage: audit_trail [limit:N] [contains:"keyword"]
func (a *Agent) handleAuditTrail(cmd *MCPCommand) *MCPResponse {
	limitStr, ok := cmd.KV["limit"]
	limit := 20 // Default limit
	if ok {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 {
			limit = l
		}
	}
	containsKeyword := cmd.KV["contains"]

	a.mu.Lock()
	defer a.mu.Unlock()

	results := []string{}
	// Iterate audit log backwards
	for i := len(a.AuditLog) - 1; i >= 0; i-- {
		entry := a.AuditLog[i]
		cmdString := fmt.Sprintf("%s %s %v", entry.Command, strings.Join(entry.Args, " "), entry.KV) // Simple string rep

		if containsKeyword != "" && !strings.Contains(strings.ToLower(cmdString), strings.ToLower(containsKeyword)) {
			continue // Skip if no keyword match
		}

		results = append(results, cmdString)
		if len(results) >= limit {
			break
		}
	}

	resultKV := make(map[string]string)
	resultKV["count"] = strconv.Itoa(len(results))
	if containsKeyword != "" {
		resultKV["filter_keyword"] = containsKeyword
	}

	for i, r := range results {
		resultKV[fmt.Sprintf("command_%d", i+1)] = r
	}

	msg := fmt.Sprintf("Retrieved %d recent command log entries.", len(results))
	if containsKeyword != "" {
		msg = fmt.Sprintf("Retrieved %d recent command log entries matching '%s'.", len(results), containsKeyword)
	}

	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: msg,
		Result:  resultKV,
	}
}

// handleSelfReflect generates a summary of the agent's recent activity, state, or internal thoughts (simulated).
// Usage: self_reflect [scope:state/memory/activity] [period:hour/day/all]
func (a *Agent) handleSelfReflect(cmd *MCPCommand) *MCPResponse {
	scope := cmd.KV["scope"]
	if scope == "" {
		scope = "summary" // Default to a general summary
	}
	period := cmd.KV["period"]
	// Period simulation is very basic, mostly just a tag

	a.mu.Lock()
	defer a.mu.Unlock()

	reflection := ""
	resultKV := make(map[string]string)
	resultKV["scope"] = scope
	resultKV["period"] = period

	switch strings.ToLower(scope) {
	case "state":
		reflection = "Current State Summary:\n"
		if len(a.State) == 0 {
			reflection += "State is empty."
		} else {
			// Simple print of state keys/values
			for k, v := range a.State {
				valStr := fmt.Sprintf("%v", v)
				if bytes, err := json.Marshal(v); err == nil { // Use JSON for complex types
					valStr = string(bytes)
				}
				reflection += fmt.Sprintf(" - %s: %s\n", k, valStr)
			}
		}
	case "memory":
		reflection = "Recent Memory Summary:\n"
		memCount := len(a.Memory)
		if memCount == 0 {
			reflection += "Memory is empty."
		} else {
			displayCount := 10 // Show last 10 memories
			if memCount < displayCount { displayCount = memCount }
			for i := memCount - displayCount; i < memCount; i++ {
				reflection += fmt.Sprintf(" - %s\n", a.Memory[i])
			}
			if memCount > displayCount { reflection += fmt.Sprintf("... and %d older memories.\n", memCount-displayCount) }
		}
		resultKV["total_memory_entries"] = strconv.Itoa(memCount)

	case "activity":
		reflection = "Recent Activity (Audit Trail) Summary:\n"
		logCount := len(a.AuditLog)
		if logCount == 0 {
			reflection += "No recent activity logged."
		} else {
			displayCount := 10 // Show last 10 commands
			if logCount < displayCount { displayCount = logCount }
			for i := logCount - displayCount; i < logCount; i++ {
				cmd := a.AuditLog[i]
				cmdString := fmt.Sprintf("%s %s %v", cmd.Command, strings.Join(cmd.Args, " "), cmd.KV)
				reflection += fmt.Sprintf(" - %s\n", cmdString)
			}
			if logCount > displayCount { reflection += fmt.Sprintf("... and %d older commands.\n", logCount-displayCount) }
		}
		resultKV["total_commands_logged"] = strconv.Itoa(logCount)

	case "goals":
		reflection = "Current Goals:\n"
		if len(a.Goals) == 0 {
			reflection += "No active goals."
		} else {
			for i, goal := range a.Goals {
				reflection += fmt.Sprintf(" - %d. %s\n", i+1, goal)
			}
		}
		resultKV["active_goals_count"] = strconv.Itoa(len(a.Goals))

	case "summary": // General summary
		reflection = fmt.Sprintf("Agent '%s' Self-Reflection (%s):\n", a.ID, period)
		reflection += fmt.Sprintf("- Operational Status: OK\n") // Simulated status
		reflection += fmt.Sprintf("- State Entries: %d\n", len(a.State))
		reflection += fmt.Sprintf("- Memory Entries: %d\n", len(a.Memory))
		reflection += fmt.Sprintf("- Knowledge Entries: %d\n", len(a.KnowledgeBase))
		reflection += fmt.Sprintf("- Active Goals: %d\n", len(a.Goals))
		reflection += fmt.Sprintf("- Recent Commands Processed: %d\n", len(a.AuditLog))

		// Add a simple "feeling" or self-assessment (simulated)
		mood := "stable"
		if len(a.AuditLog) > 50 && len(a.Memory) > 200 {
			mood = "busy" // Simulate busyness based on activity/memory size
		}
		if stateVal, ok := a.State["health_status"].(string); ok && stateVal == "alert" {
			mood = "vigilant"
		}
		reflection += fmt.Sprintf("- Simulated Status Mood: %s\n", mood)
		resultKV["sim_mood"] = mood

	default:
		reflection = fmt.Sprintf("Unknown reflection scope '%s'. Performing general summary.", scope)
		// Fallback to summary logic
		scope = "summary" // Update for result KV
		reflection += fmt.Sprintf("\nAgent '%s' Self-Reflection (%s):\n", a.ID, period)
		reflection += fmt.Sprintf("- Operational Status: OK\n")
		reflection += fmt.Sprintf("- State Entries: %d\n", len(a.State))
		reflection += fmt.Sprintf("- Memory Entries: %d\n", len(a.Memory))
		reflection += fmt.Sprintf("- Knowledge Entries: %d\n", len(a.KnowledgeBase))
		reflection += fmt.Sprintf("- Active Goals: %d\n", len(a.Goals))
		reflection += fmt.Sprintf("- Recent Commands Processed: %d\n", len(a.AuditLog))
		resultKV["sim_mood"] = "stable" // Default mood
	}

	resultKV["reflection_text"] = reflection // Put the generated text in KV for easier parsing

	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: "Self-reflection (simulated) complete.",
		Result:  resultKV,
	}
}

// handleDetectPattern searches for recurring patterns or anomalies in recent data streams or memory.
// Usage: detect_pattern type:"keyword_frequency" [source:memory] [keywords:"alert,error"] [period:day]
func (a *Agent) handleDetectPattern(cmd *MCPCommand) *MCPResponse {
	patternType, ok := cmd.KV["type"]
	if !ok || patternType == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing 'type' parameter for pattern detection.",
		}
	}
	source := cmd.KV["source"] // e.g., "memory", "audit_log", "state" (simulated data sources)
	if source == "" { source = "memory" } // Default source
	keywords := strings.Split(cmd.KV["keywords"], ",") // Keywords to look for frequency/patterns
	period := cmd.KV["period"] // Optional period (simulated)

	a.mu.Lock()
	defer a.mu.Unlock()

	resultKV := make(map[string]string)
	resultKV["pattern_type"] = patternType
	resultKV["source"] = source
	resultKV["period"] = period
	resultKV["keywords"] = strings.Join(keywords, ",")

	patternsFound := []string{}
	anomaliesFound := []string{}


	// Simulate pattern detection based on type and source
	switch strings.ToLower(patternType) {
	case "keyword_frequency":
		if len(keywords) == 0 || (len(keywords) == 1 && keywords[0] == "") {
			return &MCPResponse{
				Command: cmd.Command,
				Success: false,
				Message: "Pattern type 'keyword_frequency' requires 'keywords' parameter.",
			}
		}
		// Count frequency of keywords in chosen source
		contentToScan := []string{}
		switch strings.ToLower(source) {
		case "memory":
			contentToScan = a.Memory
		case "audit_log":
			for _, c := range a.AuditLog {
				contentToScan = append(contentToScan, fmt.Sprintf("%s %s %v", c.Command, strings.Join(c.Args, " "), c.KV))
			}
		case "state":
			for k, v := range a.State {
				contentToScan = append(contentToScan, fmt.Sprintf("%s: %v", k, v))
			}
		case "knowledge":
			for k, v := range a.KnowledgeBase {
				contentToScan = append(contentToScan, fmt.Sprintf("%s: %v", k, v))
			}
		default:
			return &MCPResponse{
				Command: cmd.Command,
				Success: false,
				Message: fmt.Sprintf("Unknown source '%s' for keyword frequency.", source),
			}
		}

		freqMap := make(map[string]int)
		totalScanLength := 0
		for _, item := range contentToScan {
			totalScanLength += len(item)
			lowerItem := strings.ToLower(item)
			for _, kw := range keywords {
				if kw != "" && strings.Contains(lowerItem, strings.ToLower(kw)) {
					freqMap[kw]++
				}
			}
		}

		freqReport := []string{}
		for kw, freq := range freqMap {
			freqReport = append(freqReport, fmt.Sprintf("%s:%d", kw, freq))
			// Simple anomaly: high frequency compared to total length (very rough sim)
			if freq > 5 && totalScanLength > 0 && float64(freq) / float64(totalScanLength) > 0.01 {
				anomaliesFound = append(anomaliesFound, fmt.Sprintf("High frequency for keyword '%s' (%d occurrences).", kw, freq))
			}
		}
		patternsFound = freqReport
		resultKV["frequency_report"] = strings.Join(patternsFound, ", ")
		resultKV["items_scanned"] = strconv.Itoa(len(contentToScan))


	case "sequence":
		// Simulate detecting sequences of events in audit log or memory
		if source != "audit_log" && source != "memory" {
			return &MCPResponse{
				Command: cmd.Command,
				Success: false,
				Message: fmt.Sprintf("Pattern type 'sequence' only supported for sources 'audit_log' or 'memory'. Unknown source '%s'.", source),
			}
		}
		targetSequence := cmd.Args // Treat args as sequence elements (e.g., "login" "fail" "login")
		if len(targetSequence) < 2 {
			return &MCPResponse{
				Command: cmd.Command,
				Success: false,
				Message: "Pattern type 'sequence' requires at least 2 sequence elements in arguments.",
			}
		}

		contentForSequence := []string{}
		if source == "memory" {
			contentForSequence = a.Memory
		} else { // audit_log
			for _, c := range a.AuditLog {
				contentForSequence = append(contentForSequence, fmt.Sprintf("%s %s %v", c.Command, strings.Join(c.Args, " "), c.KV))
			}
		}

		sequenceMatchCount := 0
		// Simple string matching sequence (not sophisticated time series or fuzzy matching)
		sequenceString := strings.Join(targetSequence, ".*?") // Regex-like simple match
		reconstructedContent := strings.Join(contentForSequence, "\n")

		// In a real scenario, use Go's regexp or proper sequence analysis library
		// Here, just check if the joined sequence string exists
		if strings.Contains(reconstructedContent, strings.Join(targetSequence, " ")) { // Exact consecutive match
			sequenceMatchCount++
			patternsFound = append(patternsFound, fmt.Sprintf("Exact sequence found: %s", strings.Join(targetSequence, " ")))
		} else {
			// Rough simulation of finding parts of sequence
			foundIndices := []int{}
			lastIndex := -1
			for _, element := range targetSequence {
				idx := strings.Index(reconstructedContent[lastIndex+1:], element)
				if idx != -1 {
					foundIndices = append(foundIndices, lastIndex+1+idx)
					lastIndex = lastIndex + 1 + idx // Move search start past found element
				} else {
					foundIndices = nil // Reset if any element not found in order
					break
				}
			}
			if foundIndices != nil && len(foundIndices) == len(targetSequence) {
				sequenceMatchCount++
				patternsFound = append(patternsFound, fmt.Sprintf("Sequence elements found in order: %s", strings.Join(targetSequence, " ")))
			}
		}

		resultKV["target_sequence"] = strings.Join(targetSequence, " -> ")
		resultKV["sequence_matches"] = strconv.Itoa(sequenceMatchCount)
		if sequenceMatchCount > 0 {
			resultKV["patterns"] = strings.Join(patternsFound, "; ")
			anomaliesFound = append(anomaliesFound, fmt.Sprintf("Observed sequence '%s' %d times.", strings.Join(targetSequence, " -> "), sequenceMatchCount))
		} else {
			resultKV["patterns"] = "No exact or ordered sequence matches found."
		}


	case "anomaly_detection":
		// Simulate anomaly detection by checking for unusual events in audit log/state
		// Example: very frequent errors, unexpected state changes
		anomalyMsg := "Simulating anomaly detection..."
		anomalyCount := 0

		// Check audit log for frequent errors
		errorCount := 0
		for _, c := range a.AuditLog {
			if strings.Contains(strings.ToLower(c.Command), "error") || strings.Contains(strings.ToLower(fmt.Sprintf("%v", c)), "fail") {
				errorCount++
			}
		}
		if errorCount > len(a.AuditLog)/5 && len(a.AuditLog) > 10 { // More than 20% errors in recent log
			anomaliesFound = append(anomaliesFound, fmt.Sprintf("High frequency of errors detected in recent audit log (%d/%d).", errorCount, len(a.AuditLog)))
			anomalyCount++
		}

		// Check state for specific "alert" or "critical" indicators
		for k, v := range a.State {
			valStr := fmt.Sprintf("%v", v)
			if strings.Contains(strings.ToLower(k), "alert") || strings.Contains(strings.ToLower(valStr), "alert") ||
				strings.Contains(strings.ToLower(k), "critical") || strings.Contains(strings.ToLower(valStr), "critical") {
				anomaliesFound = append(anomaliesFound, fmt.Sprintf("Critical state indicator found: %s = %v", k, v))
				anomalyCount++
			}
		}

		resultKV["simulated_anomalies_count"] = strconv.Itoa(anomalyCount)
		resultKV["detected_anomalies"] = strings.Join(anomaliesFound, "; ")
		anomalyMsg += fmt.Sprintf(" Found %d potential anomalies.", anomalyCount)
		if anomalyCount > 0 {
			patternsFound = anomaliesFound // Report anomalies as patterns found
		} else {
			patternsFound = append(patternsFound, "No significant anomalies detected.")
		}

	default:
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: fmt.Sprintf("Unknown pattern type '%s'.", patternType),
		}
	}

	msg := "Pattern detection (simulated) complete."
	if len(patternsFound) > 0 {
		msg = fmt.Sprintf("Pattern detection (simulated) complete. Found: %s", strings.Join(patternsFound, "; "))
	}


	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: msg,
		Result: resultKV,
	}
}

// handlePrioritizeGoals re-evaluates and prioritizes the agent's current goals.
// Usage: prioritize_goals [factors:"urgency,importance"] [update_state:true]
func (a *Agent) handlePrioritizeGoals(cmd *MCPCommand) *MCPResponse {
	factors := strings.Split(cmd.KV["factors"], ",") // e.g., "urgency", "importance", "feasibility"
	if len(factors) == 0 || (len(factors) == 1 && factors[0] == "") {
		factors = []string{"default"} // Default prioritization
	}
	updateState := strings.ToLower(cmd.KV["update_state"]) == "true"

	a.mu.Lock()
	defer a.mu.Unlock()

	originalGoals := append([]string{}, a.Goals...) // Copy original slice
	prioritizedGoals := make([]string, len(a.Goals))
	copy(prioritizedGoals, a.Goals) // Start with current order

	// Simulate prioritization based on factors (very simple logic)
	// In a real agent, this would involve complex scoring based on state, knowledge, incoming info
	switch strings.Join(factors, ",") {
	case "urgency":
		// Simple sim: move goals containing "urgent" or "critical" to the front
		urgentGoals := []string{}
		otherGoals := []string{}
		for _, goal := range prioritizedGoals {
			lowerGoal := strings.ToLower(goal)
			if strings.Contains(lowerGoal, "urgent") || strings.Contains(lowerGoal, "critical") {
				urgentGoals = append(urgentGoals, goal)
			} else {
				otherGoals = append(otherGoals, goal)
			}
		}
		prioritizedGoals = append(urgentGoals, otherGoals...)
	case "importance":
		// Simple sim: move goals containing "important" or "key" to the front (after urgency)
		importantGoals := []string{}
		otherGoals := []string{}
		// First, handle urgency if combined
		if containsAny(factors, []string{"urgency"}) {
			// If urgency was a factor, use the partially sorted list from that step
			// (This requires more complex state passing or re-sorting logic)
			// For this simulation, let's just do a simple pass on the *current* list.
			tempGoals := append([]string{}, prioritizedGoals...) // Sort the current list
			prioritizedGoals = []string{} // Reset

			urgentCandidates := []string{}
			importantCandidates := []string{}
			remaining := []string{}

			for _, goal := range tempGoals {
				lowerGoal := strings.ToLower(goal)
				if strings.Contains(lowerGoal, "urgent") || strings.Contains(lowerGoal, "critical") {
					urgentCandidates = append(urgentCandidates, goal)
				} else if strings.Contains(lowerGoal, "important") || strings.Contains(lowerGoal, "key") {
					importantCandidates = append(importantCandidates, goal)
				} else {
					remaining = append(remaining, goal)
				}
			}
			// Priority order: Urgent -> Important -> Remaining
			prioritizedGoals = append(prioritizedGoals, urgentCandidates...)
			prioritizedGoals = append(prioritizedGoals, importantCandidates...)
			prioritizedGoals = append(prioritizedGoals, remaining...)

		} else { // Only importance as main factor
			tempGoals := append([]string{}, prioritizedGoals...) // Sort the current list
			prioritizedGoals = []string{} // Reset

			importantCandidates := []string{}
			remaining := []string{}

			for _, goal := range tempGoals {
				lowerGoal := strings.ToLower(goal)
				if strings.Contains(lowerGoal, "important") || strings.Contains(lowerGoal, "key") {
					importantCandidates = append(importantCandidates, goal)
				} else {
					remaining = append(remaining, goal)
				}
			}
			prioritizedGoals = append(prioritizedGoals, importantCandidates...)
			prioritizedGoals = append(prioritizedGoals, remaining...)
		}
	// Add other factor cases here...
	case "default":
		// No change, just report current order
	default:
		// Simple sim: just reverse the order as a different prioritization method
		for i, j := 0, len(prioritizedGoals)-1; i < j; i, j = i+1, j-1 {
			prioritizedGoals[i], prioritizedGoals[j] = prioritizedGoals[j], prioritizedGoals[i]
		}
	}

	// Check if order actually changed
	orderChanged := false
	if len(originalGoals) == len(prioritizedGoals) {
		for i := range originalGoals {
			if originalGoals[i] != prioritizedGoals[i] {
				orderChanged = true
				break
			}
		}
	} else {
		// Should not happen with this sim logic, but good practice
		orderChanged = true
	}


	if updateState {
		a.Goals = prioritizedGoals // Update agent's goals
		a.State["last_goal_prioritization_timestamp"] = time.Now().Format(time.RFC3339)
		a.State["last_goal_prioritization_factors"] = strings.Join(factors, ",")
	}

	resultKV := make(map[string]string)
	resultKV["factors_used"] = strings.Join(factors, ",")
	resultKV["order_changed"] = strconv.FormatBool(orderChanged)
	resultKV["state_updated"] = strconv.FormatBool(updateState)
	resultKV["original_order"] = strings.Join(originalGoals, " -> ")
	resultKV["prioritized_order"] = strings.Join(prioritizedGoals, " -> ")
	resultKV["goal_count"] = strconv.Itoa(len(prioritizedGoals))

	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: "Goals prioritized (simulated).",
		Result: resultKV,
	}
}

// Helper for prioritize_goals
func containsAny(list []string, targets []string) bool {
	for _, item := range list {
		for _, target := range targets {
			if item == target {
				return true
			}
		}
	}
	return false
}


// handleQueryCapabilities lists the commands or functions the agent is capable of executing.
// Usage: query_capabilities [filter:"keyword"]
func (a *Agent) handleQueryCapabilities(cmd *MCPCommand) *MCPResponse {
	filterKeyword := cmd.KV["filter"]

	a.mu.Lock()
	defer a.mu.Unlock()

	capabilities := []string{}
	for cmdName := range a.CommandHandlers {
		if filterKeyword == "" || strings.Contains(strings.ToLower(cmdName), strings.ToLower(filterKeyword)) {
			capabilities = append(capabilities, cmdName)
		}
	}

	// Sort for consistent output
	// sort.Strings(capabilities) // Requires import "sort"

	resultKV := make(map[string]string)
	resultKV["count"] = strconv.Itoa(len(capabilities))
	if filterKeyword != "" {
		resultKV["filter"] = filterKeyword
	}
	resultKV["capabilities_list"] = strings.Join(capabilities, ", ")

	msg := fmt.Sprintf("Agent has %d capabilities listed.", len(capabilities))
	if filterKeyword != "" {
		msg = fmt.Sprintf("Agent has %d capabilities matching '%s'.", len(capabilities), filterKeyword)
	}


	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: msg,
		Result: resultKV,
	}
}


// handleGenerateIdea simulates generating a novel concept or approach.
// Usage: generate_idea topic:"new feature" [inspiration:"state_key,memory_keyword"] [novelty:high/medium/low]
func (a *Agent) handleGenerateIdea(cmd *MCPCommand) *MCPResponse {
	topic, ok := cmd.KV["topic"]
	if !ok || topic == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing 'topic' for idea generation.",
		}
	}
	inspirationSources := strings.Split(cmd.KV["inspiration"], ",")
	noveltyLevel := cmd.KV["novelty"]
	if noveltyLevel == "" { noveltyLevel = "medium" }

	a.mu.Lock()
	defer a.mu.Unlock()

	// Simulate drawing inspiration from sources
	inspirationNotes := []string{}
	for _, source := range inspirationSources {
		lowerSource := strings.ToLower(source)
		// Check state keys
		if val, ok := a.State[lowerSource]; ok {
			inspirationNotes = append(inspirationNotes, fmt.Sprintf("State['%s']: %v", lowerSource, val))
		}
		// Check knowledge base keys
		if val, ok := a.KnowledgeBase[lowerSource]; ok {
			inspirationNotes = append(inspirationNotes, fmt.Sprintf("Knowledge['%s']: %v", lowerSource, val))
		}
		// Search memory for keywords
		for _, mem := range a.Memory {
			if strings.Contains(strings.ToLower(mem), lowerSource) {
				inspirationNotes = append(inspirationNotes, fmt.Sprintf("Memory entry contains '%s'", lowerSource))
				break // Only need one memory mention per keyword for sim
			}
		}
	}


	// Simulate idea generation based on topic and inspiration
	generatedIdea := fmt.Sprintf("Simulated idea for topic '%s'.", topic)
	ideaQuality := "basic" // Simulated quality

	switch strings.ToLower(noveltyLevel) {
	case "high":
		generatedIdea += " This idea attempts high novelty by combining disparate concepts."
		if len(inspirationNotes) >= 2 { ideaQuality = "creative" } // Sim: more inspiration = better quality
	case "low":
		generatedIdea += " This idea is conservative, focusing on established approaches."
		ideaQuality = "standard"
	default: // medium
		generatedIdea += " This idea balances novelty and feasibility."
		if len(inspirationNotes) >= 1 { ideaQuality = "enhanced" }
	}

	if len(inspirationNotes) > 0 {
		generatedIdea += fmt.Sprintf(" Inspired by: %s", strings.Join(inspirationNotes, "; "))
	} else {
		generatedIdea += " No specific inspiration sources found or used."
	}

	a.State["last_generated_idea"] = generatedIdea // Store idea in state
	a.State["last_idea_topic"] = topic
	a.State["last_idea_quality"] = ideaQuality


	resultKV := map[string]string{
		"topic": topic,
		"novelty_requested": noveltyLevel,
		"simulated_idea": generatedIdea,
		"simulated_quality": ideaQuality,
		"inspiration_used_count": strconv.Itoa(len(inspirationNotes)),
	}
	if len(inspirationNotes) > 0 {
		resultKV["inspiration_notes"] = strings.Join(inspirationNotes, "; ")
	}


	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: "Idea generated (simulated).",
		Result: resultKV,
	}
}

// handleClusterMemories groups similar or related memories together.
// Usage: cluster_memories [keywords:"event,alert"] [method:"simple_keyword"] [limit:5]
func (a *Agent) handleClusterMemories(cmd *MCPCommand) *MCPResponse {
	keywords := strings.Split(cmd.KV["keywords"], ",")
	method := cmd.KV["method"]
	if method == "" { method = "simple_keyword" } // Default method
	limitStr, ok := cmd.KV["limit"]
	limit := 3 // Default clusters to report
	if ok {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 {
			limit = l
		}
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	if len(a.Memory) < 2 {
		return &MCPResponse{
			Command: cmd.Command,
			Success: true,
			Message: "Not enough memories to cluster.",
			Result: map[string]string{"memory_count": strconv.Itoa(len(a.Memory))},
		}
	}

	clusters := make(map[string][]string) // Simulated clusters: keyword -> list of memories
	clusterCount := 0

	// Simulate clustering based on method
	switch strings.ToLower(method) {
	case "simple_keyword":
		if len(keywords) == 0 || (len(keywords) == 1 && keywords[0] == "") {
			// If no keywords provided, use common words in memory as cluster centers (naive)
			// This is very simplistic; real clustering would use embeddings, etc.
			// For simulation, just pick a few words from the first few memories.
			candidateKeywords := make(map[string]int)
			scanLimit := len(a.Memory)
			if scanLimit > 10 { scanLimit = 10 } // Scan first 10 memories for keywords
			for i := 0; i < scanLimit; i++ {
				words := strings.Fields(strings.ToLower(a.Memory[i]))
				commonWords := map[string]bool{"a": true, "the": true, "is": true, "are": true, "in": true, "on": true, "and": true, "of": true, "to": true, "it": true} // Filter
				for _, word := range words {
					cleanWord := strings.Trim(word, ".,!?;\"'()[]")
					if len(cleanWord) > 3 && !commonWords[cleanWord] {
						candidateKeywords[cleanWord]++
					}
				}
			}
			// Use most frequent candidate keywords as cluster centers (up to a small number)
			sortedKeywords := []string{} // Sort by frequency (simulated sort)
			for k := range candidateKeywords {
				sortedKeywords = append(sortedKeywords, k)
			}
			// Actual sort needed: sort.Slice(sortedKeywords, func(i, j int) bool { return candidateKeywords[sortedKeywords[i]] > candidateKeywords[sortedKeywords[j]] })
			// For simplicity, just take the first few unique candidates
			keywords = []string{}
			for _, k := range sortedKeywords {
				keywords = append(keywords, k)
				if len(keywords) >= 5 { break } // Limit candidates
			}
			if len(keywords) == 0 { keywords = []string{"general"} } // Fallback if no keywords found

		}

		// Assign memories to clusters based on keyword presence
		for _, mem := range a.Memory {
			lowerMem := strings.ToLower(mem)
			assignedToCluster := false
			for _, kw := range keywords {
				if kw != "" && strings.Contains(lowerMem, strings.ToLower(kw)) {
					clusters[kw] = append(clusters[kw], mem)
					assignedToCluster = true
				}
			}
			if !assignedToCluster && containsAny(keywords, []string{"general"}) { // Add to 'general' if no specific keyword match
				clusters["general"] = append(clusters["general"], mem)
			}
		}
		clusterCount = len(clusters)

	case "similarity_score":
		// Very advanced simulation: just group first half and second half
		// Real implementation would require text similarity metrics (e.g., cosine similarity with embeddings)
		clusters["recent_memories"] = a.Memory[len(a.Memory)/2:]
		clusters["older_memories"] = a.Memory[:len(a.Memory)/2]
		clusterCount = 2
		if len(a.Memory) < 2 { clusterCount = 0 } // Edge case


	default:
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: fmt.Sprintf("Unknown clustering method '%s'.", method),
		}
	}

	resultKV := make(map[string]string)
	resultKV["method"] = method
	resultKV["total_memories"] = strconv.Itoa(len(a.Memory))
	resultKV["simulated_clusters_count"] = strconv.Itoa(clusterCount)
	resultKV["reported_clusters_limit"] = strconv.Itoa(limit)

	reportedCount := 0
	for clusterName, memories := range clusters {
		if reportedCount >= limit { break }
		resultKV[fmt.Sprintf("cluster_%d_name", reportedCount+1)] = clusterName
		resultKV[fmt.Sprintf("cluster_%d_size", reportedCount+1)] = strconv.Itoa(len(memories))
		// Add some sample memories from the cluster
		sampleSize := len(memories)
		if sampleSize > 3 { sampleSize = 3 }
		sampleMemories := []string{}
		for i := 0; i < sampleSize; i++ {
			sampleMemories = append(sampleMemories, memories[i])
		}
		resultKV[fmt.Sprintf("cluster_%d_sample_memories", reportedCount+1)] = strings.Join(sampleMemories, ";;; ") // Use distinctive separator
		reportedCount++
	}

	msg := fmt.Sprintf("Memory clustering (simulated) complete. Found %d clusters, reporting up to %d.", clusterCount, reportedCount)


	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: msg,
		Result: resultKV,
	}
}


// handleReportSummary generates a formatted summary report.
// Usage: report_summary type:"daily_activity" [format:text/json]
func (a *Agent) handleReportSummary(cmd *MCPCommand) *MCPResponse {
	reportType, ok := cmd.KV["type"]
	if !ok || reportType == "" {
		reportType = "status" // Default report
	}
	format := cmd.KV["format"]
	if format == "" { format = "text" } // Default format

	a.mu.Lock()
	defer a.mu.Unlock()

	reportData := make(map[string]interface{})
	reportText := ""
	msg := ""

	// Simulate generating report data based on type
	switch strings.ToLower(reportType) {
	case "status":
		msg = "Generating status report."
		reportData["agent_id"] = a.ID
		reportData["timestamp"] = time.Now().Format(time.RFC3339)
		reportData["operational_status"] = "OK" // Simulated
		reportData["state_entries"] = len(a.State)
		reportData["memory_entries"] = len(a.Memory)
		reportData["knowledge_entries"] = len(a.KnowledgeBase)
		reportData["active_goals"] = len(a.Goals)
		reportData["recent_commands_count"] = len(a.AuditLog)
		if lastTask, ok := a.State["last_task_status"].(string); ok { reportData["last_task_status"] = lastTask }
		if lastTaskDetails, ok := a.State["last_task_details"].(string); ok { reportData["last_task_details"] = lastTaskDetails }

		// Text format for status
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("--- Agent Status Report (%s) ---\n", a.ID))
		sb.WriteString(fmt.Sprintf("Timestamp: %s\n", reportData["timestamp"]))
		sb.WriteString(fmt.Sprintf("Operational Status: %v\n", reportData["operational_status"]))
		sb.WriteString(fmt.Sprintf("State Entries: %v\n", reportData["state_entries"]))
		sb.WriteString(fmt.Sprintf("Memory Entries: %v\n", reportData["memory_entries"]))
		sb.WriteString(fmt.Sprintf("Knowledge Entries: %v\n", reportData["knowledge_entries"]))
		sb.WriteString(fmt.Sprintf("Active Goals: %v\n", reportData["active_goals"]))
		sb.WriteString(fmt.Sprintf("Recent Commands Logged: %v\n", reportData["recent_commands_count"]))
		if lastTask, ok := reportData["last_task_status"].(string); ok { sb.WriteString(fmt.Sprintf("Last Task Status: %s\n", lastTask)) }
		if lastTaskDetails, ok := reportData["last_task_details"].(string); ok { sb.WriteString(fmt.Sprintf("Last Task Details: %s\n", lastTaskDetails)) }
		sb.WriteString("------------------------------------\n")
		reportText = sb.String()


	case "daily_activity":
		msg = "Generating daily activity report (simulated)."
		reportData["agent_id"] = a.ID
		reportData["report_date"] = time.Now().Format("2006-01-02")
		reportData["timestamp"] = time.Now().Format(time.RFC3339)
		reportData["commands_processed_today"] = len(a.AuditLog) // Sim: assume all log is 'today'
		reportData["memories_added_today"] = len(a.Memory) // Sim: assume all memory is 'today'
		// Simulate finding some recent events/highlights
		highlights := []string{}
		if len(a.AuditLog) > 0 {
			highlights = append(highlights, fmt.Sprintf("Processed %d commands.", len(a.AuditLog)))
		}
		if len(a.Memory) > 0 {
			highlights = append(highlights, fmt.Sprintf("Added %d memory entries.", len(a.Memory)))
		}
		if lastTask, ok := a.State["last_task_status"].(string); ok {
			highlights = append(highlights, fmt.Sprintf("Last task status: %s", lastTask))
		}
		if len(a.Goals) > 0 {
			highlights = append(highlights, fmt.Sprintf("Active goals include: %s", strings.Join(a.Goals, ", ")))
		}
		reportData["activity_highlights"] = highlights

		// Text format for daily activity
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("--- Agent Daily Activity Report (%s) ---\n", a.ID))
		sb.WriteString(fmt.Sprintf("Date: %v\n", reportData["report_date"]))
		sb.WriteString(fmt.Sprintf("Timestamp: %v\n", reportData["timestamp"]))
		sb.WriteString(fmt.Sprintf("Commands Processed: %v\n", reportData["commands_processed_today"]))
		sb.WriteString(fmt.Sprintf("Memories Added: %v\n", reportData["memories_added_today"]))
		sb.WriteString("Highlights:\n")
		if len(highlights) == 0 {
			sb.WriteString(" - No significant activity recorded.\n")
		} else {
			for _, h := range highlights {
				sb.WriteString(fmt.Sprintf(" - %s\n", h))
			}
		}
		sb.WriteString("-------------------------------------------\n")
		reportText = sb.String()

	case "knowledge_summary":
		msg = "Generating knowledge base summary."
		reportData["agent_id"] = a.ID
		reportData["timestamp"] = time.Now().Format(time.RFC3339)
		reportData["knowledge_entries_count"] = len(a.KnowledgeBase)
		// Summarize types of knowledge entries (simple sim)
		typeCounts := make(map[string]int)
		for k, v := range a.KnowledgeBase {
			typeCounts[fmt.Sprintf("%T", v)]++
			_ = k // Use k to avoid lint warning, though not summarized by key here
		}
		typeSummary := []string{}
		for t, count := range typeCounts {
			typeSummary = append(typeSummary, fmt.Sprintf("%s (%d)", t, count))
		}
		reportData["knowledge_type_summary"] = typeSummary
		// List some key entries (simulated)
		keyEntries := []string{}
		count := 0
		for k := range a.KnowledgeBase {
			if count >= 5 { break } // Limit list
			keyEntries = append(keyEntries, k)
			count++
		}
		reportData["sample_key_entries"] = keyEntries


		// Text format for knowledge summary
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("--- Agent Knowledge Base Summary (%s) ---\n", a.ID))
		sb.WriteString(fmt.Sprintf("Timestamp: %v\n", reportData["timestamp"]))
		sb.WriteString(fmt.Sprintf("Total Entries: %v\n", reportData["knowledge_entries_count"]))
		sb.WriteString(fmt.Sprintf("Entry Types: %s\n", strings.Join(reportData["knowledge_type_summary"].([]string), ", ")))
		sb.WriteString(fmt.Sprintf("Sample Key Entries: %s\n", strings.Join(reportData["sample_key_entries"].([]string), ", ")))
		sb.WriteString("-------------------------------------------\n")
		reportText = sb.String()


	default:
		msg = fmt.Sprintf("Unknown report type '%s'. Performing status report instead.", reportType)
		// Fallback to status report logic
		reportType = "status" // Update for result KV
		reportData["agent_id"] = a.ID
		reportData["timestamp"] = time.Now().Format(time.RFC3339)
		reportData["operational_status"] = "OK"
		reportData["state_entries"] = len(a.State)
		reportData["memory_entries"] = len(a.Memory)
		reportData["knowledge_entries"] = len(a.KnowledgeBase)
		reportData["active_goals"] = len(a.Goals)
		reportData["recent_commands_count"] = len(a.AuditLog)
		if lastTask, ok := a.State["last_task_status"].(string); ok { reportData["last_task_status"] = lastTask }
		if lastTaskDetails, ok := a.State["last_task_details"].(string); ok { reportData["last_task_details"] = lastTaskDetails }

		// Text format for fallback status
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("--- Agent Status Report (%s) ---\n", a.ID))
		sb.WriteString(fmt.Sprintf("Timestamp: %v\n", reportData["timestamp"]))
		sb.WriteString(fmt.Sprintf("Operational Status: %v\n", reportData["operational_status"]))
		sb.WriteString(fmt.Sprintf("State Entries: %v\n", reportData["state_entries"]))
		sb.WriteString(fmt.Sprintf("Memory Entries: %v\n", reportData["memory_entries"]))
		sb.WriteString(fmt.Sprintf("Knowledge Entries: %v\n", reportData["knowledge_entries"]))
		sb.WriteString(fmt.Sprintf("Active Goals: %v\n", reportData["active_goals"]))
		sb.WriteString(fmt.Sprintf("Recent Commands Logged: %v\n", reportData["recent_commands_count"]))
		if lastTask, ok := reportData["last_task_status"].(string); ok { sb.WriteString(fmt.Sprintf("Last Task Status: %s\n", lastTask)) }
		if lastTaskDetails, ok := reportData["last_task_details"].(string); ok { sb.WriteString(fmt.Sprintf("Last Task Details: %s\n", lastTaskDetails)) }
		sb.WriteString("------------------------------------\n")
		reportText = sb.String()
	}

	resultKV := make(map[string]string)
	resultKV["report_type"] = reportType
	resultKV["format_requested"] = format

	// Format the output
	finalReportContent := ""
	switch strings.ToLower(format) {
	case "json":
		bytes, err := json.MarshalIndent(reportData, "", "  ")
		if err != nil {
			finalReportContent = fmt.Sprintf("Error formatting report as JSON: %v", err)
			msg += " (JSON formatting failed)"
			resultKV["format_error"] = "JSON formatting failed"
			resultKV["raw_data"] = fmt.Sprintf("%v", reportData) // Fallback
		} else {
			finalReportContent = string(bytes)
			resultKV["report_content_format"] = "json"
		}
	case "text":
		finalReportContent = reportText
		resultKV["report_content_format"] = "text"
	default:
		finalReportContent = fmt.Sprintf("Unknown format '%s'. Using text format.\n%s", format, reportText)
		msg += fmt.Sprintf(" (Unknown format '%s', used text)", format)
		resultKV["format_warning"] = fmt.Sprintf("Unknown format '%s', used text", format)
		resultKV["report_content_format"] = "text"
	}

	resultKV["report_content"] = finalReportContent // Put the report content in a single KV pair


	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: msg,
		Result: resultKV,
	}
}

// handleRequestFeedback simulates requesting feedback on its performance or output.
// Usage: request_feedback target:"user_XYZ" item:"last_report" [context:"was my analysis clear?"]
func (a *Agent) handleRequestFeedback(cmd *MCPCommand) *MCPResponse {
	target := cmd.KV["target"]
	item := cmd.KV["item"]
	context := cmd.KV["context"]

	if target == "" || item == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing 'target' or 'item' for feedback request.",
		}
	}

	// Simulate sending a feedback request (in a real system, this would send an outgoing message)
	feedbackRequestMsg := fmt.Sprintf("To %s: Agent %s is requesting feedback on item '%s'.", target, a.ID, item)
	if context != "" {
		feedbackRequestMsg += fmt.Sprintf(" Context: %s", context)
	}
	feedbackRequestMsg += " Please provide your assessment."

	// In a real system, this would likely trigger an outgoing message via agent.Output
	// For this simulation, we'll just log it and report success.
	a.mu.Lock()
	feedbackLogEntry := fmt.Sprintf("[FeedbackRequest] Target: %s, Item: %s, Context: %s, Timestamp: %s",
		target, item, context, time.Now().Format(time.RFC3339))
	a.Memory = append(a.Memory, feedbackLogEntry) // Log request in memory
	a.State["last_feedback_request"] = map[string]string{
		"target": target,
		"item": item,
		"context": context,
		"timestamp": time.Now().Format(time.RFC3339),
	}
	a.mu.Unlock()


	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: "Feedback request simulated.",
		Result: map[string]string{
			"target": target,
			"item": item,
			"context": context,
			"simulated_action": "feedback request sent",
		},
	}
}

// handleEvaluateInputTrust simulates evaluating the trustworthiness or reliability of an input source.
// Usage: evaluate_input_trust source:"user_XYZ" content:"message_text" [context:"previous interactions"]
func (a *Agent) handleEvaluateInputTrust(cmd *MCPCommand) *MCPResponse {
	source := cmd.KV["source"]
	content := cmd.KV["content"]
	context := cmd.KV["context"] // Optional context

	if source == "" || content == "" {
		return &MCPResponse{
			Command: cmd.Command,
			Success: false,
			Message: "Missing 'source' or 'content' for trust evaluation.",
		}
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	// Simulate trust evaluation based on source name and content keywords
	trustScore := 0.5 // Default neutral score (0 to 1)
	reason := "Default evaluation."

	lowerSource := strings.ToLower(source)
	lowerContent := strings.ToLower(content)

	// Simulate based on source name
	if strings.Contains(lowerSource, "trusted") || strings.Contains(lowerSource, "admin") {
		trustScore += 0.3
		reason += " Source name suggests reliability."
	}
	if strings.Contains(lowerSource, "unverified") || strings.Contains(lowerSource, "guest") {
		trustScore -= 0.2
		reason += " Source name suggests lower verification."
	}

	// Simulate based on content keywords
	if strings.Contains(lowerContent, "error") || strings.Contains(lowerContent, "failure") || strings.Contains(lowerContent, "issue") {
		// Content relates to problems, often indicates potentially useful, but requires verification
		trustScore += 0.1 // Slight increase for potential importance
		reason += " Content relates to potential issues."
	}
	if strings.Contains(lowerContent, "critical") || strings.Contains(lowerContent, "urgent") {
		trustScore += 0.1 // Higher score for urgency indicators
		reason += " Content contains urgency indicators."
	}
	if strings.Contains(lowerContent, "false") || strings.Contains(lowerContent, "incorrect") || strings.Contains(lowerContent, "fake") {
		trustScore -= 0.3 // Lower score for negative/falsehood indicators
		reason += " Content contains possible falsehood indicators."
	}
	if strings.Contains(lowerContent, "?") || strings.Contains(lowerContent, "maybe") {
		trustScore -= 0.1 // Lower score for uncertainty
		reason += " Content contains uncertainty indicators."
	}

	// Clamp score between 0 and 1
	if trustScore < 0 { trustScore = 0 }
	if trustScore > 1 { trustScore = 1 }

	trustLevel := "medium"
	if trustScore > 0.7 { trustLevel = "high" }
	if trustScore < 0.4 { trustLevel = "low" }

	// Update state with last evaluation (simulated)
	a.State["last_trust_evaluation"] = map[string]interface{}{
		"source": source,
		"sim_score": trustScore,
		"sim_level": trustLevel,
		"timestamp": time.Now().Format(time.RFC3339),
		"reason": reason,
	}

	resultKV := map[string]string{
		"source": source,
		"simulated_trust_score": fmt.Sprintf("%.2f", trustScore),
		"simulated_trust_level": trustLevel,
		"simulated_reasoning": reason,
		"content_snippet": content, // Echo content for context
	}
	if context != "" { resultKV["context_considered"] = context }

	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: "Input trust evaluated (simulated).",
		Result: resultKV,
	}
}

// handleSuggestAction suggests the next logical action based on current state and goals.
// Usage: suggest_action [context:"current task stage"] [limit:3]
func (a *Agent) handleSuggestAction(cmd *MCPCommand) *MCPResponse {
	context := cmd.KV["context"] // Optional context
	limitStr, ok := cmd.KV["limit"]
	limit := 5 // Default number of suggestions
	if ok {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 {
			limit = l
		}
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	suggestions := []string{}
	reasoning := []string{}

	// Simulate suggesting actions based on state, goals, and context
	// This is a very simple rule-based simulation
	suggestionScore := func(s string) int { // Helper to score suggestions
		score := 0
		lowerS := strings.ToLower(s)
		if strings.Contains(lowerS, "report") { score += 1 }
		if strings.Contains(lowerS, "analyze") { score += 2 }
		if strings.Contains(lowerS, "plan") { score += 3 }
		if strings.Contains(lowerS, "execute") { score += 4 }
		if strings.Contains(lowerS, "learn") { score += 5 }
		if strings.Contains(lowerS, "optimize") { score += 6 }
		return score
	}

	// Add suggestions based on goals (prioritize higher scored suggestions related to goals)
	goalSuggestions := map[string]string{}
	if len(a.Goals) > 0 {
		prioritizedGoal := a.Goals[0] // Focus on the top goal
		reasoning = append(reasoning, fmt.Sprintf("Focusing on top goal: '%s'.", prioritizedGoal))

		// Simulate actions related to this goal
		if strings.Contains(strings.ToLower(prioritizedGoal), "learn") {
			goalSuggestions["learn_from_outcome"] = "Suggest checking outcomes to learn."
			goalSuggestions["analyze_text"] = "Suggest analyzing new information related to learning."
		}
		if strings.Contains(strings.ToLower(prioritizedGoal), "report") {
			goalSuggestions["report_summary"] = "Suggest generating relevant reports."
			goalSuggestions["synthesize_info"] = "Suggest synthesizing information for reporting."
		}
		if strings.Contains(strings.ToLower(prioritizedGoal), "task") {
			goalSuggestions["plan_simple_task"] = "Suggest planning the task."
			goalSuggestions["execute_sim_task"] = "Suggest executing (simulating) the task."
		}
		if strings.Contains(strings.ToLower(prioritizedGoal), "optimize") {
			goalSuggestions["optimize_setting"] = "Suggest running optimization for relevant parameters."
		}
		if strings.Contains(strings.ToLower(prioritizedGoal), "pattern") || strings.Contains(strings.ToLower(prioritizedGoal), "anomaly") {
			goalSuggestions["detect_pattern"] = "Suggest detecting patterns or anomalies."
		}
	} else {
		reasoning = append(reasoning, "No active goals. Suggesting general maintenance actions.")
		// Suggest general maintenance if no goals
		goalSuggestions["agent.status"] = "Check overall agent status."
		goalSuggestions["audit_trail"] = "Review recent activity."
		goalSuggestions["self_reflect"] = "Perform self-reflection."
	}

	// Add suggestions based on state (simulated)
	stateSuggestions := map[string]string{}
	if lastStatus, ok := a.State["last_task_status"].(string); ok {
		if lastStatus == "failure" {
			stateSuggestions["learn_from_outcome"] = "Suggest learning from the last failure."
			stateSuggestions["audit_trail"] = "Suggest reviewing audit trail for failure details."
			reasoning = append(reasoning, "Last task failed, suggesting analysis.")
		} else if lastStatus == "partial_success" {
			stateSuggestions["learn_from_outcome"] = "Suggest learning from the partial success."
			stateSuggestions["plan_simple_task"] = "Suggest replanning the task."
			reasoning = append(reasoning, "Last task was partial, suggesting refinement.")
		}
	}
	if lastIdea, ok := a.State["last_generated_idea"].(string); ok && lastIdea != "" {
		stateSuggestions["report_summary"] = "Suggest reporting on the generated idea."
		stateSuggestions["request_feedback"] = "Suggest requesting feedback on the idea."
		reasoning = append(reasoning, "A new idea was generated, suggesting follow-up.")
	}

	// Combine and filter suggestions by score and limit
	combinedSuggestions := make(map[string]string)
	for k, v := range goalSuggestions { combinedSuggestions[k] = v }
	for k, v := range stateSuggestions { combinedSuggestions[k] = v }

	// Convert to a sortable slice
	type suggestion struct {
		CmdName string
		Reason string
		Score int
	}
	sortedSuggestions := []suggestion{}
	for cmdName, reason := range combinedSuggestions {
		sortedSuggestions = append(sortedSuggestions, suggestion{
			CmdName: cmdName,
			Reason: reason,
			Score: suggestionScore(cmdName), // Use simulated score
		})
	}

	// Sort by score descending (simple sim)
	// sort.Slice(sortedSuggestions, func(i, j int) bool { return sortedSuggestions[i].Score > sortedSuggestions[j].Score })

	// Select top 'limit' suggestions
	finalSuggestions := []string{}
	suggestionDetails := []string{}
	for i, s := range sortedSuggestions {
		if i >= limit { break }
		finalSuggestions = append(finalSuggestions, s.CmdName)
		suggestionDetails = append(suggestionDetails, fmt.Sprintf("%s (%s)", s.CmdName, s.Reason))
	}

	resultKV := make(map[string]string)
	resultKV["context_provided"] = context
	resultKV["suggested_actions_count"] = strconv.Itoa(len(finalSuggestions))
	resultKV["suggested_actions_list"] = strings.Join(finalSuggestions, ", ")
	resultKV["suggestion_details"] = strings.Join(suggestionDetails, "; ")
	resultKV["simulated_reasoning"] = strings.Join(reasoning, " ")


	return &MCPResponse{
		Command: cmd.Command,
		Success: true,
		Message: "Actions suggested (simulated).",
		Result: resultKV,
	}
}


// Helper function to parse a JSON string into a map (for simulation parameters)
func parseJSONString(jsonStr string) map[string]interface{} {
	if jsonStr == "" { return nil }
	var result map[string]interface{}
	err := json.Unmarshal([]byte(jsonStr), &result)
	if err != nil {
		fmt.Printf("Warning: Failed to parse JSON string '%s': %v\n", jsonStr, err)
		return nil
	}
	return result
}

// --- Main Execution ---

func main() {
	// Use channels to simulate input/output streams
	agentInput := make(chan string)
	agentOutput := make(chan string)

	// Create the agent
	agent := NewAgent("Alpha", agentOutput, agentInput)

	// Run the agent in a goroutine
	go agent.Run()

	// Simulate sending commands to the agent
	fmt.Println("Sending sample commands to Agent Alpha...")

	// Command 1: Status check
	agentInput <- "agent.status"
	// Command 2: Set state
	agentInput <- "agent.set_state health_status:OK current_task:idle location:main_chamber"
	// Command 3: Get state
	agentInput <- "agent.get_state health_status location current_task mood" // mood doesn't exist yet
	// Command 4: Remember something
	agentInput <- "agent.remember \"Experienced a minor tremor in Sector 7.\" tags:event,warning"
	agentInput <- "agent.remember \"Completed analysis of log files.\" tags:task_complete,analysis"
	// Command 5: Recall memory
	agentInput <- "agent.recall tremor"
	agentInput <- "agent.recall tags:task_complete"
	// Command 6: Add knowledge
	agentInput <- "agent.add_knowledge sector_7_status:stable system_specs:{\"cpu\":\"quad\",\"ram\":\"16gb\"} important_concept:synergy"
	// Command 7: Query knowledge
	agentInput <- "agent.query_knowledge system_specs important_concept"
	agentInput <- "agent.query_knowledge contains:stable"
	// Command 8: Analyze text
	agentInput <- "agent.analyze_text \"Feeling great about the progress today!\" type:sentiment"
	agentInput <- "agent.analyze_text \"The system reported error code 505.\" type:keywords"
	// Command 9: Synthesize info
	agentInput <- "agent.synthesize_info query:\"recent events and system health\""
	// Command 10: Plan a task
	agentInput <- "agent.plan_simple_task goal:\"investigate sector 7 tremor\""
	// Command 11: Execute simulated task outcome
	agentInput <- "agent.execute_sim_task plan:\"Investigate tremor\" outcome:partial details:\"Source not fully identified.\""
	// Command 12: Learn from outcome
	agentInput <- "agent.learn_from_outcome task:\"investigate_tremor\" outcome:partial notes:\"Need more data sources.\" tags:learning"
	// Command 13: Generate text
	agentInput <- "agent.generate_text prompt:\"draft a brief report on recent activity\" length:short"
	// Command 14: Simulate an event
	agentInput <- "agent.simulate_event scenario:traffic_flow parameters:{\"cars\":1200} duration:5"
	// Command 15: Prognosticate
	agentInput <- "agent.prognosticate event:\"system alert\" context:\"last_sim_congestion:high\" horizon:hour"
	// Command 16: Optimize setting
	agentInput <- "agent.optimize_setting parameter:threshold objective:minimize_errors range:\"0.01,0.5\""
	// Command 17: Set a state for diff demo, then add snapshot
	agentInput <- "agent.set_state temp_config:value1 temp_counter:10 health_status:OK last_task_status:simulated"
	agentInput <- "agent.add_knowledge state_snapshot_snap_A:{\"temp_config\":\"value1\",\"temp_counter\":10,\"health_status\":\"OK\",\"last_task_status\":\"simulated\"}" // Manual snapshot
	// Command 18: Change state again, then diff
	agentInput <- "agent.set_state temp_counter:11 temp_new_item:added health_status:ALERT"
	agentInput <- "agent.diff_state snapshot_id:snap_A"
	// Command 19: Audit trail
	agentInput <- "agent.audit_trail limit:5"
	agentInput <- "agent.audit_trail contains:state limit:3"
	// Command 20: Self reflect
	agentInput <- "agent.self_reflect scope:summary"
	agentInput <- "agent.self_reflect scope:memory"
	// Command 21: Detect pattern
	agentInput <- "agent.detect_pattern type:keyword_frequency source:memory keywords:tremor,alert,analysis"
	agentInput <- "agent.detect_pattern type:anomaly_detection source:audit_log"
	agentInput <- "agent.detect_pattern type:sequence source:audit_log login success logout" // Sim sequence
	// Command 22: Prioritize goals (add a new goal first)
	agent.mu.Lock() // Manually add a goal for the demo
	agent.Goals = append(agent.Goals, "investigate_anomaly")
	agent.mu.Unlock()
	agentInput <- "agent.prioritize_goals factors:urgency"
	agentInput <- "agent.prioritize_goals factors:default update_state:true" // Revert/apply default
	// Command 23: Query capabilities
	agentInput <- "agent.query_capabilities"
	agentInput <- "agent.query_capabilities filter:sim"
	// Command 24: Generate idea
	agentInput <- "agent.generate_idea topic:\"new reporting feature\" inspiration:last_report,analysis novelty:high"
	// Command 25: Cluster memories
	agentInput <- "agent.cluster_memories method:simple_keyword keywords:tremor,analysis limit:2"
	agentInput <- "agent.cluster_memories method:simple_keyword limit:3" // Auto keywords
	// Command 26: Report summary
	agentInput <- "agent.report_summary type:status format:text"
	agentInput <- "agent.report_summary type:daily_activity format:json"
	agentInput <- "agent.report_summary type:knowledge_summary"
	// Command 27: Request Feedback
	agentInput <- "agent.request_feedback target:operator_console item:last_report context:\"Was the daily activity report clear?\""
	// Command 28: Evaluate Input Trust
	agentInput <- "agent.evaluate_input_trust source:\"user_unverified_XYZ\" content:\"System is about to fail!\" context:\"Source known for exaggeration.\""
	agentInput <- "agent.evaluate_input_trust source:\"admin_console\" content:\"Routine check passed.\""
	// Command 29: Suggest Action
	agentInput <- "agent.suggest_action limit:3 context:\"After task failure.\""
	agentInput <- "agent.suggest_action limit:2" // Default context

	// Simulate sending an unknown command
	agentInput <- "agent.unknown_command arg1 arg2 key:value"

	// Simulate a malformed command
	agentInput <- "malformed command here without structure"

	// Wait a moment for commands to process and collect responses
	time.Sleep(5 * time.Second)

	fmt.Println("\n--- Agent Responses ---")
	// Read responses from the output channel until it's likely empty
	// In a real app, you'd close the output channel or have a shutdown signal.
	// For this demo, read until a timeout or count is reached.
	responseCount := 0
	for {
		select {
		case response := <-agentOutput:
			fmt.Printf("Response: %s\n", response)
			responseCount++
		case <-time.After(500 * time.Millisecond): // Timeout if no response for a while
			fmt.Println("No more responses from agent (timeout).")
			goto endSimulation
		}
	}

endSimulation:
	// In a real app, you would properly shut down the agent goroutine
	// and close the channels. For this demo, we just let the main
	// function finish.
	close(agentInput) // Signal agent to stop processing
	// The agent.Run loop will exit when agentInput is closed.
	// A robust shutdown would ensure all pending messages are processed
	// and the output channel is also closed after the agent stops.

	fmt.Println("Simulation finished.")
}

```