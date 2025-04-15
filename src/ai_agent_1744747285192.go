```go
/*
AI Agent with MCP Interface in Go

Outline and Function Summary:

This AI Agent, named "SynergyOS," operates with a Message Control Protocol (MCP) interface for communication. It is designed to be a versatile and advanced agent capable of performing a range of creative, trendy, and intelligent tasks. It is built with a focus on personalization, proactive assistance, and novel functionalities beyond typical open-source AI agents.

Function Summary (20+ Functions):

1.  **Personalized Creative Storytelling (TellStory):** Generates unique stories tailored to user preferences (genre, themes, characters).
2.  **Dynamic Skill Recommendation (RecommendSkill):** Analyzes user activities and recommends relevant skills to learn for personal/professional growth.
3.  **Sentiment Trend Analysis (AnalyzeSentimentTrend):**  Monitors social media or news feeds to identify and summarize emerging sentiment trends on specific topics.
4.  **Contextual Memory Management (StoreContext, RetrieveContext, ForgetContext):**  Maintains and manages contextual memory of user interactions for more coherent and personalized responses.
5.  **Interactive Data Visualization Generation (GenerateVisualization):** Creates dynamic and interactive data visualizations based on user-provided datasets or requests.
6.  **Personalized News Summarization (SummarizeNews):**  Aggregates and summarizes news articles based on user's interests and reading history.
7.  **Ethical Reasoning Module (CheckEthicalImplications):** Evaluates the ethical implications of user requests or AI actions to ensure responsible AI behavior.
8.  **Predictive Wellness Guidance (PredictWellnessRisk):**  Analyzes user's lifestyle data (simulated in this example) to predict potential wellness risks and suggest proactive measures.
9.  **Adaptive Learning Path Generation (GenerateLearningPath):** Creates personalized learning paths for users based on their goals, current knowledge, and learning style.
10. **Real-time Language Style Adaptation (AdaptLanguageStyle):** Dynamically adjusts its language style (formal, informal, technical, etc.) based on the detected user communication style.
11. **Interactive Code Snippet Generation (GenerateCodeSnippet):**  Generates code snippets in various programming languages based on natural language descriptions and context.
12. **Creative Recipe Generation (GenerateRecipe):**  Creates unique and personalized recipes based on dietary preferences, available ingredients, and cuisine types.
13. **Personalized Music Playlist Curation (CuratePlaylist):**  Generates personalized music playlists based on user's mood, activity, and musical taste.
14. **Anomaly Detection in User Behavior (DetectBehaviorAnomaly):**  Monitors user interaction patterns to detect unusual or anomalous behavior patterns for security or personalized support.
15. **Dynamic Task Prioritization (PrioritizeTasks):** Helps users prioritize tasks based on urgency, importance, and user-defined goals.
16. **Interactive Simulation Environment Generation (GenerateSimulation):** Creates simple interactive simulation environments for learning or experimentation based on user requests.
17. **Personalized Travel Itinerary Planning (PlanTravelItinerary):** Generates personalized travel itineraries based on user preferences, budget, and travel style.
18. **Smart Home Automation Script Generation (GenerateAutomationScript):** Generates scripts for smart home automation based on user-defined scenarios and device capabilities.
19. **Multi-Modal Input Handling (ProcessMultiModalInput):**  Demonstrates basic capability to process and integrate information from multiple input types (text, image descriptions - simulated).
20. **Proactive Contextual Reminders (SetContextualReminder, TriggerContextualReminder):** Sets and triggers reminders based on user context (location, activity, keywords in conversation).
21. **Agent Self-Improvement Learning (SimulateSelfImprovement):** Simulates a basic self-improvement mechanism by tracking user feedback and adjusting responses over time (simplified).
22. **Interactive Visual Art Generation (GenerateVisualArt):** Generates abstract or stylized visual art based on user-specified themes or emotions.

MCP Interface:
The agent uses a simple message-based interface (MCP) where messages are structs containing a `MessageType` and `Payload`. This allows for structured communication and extensibility.

Note: This is a simplified example demonstrating the architecture and function concepts. Actual implementation of each function would require more complex AI/ML models and data processing logic. Some functions are simulated or use placeholder logic for demonstration purposes.
*/

package main

import (
	"fmt"
	"math/rand"
	"strings"
	"time"
)

// MessageType defines the types of messages the agent can process
type MessageType string

const (
	TypeTellStory             MessageType = "TellStory"
	TypeRecommendSkill          MessageType = "RecommendSkill"
	TypeAnalyzeSentimentTrend   MessageType = "AnalyzeSentimentTrend"
	TypeStoreContext            MessageType = "StoreContext"
	TypeRetrieveContext         MessageType = "RetrieveContext"
	TypeForgetContext           MessageType = "ForgetContext"
	TypeGenerateVisualization   MessageType = "GenerateVisualization"
	TypeSummarizeNews           MessageType = "SummarizeNews"
	TypeCheckEthicalImplications MessageType = "CheckEthicalImplications"
	TypePredictWellnessRisk     MessageType = "PredictWellnessRisk"
	TypeGenerateLearningPath    MessageType = "GenerateLearningPath"
	TypeAdaptLanguageStyle      MessageType = "AdaptLanguageStyle"
	TypeGenerateCodeSnippet     MessageType = "GenerateCodeSnippet"
	TypeGenerateRecipe          MessageType = "GenerateRecipe"
	TypeCuratePlaylist          MessageType = "CuratePlaylist"
	TypeDetectBehaviorAnomaly   MessageType = "DetectBehaviorAnomaly"
	TypePrioritizeTasks         MessageType = "PrioritizeTasks"
	TypeGenerateSimulation      MessageType = "GenerateSimulation"
	TypePlanTravelItinerary     MessageType = "PlanTravelItinerary"
	TypeGenerateAutomationScript MessageType = "GenerateAutomationScript"
	TypeProcessMultiModalInput  MessageType = "ProcessMultiModalInput"
	TypeSetContextualReminder   MessageType = "SetContextualReminder"
	TypeTriggerContextualReminder MessageType = "TriggerContextualReminder"
	TypeSimulateSelfImprovement MessageType = "SimulateSelfImprovement"
	TypeGenerateVisualArt       MessageType = "GenerateVisualArt"
	TypeUnknown               MessageType = "Unknown"
)

// Message struct for MCP interface
type Message struct {
	Type    MessageType
	Payload string // Can be JSON or string, depending on complexity
}

// Agent struct representing the AI Agent
type Agent struct {
	Name            string
	ContextMemory   map[string]string // Simple in-memory context storage
	UserPreferences map[string]string // Simulated user preferences
	LearningData    map[MessageType][]string // Simulate learning data for self-improvement
}

// NewAgent creates a new AI Agent instance
func NewAgent(name string) *Agent {
	return &Agent{
		Name:            name,
		ContextMemory:   make(map[string]string),
		UserPreferences: make(map[string]string),
		LearningData:    make(map[MessageType][]string),
	}
}

// ProcessMessage is the MCP interface entry point for the agent
func (a *Agent) ProcessMessage(msg Message) string {
	fmt.Printf("Agent '%s' received message: Type='%s', Payload='%s'\n", a.Name, msg.Type, msg.Payload)

	switch msg.Type {
	case TypeTellStory:
		return a.TellStory(msg.Payload)
	case TypeRecommendSkill:
		return a.RecommendSkill(msg.Payload)
	case TypeAnalyzeSentimentTrend:
		return a.AnalyzeSentimentTrend(msg.Payload)
	case TypeStoreContext:
		return a.StoreContext(msg.Payload)
	case TypeRetrieveContext:
		return a.RetrieveContext(msg.Payload)
	case TypeForgetContext:
		return a.ForgetContext(msg.Payload)
	case TypeGenerateVisualization:
		return a.GenerateVisualization(msg.Payload)
	case TypeSummarizeNews:
		return a.SummarizeNews(msg.Payload)
	case TypeCheckEthicalImplications:
		return a.CheckEthicalImplications(msg.Payload)
	case TypePredictWellnessRisk:
		return a.PredictWellnessRisk(msg.Payload)
	case TypeGenerateLearningPath:
		return a.GenerateLearningPath(msg.Payload)
	case TypeAdaptLanguageStyle:
		return a.AdaptLanguageStyle(msg.Payload)
	case TypeGenerateCodeSnippet:
		return a.GenerateCodeSnippet(msg.Payload)
	case TypeGenerateRecipe:
		return a.GenerateRecipe(msg.Payload)
	case TypeCuratePlaylist:
		return a.CuratePlaylist(msg.Payload)
	case TypeDetectBehaviorAnomaly:
		return a.DetectBehaviorAnomaly(msg.Payload)
	case TypePrioritizeTasks:
		return a.PrioritizeTasks(msg.Payload)
	case TypeGenerateSimulation:
		return a.GenerateSimulation(msg.Payload)
	case TypePlanTravelItinerary:
		return a.PlanTravelItinerary(msg.Payload)
	case TypeGenerateAutomationScript:
		return a.GenerateAutomationScript(msg.Payload)
	case TypeProcessMultiModalInput:
		return a.ProcessMultiModalInput(msg.Payload)
	case TypeSetContextualReminder:
		return a.SetContextualReminder(msg.Payload)
	case TypeTriggerContextualReminder:
		return a.TriggerContextualReminder(msg.Payload)
	case TypeSimulateSelfImprovement:
		return a.SimulateSelfImprovement(msg.Payload)
	case TypeGenerateVisualArt:
		return a.GenerateVisualArt(msg.Payload)
	default:
		return a.handleUnknownMessage(msg)
	}
}

func (a *Agent) handleUnknownMessage(msg Message) string {
	return fmt.Sprintf("Unknown message type: %s. Agent '%s' cannot process this.", msg.Type, a.Name)
}

// 1. Personalized Creative Storytelling
func (a *Agent) TellStory(preferences string) string {
	genres := []string{"Fantasy", "Sci-Fi", "Mystery", "Romance", "Adventure"}
	themes := []string{"Friendship", "Courage", "Discovery", "Loss", "Hope"}
	characters := []string{"Brave Knight", "Wise Wizard", "Curious Explorer", "Loyal Companion", "Mysterious Stranger"}

	genre := genres[rand.Intn(len(genres))]
	theme := themes[rand.Intn(len(themes))]
	character := characters[rand.Intn(len(characters))]

	story := fmt.Sprintf("Once upon a time, in a land of %s, there was a %s who embodied the theme of %s. This is their story...", genre, character, theme)
	return fmt.Sprintf("Personalized Story (Genre: %s, Theme: %s, Character: %s):\n%s", genre, theme, character, story)
}

// 2. Dynamic Skill Recommendation
func (a *Agent) RecommendSkill(activity string) string {
	skills := map[string][]string{
		"coding":    {"Go", "Python", "JavaScript", "Data Science"},
		"writing":   {"Creative Writing", "Technical Writing", "Copywriting", "Journalism"},
		"designing": {"UI/UX Design", "Graphic Design", "Web Design", "3D Modeling"},
		"analyzing": {"Data Analysis", "Statistical Analysis", "Financial Analysis", "Market Research"},
	}

	relevantSkills := []string{}
	for activityKeyword, skillList := range skills {
		if strings.Contains(strings.ToLower(activity), activityKeyword) {
			relevantSkills = append(relevantSkills, skillList...)
		}
	}

	if len(relevantSkills) == 0 {
		return "Based on the activity, I recommend exploring general skills like 'Problem Solving' or 'Communication'."
	}

	recommendedSkill := relevantSkills[rand.Intn(len(relevantSkills))]
	return fmt.Sprintf("Based on your activity '%s', I recommend learning '%s'.", activity, recommendedSkill)
}

// 3. Sentiment Trend Analysis (Simulated)
func (a *Agent) AnalyzeSentimentTrend(topic string) string {
	sentiments := []string{"Positive", "Negative", "Neutral"}
	trend := sentiments[rand.Intn(len(sentiments))]
	percentage := rand.Intn(100)
	return fmt.Sprintf("Sentiment trend analysis for '%s': Currently trending '%s' at %d%%.", topic, trend, percentage)
}

// 4. Store Context
func (a *Agent) StoreContext(contextData string) string {
	parts := strings.SplitN(contextData, ":", 2)
	if len(parts) != 2 {
		return "Invalid context data format. Use 'key:value'."
	}
	key := strings.TrimSpace(parts[0])
	value := strings.TrimSpace(parts[1])
	a.ContextMemory[key] = value
	return fmt.Sprintf("Context stored: Key='%s'", key)
}

// 5. Retrieve Context
func (a *Agent) RetrieveContext(key string) string {
	value, exists := a.ContextMemory[key]
	if exists {
		return fmt.Sprintf("Retrieved context for key '%s': '%s'", key, value)
	}
	return fmt.Sprintf("No context found for key '%s'.", key)
}

// 6. Forget Context
func (a *Agent) ForgetContext(key string) string {
	_, exists := a.ContextMemory[key]
	if exists {
		delete(a.ContextMemory, key)
		return fmt.Sprintf("Context for key '%s' forgotten.", key)
	}
	return fmt.Sprintf("No context found for key '%s' to forget.", key)
}

// 7. Interactive Data Visualization Generation (Simulated)
func (a *Agent) GenerateVisualization(dataDescription string) string {
	chartTypes := []string{"Bar Chart", "Line Graph", "Pie Chart", "Scatter Plot"}
	chartType := chartTypes[rand.Intn(len(chartTypes))]
	return fmt.Sprintf("Generated a '%s' visualization for data description: '%s' (Visualization is simulated text-based).", chartType, dataDescription)
}

// 8. Personalized News Summarization (Simulated)
func (a *Agent) SummarizeNews(interests string) string {
	newsTopics := []string{"Technology", "World News", "Business", "Sports", "Science"}
	topic1 := newsTopics[rand.Intn(len(newsTopics))]
	topic2 := newsTopics[rand.Intn(len(newsTopics))]
	summary := fmt.Sprintf("News Summary based on interests '%s': Top stories include '%s' and '%s'. (Full summaries are simulated).", interests, topic1, topic2)
	return summary
}

// 9. Ethical Reasoning Module (Placeholder)
func (a *Agent) CheckEthicalImplications(request string) string {
	if strings.Contains(strings.ToLower(request), "harm") || strings.Contains(strings.ToLower(request), "illegal") {
		return "Ethical check: Request flagged as potentially unethical. Please reconsider your request."
	}
	return "Ethical check: Request appears to be ethically sound."
}

// 10. Predictive Wellness Guidance (Simulated)
func (a *Agent) PredictWellnessRisk(lifestyleData string) string {
	risks := []string{"Stress", "Burnout", "Sedentary Lifestyle", "Poor Sleep"}
	risk := risks[rand.Intn(len(risks))]
	return fmt.Sprintf("Wellness prediction based on lifestyle data: Potential risk of '%s' detected. Consider taking preventative measures.", risk)
}

// 11. Adaptive Learning Path Generation (Simulated)
func (a *Agent) GenerateLearningPath(goal string) string {
	steps := []string{"Step 1: Foundation Course", "Step 2: Intermediate Module", "Step 3: Advanced Workshop", "Step 4: Project Assignment"}
	path := strings.Join(steps, " -> ")
	return fmt.Sprintf("Generated learning path for goal '%s': %s", goal, path)
}

// 12. Real-time Language Style Adaptation (Simulated)
func (a *Agent) AdaptLanguageStyle(userStyle string) string {
	styles := []string{"Formal", "Informal", "Technical", "Casual"}
	adaptedStyle := styles[rand.Intn(len(styles))]
	return fmt.Sprintf("Language style adapted based on user style '%s'. Now communicating in '%s' style.", userStyle, adaptedStyle)
}

// 13. Interactive Code Snippet Generation (Simulated)
func (a *Agent) GenerateCodeSnippet(description string) string {
	languages := []string{"Python", "Go", "JavaScript", "Java"}
	language := languages[rand.Intn(len(languages))]
	snippet := fmt.Sprintf("// Simulated %s code snippet for: %s\nfunction example() {\n  // ...code...\n}", language, description)
	return fmt.Sprintf("Generated code snippet (%s) for: '%s'\n```%s\n```", language, description, snippet)
}

// 14. Creative Recipe Generation (Simulated)
func (a *Agent) GenerateRecipe(preferences string) string {
	cuisines := []string{"Italian", "Mexican", "Indian", "Japanese"}
	cuisine := cuisines[rand.Intn(len(cuisines))]
	dish := fmt.Sprintf("Simulated %s Dish", cuisine)
	recipe := fmt.Sprintf("Recipe for '%s' (%s cuisine) - Ingredients and steps are simulated.", dish, cuisine)
	return fmt.Sprintf("Generated recipe (%s cuisine) based on preferences '%s':\n%s", cuisine, preferences, recipe)
}

// 15. Personalized Music Playlist Curation (Simulated)
func (a *Agent) CuratePlaylist(mood string) string {
	genres := []string{"Pop", "Rock", "Classical", "Electronic", "Jazz"}
	genre := genres[rand.Intn(len(genres))]
	playlistName := fmt.Sprintf("%s Mood Mix", strings.Title(mood))
	playlist := fmt.Sprintf("Personalized playlist '%s' (Genre: %s) - Song list is simulated.", playlistName, genre)
	return fmt.Sprintf("Curated playlist for mood '%s' (Genre: %s):\n%s", mood, genre, playlist)
}

// 16. Anomaly Detection in User Behavior (Simulated)
func (a *Agent) DetectBehaviorAnomaly(userActivity string) string {
	anomalyTypes := []string{"Unusual Login Location", "Sudden Data Access", "Suspicious Transaction"}
	anomaly := anomalyTypes[rand.Intn(len(anomalyTypes))]
	if rand.Float64() < 0.3 { // Simulate anomaly detection 30% of the time
		return fmt.Sprintf("Behavior anomaly detected in user activity '%s': '%s'. Alerting security team. (Simulated detection).", userActivity, anomaly)
	}
	return "User behavior within normal range. No anomalies detected (Simulated). "
}

// 17. Dynamic Task Prioritization (Simulated)
func (a *Agent) PrioritizeTasks(taskList string) string {
	tasks := strings.Split(taskList, ",")
	if len(tasks) == 0 {
		return "No tasks provided for prioritization."
	}
	prioritizedTasks := []string{}
	for i := range tasks {
		prioritizedTasks = append(prioritizedTasks, fmt.Sprintf("%d. %s (Priority: %d)", i+1, strings.TrimSpace(tasks[i]), rand.Intn(5)+1)) // Simulate priority 1-5
	}
	return fmt.Sprintf("Prioritized tasks:\n%s", strings.Join(prioritizedTasks, "\n"))
}

// 18. Interactive Simulation Environment Generation (Simulated)
func (a *Agent) GenerateSimulation(scenario string) string {
	envTypes := []string{"City Traffic", "Ecosystem", "Market Trading", "Social Network"}
	envType := envTypes[rand.Intn(len(envTypes))]
	return fmt.Sprintf("Generated a simulated '%s' environment for scenario '%s' (Simulation is text-based and interactive commands are simulated).", envType, scenario)
}

// 19. Personalized Travel Itinerary Planning (Simulated)
func (a *Agent) PlanTravelItinerary(preferences string) string {
	destinations := []string{"Paris", "Tokyo", "New York", "Rome", "Barcelona"}
	destination := destinations[rand.Intn(len(destinations))]
	duration := rand.Intn(7) + 3 // 3-9 days
	itinerary := fmt.Sprintf("Travel Itinerary to '%s' (%d days) - Day-by-day plan is simulated based on preferences '%s'.", destination, duration, preferences)
	return itinerary
}

// 20. Smart Home Automation Script Generation (Simulated)
func (a *Agent) GenerateAutomationScript(scenario string) string {
	devices := []string{"Lights", "Thermostat", "Security System", "Music Player"}
	device := devices[rand.Intn(len(devices))]
	action := "Turn On" // Simplified action
	script := fmt.Sprintf("// Simulated Smart Home Automation Script for: %s\nif (time == evening) {\n  %s.%s();\n}", scenario, device, action)
	return fmt.Sprintf("Generated Smart Home Automation Script for scenario '%s' involving '%s':\n```%s\n```", scenario, device, script)
}

// 21. Multi-Modal Input Handling (Simulated)
func (a *Agent) ProcessMultiModalInput(inputData string) string {
	inputTypes := []string{"Text", "Image Description", "Voice Command"}
	inputType := inputTypes[rand.Intn(len(inputTypes))]
	return fmt.Sprintf("Processed multi-modal input of type '%s'. Input data: '%s' (Multi-modal processing is simulated).", inputType, inputData)
}

// 22. Proactive Contextual Reminders (Simulated - Setting)
func (a *Agent) SetContextualReminder(reminderDetails string) string {
	a.ContextMemory["reminder"] = reminderDetails // Simple reminder storage
	return fmt.Sprintf("Contextual reminder set: '%s'. Will trigger when context is met (Simulated).", reminderDetails)
}

// 23. Proactive Contextual Reminders (Simulated - Triggering)
func (a *Agent) TriggerContextualReminder(currentContext string) string {
	reminder, exists := a.ContextMemory["reminder"]
	if exists && strings.Contains(strings.ToLower(currentContext), strings.ToLower(reminder)) {
		delete(a.ContextMemory, "reminder") // Clear reminder after triggering
		return fmt.Sprintf("Contextual Reminder triggered! Context: '%s'. Reminder was: '%s'", currentContext, reminder)
	}
	return "No contextual reminder triggered for current context (Simulated)."
}

// 24. Agent Self-Improvement Learning (Simulated)
func (a *Agent) SimulateSelfImprovement(feedback string) string {
	messageType := TypeUnknown // Assume feedback is related to the last message type (simplified)
	if lastMsgType, ok := a.ContextMemory["lastMessageType"]; ok {
		messageType = MessageType(lastMsgType)
	}

	if messageType != TypeUnknown {
		a.LearningData[messageType] = append(a.LearningData[messageType], feedback)
		return fmt.Sprintf("Agent learning from feedback on '%s' message type. Thanks for your input!", messageType)
	} else {
		return "Agent received feedback but cannot associate it with a specific message type. Please provide context."
	}
}

// 25. Interactive Visual Art Generation (Simulated)
func (a *Agent) GenerateVisualArt(theme string) string {
	artStyles := []string{"Abstract", "Impressionist", "Surrealist", "Pop Art"}
	artStyle := artStyles[rand.Intn(len(artStyles))]
	artDescription := fmt.Sprintf("Simulated %s style visual art based on theme '%s'. (Visual output is text-based description).", artStyle, theme)
	return fmt.Sprintf("Generated visual art (%s style) for theme '%s':\n%s", artStyle, theme, artDescription)
}


func main() {
	rand.Seed(time.Now().UnixNano()) // Seed random for variety

	agent := NewAgent("SynergyOS")

	// Example MCP message processing
	messages := []Message{
		{Type: TypeTellStory, Payload: "Genre: Sci-Fi, Theme: Space Exploration"},
		{Type: TypeRecommendSkill, Payload: "I enjoy building web applications."},
		{Type: TypeAnalyzeSentimentTrend, Payload: "cryptocurrency"},
		{Type: TypeStoreContext, Payload: "userName:Alice"},
		{Type: TypeRetrieveContext, Payload: "userName"},
		{Type: TypeSummarizeNews, Payload: "technology, AI"},
		{Type: TypeGenerateCodeSnippet, Payload: "function to calculate factorial in Python"},
		{Type: TypeGenerateRecipe, Payload: "vegetarian, spicy"},
		{Type: TypeCuratePlaylist, Payload: "relaxing"},
		{Type: TypeDetectBehaviorAnomaly, Payload: "User accessed sensitive files at 3 AM"},
		{Type: TypePrioritizeTasks, Payload: "Grocery Shopping, Pay Bills, Book Appointment"},
		{Type: TypeGenerateSimulation, Payload: "stock market crash"},
		{Type: TypePlanTravelItinerary, Payload: "budget travel, beaches, europe"},
		{Type: TypeGenerateAutomationScript, Payload: "turn on living room lights at sunset"},
		{Type: TypeProcessMultiModalInput, Payload: "Image description: sunset over a mountain range"},
		{Type: TypeSetContextualReminder, Payload: "Meeting with John"},
		{Type: TypeTriggerContextualReminder, Payload: "I am now in the office"},
		{Type: TypeSimulateSelfImprovement, Payload: "The story was a bit too generic."},
		{Type: TypeGenerateVisualArt, Payload: "Theme: Serenity"},
		{Type: TypeCheckEthicalImplications, Payload: "How to build a harmless AI?"}, // Ethical check example
		{Type: TypePredictWellnessRisk, Payload: "lifestyle data here..."},
		{Type: TypeGenerateLearningPath, Payload: "Become a data scientist"},
		{Type: TypeAdaptLanguageStyle, Payload: "User style seems informal"},
		{Type: TypeForgetContext, Payload: "userName"}, // Forget stored username
		{Type: TypeRetrieveContext, Payload: "userName"}, // Try to retrieve again
		{Type: TypeGenerateVisualization, Payload: "Sales data for Q3 2023"},
		{Type: TypeUnknown, Payload: "This is an unknown message"}, // Unknown message type
	}

	for _, msg := range messages {
		response := agent.ProcessMessage(msg)
		fmt.Printf("Agent '%s' response: %s\n\n", agent.Name, response)
	}
}
```