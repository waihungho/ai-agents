```go
/*
# AI-Agent with MCP Interface in Golang

**Outline:**

1. **Function Summary:**
    * **Content Generation & Creative Tasks:**
        * `GenerateCreativeText(prompt string)`: Generates creative text formats (poems, code, scripts, musical pieces, email, letters, etc.) based on a prompt.
        * `GenerateImageDescription(imageURL string)`: Analyzes an image from URL and generates a descriptive caption.
        * `ComposeMusic(genre string, mood string)`:  Composes a short musical piece based on specified genre and mood.
        * `GenerateRecipe(ingredients []string, cuisine string)`: Creates a recipe using given ingredients and specified cuisine type.
        * `DesignLogo(companyName string, industry string)`:  Generates a textual description and potentially a simplified visual representation (ASCII art or basic SVG string) for a logo.
    * **Data Analysis & Insights:**
        * `AnalyzeSentiment(text string)`:  Performs sentiment analysis on text and returns the sentiment (positive, negative, neutral).
        * `DetectTrends(data []interface{}, dataType string)`: Analyzes a dataset (e.g., time series data, text data) and identifies emerging trends.
        * `SummarizeText(text string, length string)`: Summarizes a given text to a specified length (short, medium, long).
        * `ExtractKeywords(text string, numKeywords int)`: Extracts the most relevant keywords from a given text.
        * `TranslateLanguage(text string, targetLanguage string)`: Translates text from a detected source language to a target language.
    * **Personalized & Context-Aware Functions:**
        * `CreatePersonalizedNewsSummary(interests []string, sources []string)`: Generates a personalized news summary based on user interests and preferred sources.
        * `SmartScheduleAssistant(events []string, constraints []string)`:  Helps schedule events considering constraints and optimizes for time and resource allocation.
        * `ContextAwareReminder(task string, context string)`: Sets a reminder that is context-aware (e.g., reminds when user is near a location, or at a specific time and location).
        * `PersonalizedRecommendation(userProfile map[string]interface{}, itemType string)`: Provides personalized recommendations based on user profile and item type (movies, books, products).
        * `LearnUserProfile(interactionData []interface{})`: Learns and updates user profile based on interaction data (e.g., browsing history, preferences).
    * **Advanced & Novel Functions:**
        * `SimulateConversation(topic string, persona string)`: Simulates a conversation on a given topic with a specified persona (e.g., expert, friend, celebrity).
        * `GenerateDataVisualization(data []interface{}, chartType string, description string)`: Generates a textual description and potentially a data visualization (ASCII art or placeholder structure) based on input data and chart type.
        * `PredictNextWord(sentenceFragment string)`: Predicts the most likely next word in a sentence fragment.
        * `DebugCodeSnippet(code string, language string)`: Attempts to debug a code snippet and provides suggestions or identifies potential errors.
        * `GenerateStoryFromKeywords(keywords []string, genre string)`: Creates a short story based on given keywords and specified genre.

2. **MCP Interface Definition:**
    * Uses Go channels for asynchronous message passing.
    * Messages are structured as structs containing `Action`, `Parameters`, and `ResponseChan`.
    * Agent listens on a request channel and sends responses back on the provided response channel.

3. **Agent Structure:**
    * `AIAgent` struct with channels for communication.
    * Methods for each function defined in the summary.
    * `Start()` method to initiate the agent's message processing loop.

4. **Example Usage:**
    * Demonstrates how to create an agent, send requests, and receive responses.

*/

package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"time"
)

// Message defines the structure for communication via MCP
type Message struct {
	Action       string                 `json:"action"`
	Parameters   map[string]interface{} `json:"parameters"`
	ResponseChan chan Message         `json:"-"` // Channel to send response back
	ResponseData map[string]interface{} `json:"responseData,omitempty"` // For carrying response data
	Error        string                 `json:"error,omitempty"`        // For carrying error messages
}

// AIAgent struct to hold channels for communication
type AIAgent struct {
	RequestChan  chan Message
	ResponseChan chan Message // Optional global response channel for broadcast if needed
	// Add any internal state for the agent here if needed (e.g., user profiles, knowledge base)
}

// NewAIAgent creates and initializes a new AI Agent
func NewAIAgent() *AIAgent {
	return &AIAgent{
		RequestChan:  make(chan Message),
		ResponseChan: make(chan Message), // Optional global response channel
	}
}

// Start initiates the AI Agent's message processing loop
func (agent *AIAgent) Start() {
	fmt.Println("AI Agent started and listening for requests...")
	for {
		select {
		case msg := <-agent.RequestChan:
			fmt.Printf("Received request: Action='%s', Parameters='%v'\n", msg.Action, msg.Parameters)
			response := agent.processMessage(msg)
			msg.ResponseChan <- response // Send response back to the specific requester
			// agent.ResponseChan <- response // Send response to global response channel if needed
		}
	}
}

// processMessage routes the message to the appropriate function based on the Action
func (agent *AIAgent) processMessage(msg Message) Message {
	switch msg.Action {
	case "GenerateCreativeText":
		prompt, _ := msg.Parameters["prompt"].(string)
		response := agent.GenerateCreativeText(prompt)
		return agent.createResponseMessage(msg, response)
	case "GenerateImageDescription":
		imageURL, _ := msg.Parameters["imageURL"].(string)
		response := agent.GenerateImageDescription(imageURL)
		return agent.createResponseMessage(msg, response)
	case "ComposeMusic":
		genre, _ := msg.Parameters["genre"].(string)
		mood, _ := msg.Parameters["mood"].(string)
		response := agent.ComposeMusic(genre, mood)
		return agent.createResponseMessage(msg, response)
	case "GenerateRecipe":
		ingredientsInterface, _ := msg.Parameters["ingredients"].([]interface{})
		cuisine, _ := msg.Parameters["cuisine"].(string)
		ingredients := make([]string, len(ingredientsInterface))
		for i, v := range ingredientsInterface {
			ingredients[i] = fmt.Sprint(v) // Convert interface{} to string
		}
		response := agent.GenerateRecipe(ingredients, cuisine)
		return agent.createResponseMessage(msg, response)
	case "DesignLogo":
		companyName, _ := msg.Parameters["companyName"].(string)
		industry, _ := msg.Parameters["industry"].(string)
		response := agent.DesignLogo(companyName, industry)
		return agent.createResponseMessage(msg, response)
	case "AnalyzeSentiment":
		text, _ := msg.Parameters["text"].(string)
		response := agent.AnalyzeSentiment(text)
		return agent.createResponseMessage(msg, response)
	case "DetectTrends":
		dataInterface, _ := msg.Parameters["data"].([]interface{})
		dataType, _ := msg.Parameters["dataType"].(string)
		response := agent.DetectTrends(dataInterface, dataType)
		return agent.createResponseMessage(msg, response)
	case "SummarizeText":
		text, _ := msg.Parameters["text"].(string)
		length, _ := msg.Parameters["length"].(string)
		response := agent.SummarizeText(text, length)
		return agent.createResponseMessage(msg, response)
	case "ExtractKeywords":
		text, _ := msg.Parameters["text"].(string)
		numKeywordsFloat, _ := msg.Parameters["numKeywords"].(float64) // JSON numbers are float64
		numKeywords := int(numKeywordsFloat)
		response := agent.ExtractKeywords(text, numKeywords)
		return agent.createResponseMessage(msg, response)
	case "TranslateLanguage":
		text, _ := msg.Parameters["text"].(string)
		targetLanguage, _ := msg.Parameters["targetLanguage"].(string)
		response := agent.TranslateLanguage(text, targetLanguage)
		return agent.createResponseMessage(msg, response)
	case "CreatePersonalizedNewsSummary":
		interestsInterface, _ := msg.Parameters["interests"].([]interface{})
		sourcesInterface, _ := msg.Parameters["sources"].([]interface{})
		interests := make([]string, len(interestsInterface))
		sources := make([]string, len(sourcesInterface))
		for i, v := range interestsInterface {
			interests[i] = fmt.Sprint(v)
		}
		for i, v := range sourcesInterface {
			sources[i] = fmt.Sprint(v)
		}
		response := agent.CreatePersonalizedNewsSummary(interests, sources)
		return agent.createResponseMessage(msg, response)
	case "SmartScheduleAssistant":
		eventsInterface, _ := msg.Parameters["events"].([]interface{})
		constraintsInterface, _ := msg.Parameters["constraints"].([]interface{})
		events := make([]string, len(eventsInterface))
		constraints := make([]string, len(constraintsInterface))
		for i, v := range eventsInterface {
			events[i] = fmt.Sprint(v)
		}
		for i, v := range constraintsInterface {
			constraints[i] = fmt.Sprint(v)
		}
		response := agent.SmartScheduleAssistant(events, constraints)
		return agent.createResponseMessage(msg, response)
	case "ContextAwareReminder":
		task, _ := msg.Parameters["task"].(string)
		context, _ := msg.Parameters["context"].(string)
		response := agent.ContextAwareReminder(task, context)
		return agent.createResponseMessage(msg, response)
	case "PersonalizedRecommendation":
		userProfile, _ := msg.Parameters["userProfile"].(map[string]interface{})
		itemType, _ := msg.Parameters["itemType"].(string)
		response := agent.PersonalizedRecommendation(userProfile, itemType)
		return agent.createResponseMessage(msg, response)
	case "LearnUserProfile":
		interactionDataInterface, _ := msg.Parameters["interactionData"].([]interface{})
		response := agent.LearnUserProfile(interactionDataInterface)
		return agent.createResponseMessage(msg, response)
	case "SimulateConversation":
		topic, _ := msg.Parameters["topic"].(string)
		persona, _ := msg.Parameters["persona"].(string)
		response := agent.SimulateConversation(topic, persona)
		return agent.createResponseMessage(msg, response)
	case "GenerateDataVisualization":
		dataInterface, _ := msg.Parameters["data"].([]interface{})
		chartType, _ := msg.Parameters["chartType"].(string)
		description, _ := msg.Parameters["description"].(string)
		response := agent.GenerateDataVisualization(dataInterface, chartType, description)
		return agent.createResponseMessage(msg, response)
	case "PredictNextWord":
		sentenceFragment, _ := msg.Parameters["sentenceFragment"].(string)
		response := agent.PredictNextWord(sentenceFragment)
		return agent.createResponseMessage(msg, response)
	case "DebugCodeSnippet":
		code, _ := msg.Parameters["code"].(string)
		language, _ := msg.Parameters["language"].(string)
		response := agent.DebugCodeSnippet(code, language)
		return agent.createResponseMessage(msg, response)
	case "GenerateStoryFromKeywords":
		keywordsInterface, _ := msg.Parameters["keywords"].([]interface{})
		genre, _ := msg.Parameters["genre"].(string)
		keywords := make([]string, len(keywordsInterface))
		for i, v := range keywordsInterface {
			keywords[i] = fmt.Sprint(v)
		}
		response := agent.GenerateStoryFromKeywords(keywords, genre)
		return agent.createResponseMessage(msg, response)
	default:
		return agent.createErrorMessage(msg, "Unknown action: "+msg.Action)
	}
}

// createResponseMessage helper function to create a response message
func (agent *AIAgent) createResponseMessage(requestMsg Message, responseData map[string]interface{}) Message {
	return Message{
		Action:       requestMsg.Action,
		Parameters:   requestMsg.Parameters,
		ResponseChan: requestMsg.ResponseChan, // Keep the original response channel
		ResponseData: responseData,
	}
}

// createErrorMessage helper function to create an error response message
func (agent *AIAgent) createErrorMessage(requestMsg Message, errorMessage string) Message {
	return Message{
		Action:       requestMsg.Action,
		Parameters:   requestMsg.Parameters,
		ResponseChan: requestMsg.ResponseChan, // Keep the original response channel
		Error:        errorMessage,
	}
}

// ---------------------- Function Implementations (AI Agent Core Logic) ----------------------

// GenerateCreativeText - Generates creative text formats
func (agent *AIAgent) GenerateCreativeText(prompt string) map[string]interface{} {
	fmt.Println("Generating creative text for prompt:", prompt)
	time.Sleep(1 * time.Second) // Simulate processing time
	responses := []string{
		"Once upon a time in a digital forest...",
		"The code whispered secrets to the silicon...",
		"In the realm of algorithms, a new story began...",
	}
	randomIndex := rand.Intn(len(responses))
	return map[string]interface{}{"text": responses[randomIndex]}
}

// GenerateImageDescription - Analyzes image and generates description
func (agent *AIAgent) GenerateImageDescription(imageURL string) map[string]interface{} {
	fmt.Println("Generating image description for URL:", imageURL)
	time.Sleep(1 * time.Second)
	descriptions := []string{
		"A vibrant landscape with a sunset.",
		"A modern city skyline at night.",
		"A close-up of a flower in bloom.",
	}
	randomIndex := rand.Intn(len(descriptions))
	return map[string]interface{}{"description": descriptions[randomIndex]}
}

// ComposeMusic - Composes a short musical piece
func (agent *AIAgent) ComposeMusic(genre string, mood string) map[string]interface{} {
	fmt.Printf("Composing music of genre '%s' and mood '%s'\n", genre, mood)
	time.Sleep(1 * time.Second)
	musicSnippets := []string{
		"🎵 ... music snippet 1 ... 🎵",
		"🎶 ... music snippet 2 ... 🎶",
		"🎼 ... music snippet 3 ... 🎼",
	}
	randomIndex := rand.Intn(len(musicSnippets))
	return map[string]interface{}{"music": musicSnippets[randomIndex]}
}

// GenerateRecipe - Creates a recipe
func (agent *AIAgent) GenerateRecipe(ingredients []string, cuisine string) map[string]interface{} {
	fmt.Printf("Generating recipe for cuisine '%s' with ingredients: %v\n", cuisine, ingredients)
	time.Sleep(1 * time.Second)
	recipes := []string{
		"**Delicious Recipe 1:** ... instructions ...",
		"**Amazing Recipe 2:** ... instructions ...",
		"**Fantastic Recipe 3:** ... instructions ...",
	}
	randomIndex := rand.Intn(len(recipes))
	return map[string]interface{}{"recipe": recipes[randomIndex]}
}

// DesignLogo - Generates a logo description
func (agent *AIAgent) DesignLogo(companyName string, industry string) map[string]interface{} {
	fmt.Printf("Designing logo for company '%s' in industry '%s'\n", companyName, industry)
	time.Sleep(1 * time.Second)
	logoDescriptions := []string{
		"A sleek and modern logo with abstract shapes.",
		"A minimalist logo featuring the company initials.",
		"A vibrant and colorful logo representing innovation.",
	}
	randomIndex := rand.Intn(len(logoDescriptions))
	return map[string]interface{}{"logo_description": logoDescriptions[randomIndex]}
}

// AnalyzeSentiment - Performs sentiment analysis
func (agent *AIAgent) AnalyzeSentiment(text string) map[string]interface{} {
	fmt.Println("Analyzing sentiment of text:", text)
	time.Sleep(1 * time.Second)
	sentiments := []string{"Positive", "Negative", "Neutral"}
	randomIndex := rand.Intn(len(sentiments))
	return map[string]interface{}{"sentiment": sentiments[randomIndex]}
}

// DetectTrends - Detects trends in data
func (agent *AIAgent) DetectTrends(data []interface{}, dataType string) map[string]interface{} {
	fmt.Printf("Detecting trends in '%s' data: %v\n", dataType, data)
	time.Sleep(1 * time.Second)
	trends := []string{"Upward trend detected", "Downward trend detected", "No significant trend"}
	randomIndex := rand.Intn(len(trends))
	return map[string]interface{}{"trends": trends[randomIndex]}
}

// SummarizeText - Summarizes text to a specified length
func (agent *AIAgent) SummarizeText(text string, length string) map[string]interface{} {
	fmt.Printf("Summarizing text to length '%s'\n", length)
	time.Sleep(1 * time.Second)
	summaries := []string{
		"... short summary ...",
		"... medium summary ...",
		"... long summary ...",
	}
	randomIndex := rand.Intn(len(summaries))
	return map[string]interface{}{"summary": summaries[randomIndex]}
}

// ExtractKeywords - Extracts keywords from text
func (agent *AIAgent) ExtractKeywords(text string, numKeywords int) map[string]interface{} {
	fmt.Printf("Extracting %d keywords from text\n", numKeywords)
	time.Sleep(1 * time.Second)
	keywordsList := [][]string{
		{"keyword1", "keyword2", "keyword3"},
		{"important", "relevant", "key"},
		{"algorithm", "data", "AI"},
	}
	randomIndex := rand.Intn(len(keywordsList))
	return map[string]interface{}{"keywords": keywordsList[randomIndex][:numKeywords]} // Return up to numKeywords
}

// TranslateLanguage - Translates text to target language
func (agent *AIAgent) TranslateLanguage(text string, targetLanguage string) map[string]interface{} {
	fmt.Printf("Translating text to '%s'\n", targetLanguage)
	time.Sleep(1 * time.Second)
	translatedTexts := []string{
		"... translated text in language 1 ...",
		"... translated text in language 2 ...",
		"... translated text in language 3 ...",
	}
	randomIndex := rand.Intn(len(translatedTexts))
	return map[string]interface{}{"translated_text": translatedTexts[randomIndex]}
}

// CreatePersonalizedNewsSummary - Creates personalized news summary
func (agent *AIAgent) CreatePersonalizedNewsSummary(interests []string, sources []string) map[string]interface{} {
	fmt.Printf("Creating personalized news summary for interests: %v, sources: %v\n", interests, sources)
	time.Sleep(1 * time.Second)
	newsSummaries := []string{
		"**Personalized News Summary 1:** ... headlines ...",
		"**Personalized News Summary 2:** ... headlines ...",
		"**Personalized News Summary 3:** ... headlines ...",
	}
	randomIndex := rand.Intn(len(newsSummaries))
	return map[string]interface{}{"news_summary": newsSummaries[randomIndex]}
}

// SmartScheduleAssistant - Helps schedule events
func (agent *AIAgent) SmartScheduleAssistant(events []string, constraints []string) map[string]interface{} {
	fmt.Printf("Smart schedule assistant for events: %v, constraints: %v\n", events, constraints)
	time.Sleep(1 * time.Second)
	schedules := []string{
		"**Proposed Schedule 1:** ... events and times ...",
		"**Optimized Schedule 2:** ... events and times ...",
		"**Alternative Schedule 3:** ... events and times ...",
	}
	randomIndex := rand.Intn(len(schedules))
	return map[string]interface{}{"schedule": schedules[randomIndex]}
}

// ContextAwareReminder - Creates context-aware reminder
func (agent *AIAgent) ContextAwareReminder(task string, context string) map[string]interface{} {
	fmt.Printf("Creating context-aware reminder for task '%s' with context '%s'\n", task, context)
	time.Sleep(1 * time.Second)
	return map[string]interface{}{"reminder_status": "Reminder set with context: " + context}
}

// PersonalizedRecommendation - Provides personalized recommendations
func (agent *AIAgent) PersonalizedRecommendation(userProfile map[string]interface{}, itemType string) map[string]interface{} {
	fmt.Printf("Providing personalized recommendation for item type '%s' based on user profile: %v\n", itemType, userProfile)
	time.Sleep(1 * time.Second)
	recommendations := []string{
		"**Recommended Item 1:** ... item details ...",
		"**Top Recommendation 2:** ... item details ...",
		"**Consider Item 3:** ... item details ...",
	}
	randomIndex := rand.Intn(len(recommendations))
	return map[string]interface{}{"recommendation": recommendations[randomIndex]}
}

// LearnUserProfile - Learns and updates user profile
func (agent *AIAgent) LearnUserProfile(interactionDataInterface []interface{}) map[string]interface{} {
	fmt.Printf("Learning user profile from interaction data: %v\n", interactionDataInterface)
	time.Sleep(1 * time.Second)
	return map[string]interface{}{"profile_update_status": "User profile updated based on new interactions."}
}

// SimulateConversation - Simulates a conversation
func (agent *AIAgent) SimulateConversation(topic string, persona string) map[string]interface{} {
	fmt.Printf("Simulating conversation on topic '%s' with persona '%s'\n", topic, persona)
	time.Sleep(1 * time.Second)
	conversationSnippets := []string{
		"**Agent:** ... response 1 ... **User:** ... response 2 ...",
		"**Persona:** ... witty remark ... **You:** ... question ...",
		"**Expert:** ... detailed explanation ... **Listener:** ... follow-up ...",
	}
	randomIndex := rand.Intn(len(conversationSnippets))
	return map[string]interface{}{"conversation": conversationSnippets[randomIndex]}
}

// GenerateDataVisualization - Generates data visualization description
func (agent *AIAgent) GenerateDataVisualization(data []interface{}, chartType string, description string) map[string]interface{} {
	fmt.Printf("Generating data visualization for chart type '%s' with description '%s'\n", chartType, description)
	time.Sleep(1 * time.Second)
	visualizationDescriptions := []string{
		"A bar chart showing comparison of categories.",
		"A line graph illustrating trends over time.",
		"A pie chart representing proportions of parts to a whole.",
	}
	randomIndex := rand.Intn(len(visualizationDescriptions))
	return map[string]interface{}{"visualization_description": visualizationDescriptions[randomIndex]}
}

// PredictNextWord - Predicts the next word in a sentence fragment
func (agent *AIAgent) PredictNextWord(sentenceFragment string) map[string]interface{} {
	fmt.Printf("Predicting next word for sentence fragment: '%s'\n", sentenceFragment)
	time.Sleep(1 * time.Second)
	nextWords := []string{"is", "will", "could", "might", "should"}
	randomIndex := rand.Intn(len(nextWords))
	return map[string]interface{}{"next_word_prediction": nextWords[randomIndex]}
}

// DebugCodeSnippet - Attempts to debug a code snippet
func (agent *AIAgent) DebugCodeSnippet(code string, language string) map[string]interface{} {
	fmt.Printf("Debugging code snippet in language '%s'\n", language)
	time.Sleep(1 * time.Second)
	debugSuggestions := []string{
		"Possible syntax error in line 5.",
		"Check variable initialization before use.",
		"Consider using more descriptive variable names.",
	}
	randomIndex := rand.Intn(len(debugSuggestions))
	return map[string]interface{}{"debug_suggestions": debugSuggestions[randomIndex]}
}

// GenerateStoryFromKeywords - Creates a story from keywords
func (agent *AIAgent) GenerateStoryFromKeywords(keywords []string, genre string) map[string]interface{} {
	fmt.Printf("Generating story from keywords: %v, genre: '%s'\n", keywords, genre)
	time.Sleep(1 * time.Second)
	storyStarters := []string{
		"In a world filled with...",
		"The journey began with...",
		"They discovered a secret...",
	}
	randomIndex := rand.Intn(len(storyStarters))
	story := storyStarters[randomIndex] + " " + strings.Join(keywords, ", ") + "... (story continues)"
	return map[string]interface{}{"story": story}
}

// ---------------------- Main Function (Example Usage) ----------------------

func main() {
	rand.Seed(time.Now().UnixNano()) // Seed random for varied responses

	agent := NewAIAgent()
	go agent.Start() // Start the agent in a goroutine to listen for requests

	// Example Request 1: Generate Creative Text
	requestChan1 := make(chan Message)
	agent.RequestChan <- Message{
		Action: "GenerateCreativeText",
		Parameters: map[string]interface{}{
			"prompt": "Write a short poem about a robot learning to love.",
		},
		ResponseChan: requestChan1,
	}
	response1 := <-requestChan1
	fmt.Println("\nResponse 1 (Creative Text):")
	if response1.Error != "" {
		fmt.Println("Error:", response1.Error)
	} else {
		fmt.Println("Generated Text:", response1.ResponseData["text"])
	}

	// Example Request 2: Analyze Sentiment
	requestChan2 := make(chan Message)
	agent.RequestChan <- Message{
		Action: "AnalyzeSentiment",
		Parameters: map[string]interface{}{
			"text": "This is a truly amazing and wonderful day!",
		},
		ResponseChan: requestChan2,
	}
	response2 := <-requestChan2
	fmt.Println("\nResponse 2 (Sentiment Analysis):")
	if response2.Error != "" {
		fmt.Println("Error:", response2.Error)
	} else {
		fmt.Println("Sentiment:", response2.ResponseData["sentiment"])
	}

	// Example Request 3: Generate Recipe
	requestChan3 := make(chan Message)
	agent.RequestChan <- Message{
		Action: "GenerateRecipe",
		Parameters: map[string]interface{}{
			"ingredients": []string{"chicken", "rice", "soy sauce"},
			"cuisine":     "Asian",
		},
		ResponseChan: requestChan3,
	}
	response3 := <-requestChan3
	fmt.Println("\nResponse 3 (Recipe Generation):")
	if response3.Error != "" {
		fmt.Println("Error:", response3.Error)
	} else {
		fmt.Println("Generated Recipe:", response3.ResponseData["recipe"])
	}

	// Example Request 4: Debug Code Snippet
	requestChan4 := make(chan Message)
	agent.RequestChan <- Message{
		Action: "DebugCodeSnippet",
		Parameters: map[string]interface{}{
			"code":     "print('Hello world'", // Intentionally broken Python code
			"language": "python",
		},
		ResponseChan: requestChan4,
	}
	response4 := <-requestChan4
	fmt.Println("\nResponse 4 (Debug Code):")
	if response4.Error != "" {
		fmt.Println("Error:", response4.Error)
	} else {
		fmt.Println("Debug Suggestions:", response4.ResponseData["debug_suggestions"])
	}

	time.Sleep(2 * time.Second) // Keep agent running for a bit to process requests
	fmt.Println("\nExample requests completed. Agent continues to listen for requests.")
}
```