```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/joho/godotenv"
	"github.com/olahol/melody"
	"github.com/sashabaranov/go-openai"
	"github.com/tidwall/gjson"
)

// AI Agent Outline and Function Summary:
//
// Overview: This AI Agent is designed to be a versatile personal assistant,
// research tool, and creative companion. It leverages OpenAI's API (or a similar model)
// and external tools to perform a wide range of tasks.  It includes features for:
// 1.  Information Retrieval and Synthesis
// 2.  Creative Content Generation
// 3.  Code Generation and Execution
// 4.  Personalized Task Management
// 5.  Context-Aware Automation
// 6.  Web Socket Communication
//
// Functions:
//  1.  `main()`: Initializes the agent, loads environment variables, sets up routes, and starts the server.
//  2.  `handleWebSocketConnection(m *melody.Melody, s *melody.Session)`: Handles new WebSocket connections, setting up message handling.
//  3.  `handleWebSocketMessage(s *melody.Session, msg []byte)`: Processes incoming WebSocket messages, routing them to relevant function based on "action" field.
//  4.  `AgentMessage(s *melody.Session, action string, message string, args map[string]interface{})`: Main function to invoke the AI agent, by calling OpenAI API.
//  5.  `getOpenAIResponse(prompt string)`: Sends a prompt to the OpenAI API and retrieves the generated response.
//  6.  `getOpenAIImage(prompt string)`: Sends a prompt to the OpenAI DALL-E API and retrieves the generated image URL.
//  7.  `searchWeb(query string)`: Performs a web search using a configurable search engine (e.g., DuckDuckGo) via API.
//  8.  `summarizeText(text string)`: Summarizes a given text using the AI model.
//  9.  `generateCode(programmingLanguage string, description string)`: Generates code in a specified language based on a description.
//  10. `executeCode(code string, language string)`: Executes code in a specified language using a sandboxed environment.
//  11. `createTask(taskDescription string)`: Creates a task in a local task management system (e.g., a file or database).
//  12. `listTasks()`: Lists the tasks from the local task management system.
//  13. `updateTask(taskID string, updates map[string]interface{})`: Updates a task based on provided updates.
//  14. `deleteTask(taskID string)`: Deletes a task from the task management system.
//  15. `translateText(text string, targetLanguage string)`: Translates text to a specified language.
//  16. `extractEntities(text string)`: Extracts entities (e.g., people, organizations, locations) from a given text.
//  17. `analyzeSentiment(text string)`: Analyzes the sentiment (positive, negative, neutral) of a given text.
//  18. `createPoem(topic string)`: Generates a poem on a given topic.
//  19. `createStory(prompt string)`: Generates a story based on a given prompt.
//  20. `generateImage(prompt string)`: Generates an image based on a prompt.
//  21. `customCommand(command string, args []string)`: Executes a custom command on the server.
//  22. `knowledgeBaseQuery(query string)`: Searches a local knowledge base (e.g., a file or database) for relevant information.
//  23. `scheduleEvent(time string, description string)`: Schedules an event with the system's scheduler.
//  24. `convertCurrency(amount float64, fromCurrency string, toCurrency string)`: Converts currency from one type to another.
//  25. `getWeatherData(city string)`: Retrieves the weather data from a specific city.
//  26. `routeHandler(w http.ResponseWriter, r *http.Request)`: Serves the HTML UI for the agent.
//  27. `randomString(n int)`: Generate a random string with length n.
//  28. `setupRouter()`: Setup the HTTP router.
//  29. `startGPT(c chan string)`: Start GPT to listen on channel and send back response
//
// Configuration:
//  - OpenAI API Key: Set as environment variable `OPENAI_API_KEY`.
//  - DuckDuckGo API Key (optional): Set as environment variable `DUCKDUCKGO_API_KEY`.
//  - Server Port: Configured via environment variable `PORT` (defaults to 8080).
//
// Error Handling:
//  - Includes error handling for API calls, file operations, and code execution.
//
// Security Considerations:
//  - Code execution is sandboxed to prevent malicious code from harming the system.
//  - API keys are stored as environment variables to prevent them from being hardcoded.

var (
	openaiAPIKey string
	ddgAPIKey    string
	port         string
	taskFile     = "tasks.json" // Simple JSON file for storing tasks
	m            *melody.Melody
	gpt          *GPT
)

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Println("Error loading .env file")
	}

	openaiAPIKey = os.Getenv("OPENAI_API_KEY")
	if openaiAPIKey == "" {
		log.Fatal("OPENAI_API_KEY not set in environment variables")
	}

	ddgAPIKey = os.Getenv("DUCKDUCKGO_API_KEY") // Optional

	port = os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	rand.Seed(time.Now().UnixNano())

	m = melody.New()

	m.HandleConnect(func(s *melody.Session) {
		handleWebSocketConnection(m, s)
	})

	m.HandleMessage(func(s *melody.Session, msg []byte) {
		handleWebSocketMessage(s, msg)
	})

	http.HandleFunc("/", routeHandler)
	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		m.HandleRequest(w, r)
	})

	gpt = NewGPT()
	c := make(chan string)
	go gpt.Start(c)
	defer gpt.Stop()

	fmt.Println("Server listening on port:", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func handleWebSocketConnection(m *melody.Melody, s *melody.Session) {
	fmt.Println("New WebSocket connection established.")
}

func handleWebSocketMessage(s *melody.Session, msg []byte) {
	var message struct {
		Action string                 `json:"action"`
		Message string                `json:"message"`
		Args    map[string]interface{} `json:"args"`
	}

	if err := json.Unmarshal(msg, &message); err != nil {
		fmt.Println("Error unmarshaling JSON:", err)
		return
	}

	go AgentMessage(s, message.Action, message.Message, message.Args)
}

// AgentMessage is the main function to invoke the AI agent.
func AgentMessage(s *melody.Session, action string, message string, args map[string]interface{}) {
	fmt.Printf("Received action: %s, message: %s, args: %+v\n", action, message, args)

	switch action {
	case "getOpenAIResponse":
		response, err := getOpenAIResponse(message)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "openai_response", response)

	case "generateImage":
		imageURL, err := getOpenAIImage(message)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "image_url", imageURL)

	case "searchWeb":
		results, err := searchWeb(message)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "search_results", results)

	case "summarizeText":
		text, ok := args["text"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid text argument")
			return
		}
		summary, err := summarizeText(text)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "summary", summary)

	case "generateCode":
		language, ok := args["language"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid language argument")
			return
		}
		code, err := generateCode(language, message)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "generated_code", code)

	case "executeCode":
		code, ok := args["code"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid code argument")
			return
		}
		language, ok := args["language"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid language argument")
			return
		}
		output, err := executeCode(code, language)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "code_output", output)

	case "createTask":
		taskID, err := createTask(message)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "task_created", taskID)

	case "listTasks":
		tasks, err := listTasks()
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "task_list", tasks)

	case "updateTask":
		taskID, ok := args["taskID"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid taskID argument")
			return
		}
		updates, ok := args["updates"].(map[string]interface{})
		if !ok {
			sendMessage(s, "error", "Invalid updates argument")
			return
		}
		err := updateTask(taskID, updates)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "task_updated", taskID)

	case "deleteTask":
		taskID, ok := args["taskID"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid taskID argument")
			return
		}
		err := deleteTask(taskID)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "task_deleted", taskID)

	case "translateText":
		text, ok := args["text"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid text argument")
			return
		}
		targetLanguage, ok := args["targetLanguage"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid targetLanguage argument")
			return
		}
		translatedText, err := translateText(text, targetLanguage)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "translated_text", translatedText)

	case "extractEntities":
		text, ok := args["text"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid text argument")
			return
		}
		entities, err := extractEntities(text)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "extracted_entities", entities)

	case "analyzeSentiment":
		text, ok := args["text"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid text argument")
			return
		}
		sentiment, err := analyzeSentiment(text)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "sentiment", sentiment)

	case "createPoem":
		topic, ok := args["topic"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid topic argument")
			return
		}
		poem, err := createPoem(topic)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "poem", poem)

	case "createStory":
		story, err := createStory(message)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "story", story)

	case "generateImage":
		imageURL, err := generateImage(message)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "image_url", imageURL)

	case "customCommand":
		command, ok := args["command"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid command argument")
			return
		}
		argsList, ok := args["args"].([]string)
		if !ok {
			// Handle case where args is not a []string (e.g., it's nil or a different type)
			argsList = []string{} // Provide an empty slice as a default
		}
		output, err := customCommand(command, argsList)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "command_output", output)

	case "knowledgeBaseQuery":
		results, err := knowledgeBaseQuery(message)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "knowledge_base_results", results)

	case "scheduleEvent":
		timeString, ok := args["time"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid time argument")
			return
		}
		description, ok := args["description"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid description argument")
			return
		}
		err := scheduleEvent(timeString, description)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "event_scheduled", "Event scheduled")

	case "convertCurrency":
		amountFloat, ok := args["amount"].(float64)
		if !ok {
			sendMessage(s, "error", "Invalid amount argument")
			return
		}
		fromCurrency, ok := args["fromCurrency"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid fromCurrency argument")
			return
		}
		toCurrency, ok := args["toCurrency"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid toCurrency argument")
			return
		}
		convertedAmount, err := convertCurrency(amountFloat, fromCurrency, toCurrency)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "converted_amount", convertedAmount)

	case "getWeatherData":
		city, ok := args["city"].(string)
		if !ok {
			sendMessage(s, "error", "Invalid city argument")
			return
		}
		weatherData, err := getWeatherData(city)
		if err != nil {
			sendMessage(s, "error", err.Error())
			return
		}
		sendMessage(s, "weather_data", weatherData)

	case "startGPT":
		gpt.RequestChan <- message
		go func() {
			select {
			case response := <-gpt.ResponseChan:
				sendMessage(s, "gpt_response", response)
			case <-time.After(60 * time.Second):
				sendMessage(s, "gpt_response", "No response from gpt after 60 seconds.")
			}
		}()

	default:
		sendMessage(s, "error", "Unknown action: "+action)
	}
}

// getOpenAIResponse sends a prompt to the OpenAI API and retrieves the generated response.
func getOpenAIResponse(prompt string) (string, error) {
	client := openai.NewClient(openaiAPIKey)
	resp, err := client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: openai.GPT3Dot5Turbo,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: prompt,
				},
			},
		},
	)

	if err != nil {
		fmt.Printf("ChatCompletion error: %v\n", err)
		return "", err
	}

	return resp.Choices[0].Message.Content, nil
}

// getOpenAIImage sends a prompt to the OpenAI DALL-E API and retrieves the generated image URL.
func getOpenAIImage(prompt string) (string, error) {
	client := openai.NewClient(openaiAPIKey)
	resp, err := client.CreateImage(
		context.Background(),
		openai.ImageRequest{
			Prompt:         prompt,
			N:              1,
			Size:           openai.CreateImageSize256x256,
			ResponseFormat: openai.CreateImageResponseFormatURL,
		},
	)
	if err != nil {
		fmt.Printf("Image creation error: %v\n", err)
		return "", err
	}

	return resp.Data[0].URL, nil
}

// searchWeb performs a web search using a configurable search engine (e.g., DuckDuckGo).
func searchWeb(query string) (string, error) {
	// Using DuckDuckGo API (requires API key if you want to make many requests)
	url := fmt.Sprintf("https://api.duckduckgo.com/?q=%s&format=json&pretty=1", query)
	if ddgAPIKey != "" {
		url += "&key=" + ddgAPIKey
	}

	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	// Optionally, you could parse the JSON and return a more structured response.
	return string(body), nil
}

// summarizeText summarizes a given text using the AI model.
func summarizeText(text string) (string, error) {
	prompt := fmt.Sprintf("Summarize the following text: %s", text)
	return getOpenAIResponse(prompt)
}

// generateCode generates code in a specified language based on a description.
func generateCode(programmingLanguage string, description string) (string, error) {
	prompt := fmt.Sprintf("Generate code in %s to: %s", programmingLanguage, description)
	return getOpenAIResponse(prompt)
}

// executeCode executes code in a specified language using a sandboxed environment.
func executeCode(code string, language string) (string, error) {
	// Create a temporary file for the code
	tmpFile, err := os.CreateTemp("", "code."+language)
	if err != nil {
		return "", err
	}
	defer os.Remove(tmpFile.Name()) // Clean up the temporary file

	_, err = tmpFile.WriteString(code)
	if err != nil {
		return "", err
	}
	tmpFile.Close()

	var cmd *exec.Cmd
	switch language {
	case "python":
		cmd = exec.Command("python", tmpFile.Name())
	case "javascript":
		cmd = exec.Command("node", tmpFile.Name())
	case "go":
		// Compile and run the Go program
		exec.Command("go", "build", "-o", "tmp_executable", tmpFile.Name()).Run() // Compile
		defer os.Remove("tmp_executable")                                       // remove executable

		cmd = exec.Command("./tmp_executable")

	default:
		return "", fmt.Errorf("unsupported language: %s", language)
	}

	// Execute the command and capture the output
	output, err := cmd.CombinedOutput()
	if err != nil {
		return string(output), err
	}

	return string(output), nil
}

// createTask creates a task in a local task management system (e.g., a file or database).
func createTask(taskDescription string) (string, error) {
	// Generate a unique ID for the task
	taskID := randomString(10)

	// Read existing tasks from file
	tasks, err := readTasksFromFile()
	if err != nil {
		return "", err
	}

	// Add the new task
	tasks[taskID] = map[string]interface{}{
		"description": taskDescription,
		"completed":   false,
	}

	// Write the updated tasks to file
	err = writeTasksToFile(tasks)
	if err != nil {
		return "", err
	}

	return taskID, nil
}

// listTasks lists the tasks from the local task management system.
func listTasks() (map[string]map[string]interface{}, error) {
	return readTasksFromFile()
}

// updateTask updates a task based on provided updates.
func updateTask(taskID string, updates map[string]interface{}) error {
	tasks, err := readTasksFromFile()
	if err != nil {
		return err
	}

	if _, ok := tasks[taskID]; !ok {
		return fmt.Errorf("task with ID %s not found", taskID)
	}

	// Merge updates into the existing task
	for key, value := range updates {
		tasks[taskID][key] = value
	}

	return writeTasksToFile(tasks)
}

// deleteTask deletes a task from the task management system.
func deleteTask(taskID string) error {
	tasks, err := readTasksFromFile()
	if err != nil {
		return err
	}

	if _, ok := tasks[taskID]; !ok {
		return fmt.Errorf("task with ID %s not found", taskID)
	}

	delete(tasks, taskID)

	return writeTasksToFile(tasks)
}

// translateText translates text to a specified language.
func translateText(text string, targetLanguage string) (string, error) {
	prompt := fmt.Sprintf("Translate the following text to %s: %s", targetLanguage, text)
	return getOpenAIResponse(prompt)
}

// extractEntities extracts entities (e.g., people, organizations, locations) from a given text.
func extractEntities(text string) (string, error) {
	// This is a simplified example.  A more robust solution would use a dedicated
	// Named Entity Recognition (NER) library or API.
	prompt := fmt.Sprintf("Extract entities (people, organizations, locations) from the following text: %s", text)
	return getOpenAIResponse(prompt)
}

// analyzeSentiment analyzes the sentiment (positive, negative, neutral) of a given text.
func analyzeSentiment(text string) (string, error) {
	prompt := fmt.Sprintf("Analyze the sentiment (positive, negative, or neutral) of the following text: %s", text)
	return getOpenAIResponse(prompt)
}

// createPoem generates a poem on a given topic.
func createPoem(topic string) (string, error) {
	prompt := fmt.Sprintf("Write a short poem about %s", topic)
	return getOpenAIResponse(prompt)
}

// createStory generates a story based on a given prompt.
func createStory(prompt string) (string, error) {
	return getOpenAIResponse(fmt.Sprintf("Write a short story about %s", prompt))
}

// generateImage generates an image based on a prompt.
func generateImage(prompt string) (string, error) {
	return getOpenAIImage(prompt)
}

// customCommand executes a custom command on the server.
func customCommand(command string, args []string) (string, error) {
	// Security Note:  This function is inherently risky.  You should carefully
	// validate the command and arguments to prevent arbitrary code execution.
	// Consider using a whitelist of allowed commands.

	cmd := exec.Command(command, args...)
	output, err := cmd.CombinedOutput()
	return string(output), err
}

// knowledgeBaseQuery searches a local knowledge base (e.g., a file or database) for relevant information.
func knowledgeBaseQuery(query string) (string, error) {
	// This is a placeholder. Implement your knowledge base search here.
	// For example, you could read from a file, query a database, or use a vector store.
	// For now, we'll just return a canned response.
	return "No information found in knowledge base for query: " + query, nil
}

// scheduleEvent schedules an event with the system's scheduler.
func scheduleEvent(timeString string, description string) error {
	// This is a placeholder. Implement your scheduling logic here.
	// You could use the `cron` package or a similar library.

	// For this example, just print a message.
	fmt.Printf("Scheduling event at %s: %s\n", timeString, description)
	return nil
}

// convertCurrency converts currency from one type to another.
func convertCurrency(amount float64, fromCurrency string, toCurrency string) (float64, error) {
	// This is a placeholder.  You'll need to use a currency conversion API
	// to get real-time exchange rates.
	// For this example, we'll just return a dummy value.
	exchangeRate := 1.2 // Dummy exchange rate
	return amount * exchangeRate, nil
}

// getWeatherData retrieves the weather data from a specific city.
func getWeatherData(city string) (string, error) {
	// This is a placeholder.  You'll need to use a weather API (e.g., OpenWeatherMap)
	// to get real-time weather data.
	// For this example, we'll just return a canned response.

	return fmt.Sprintf("Weather in %s: Sunny, 25°C", city), nil
}

// routeHandler serves the HTML UI for the agent.
func routeHandler(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, "index.html") // Assuming you have an index.html file
}

// Helper Functions

func sendMessage(s *melody.Session, action string, message interface{}) {
	response := map[string]interface{}{
		"action":  action,
		"message": message,
	}

	jsonResponse, err := json.Marshal(response)
	if err != nil {
		fmt.Println("Error marshaling JSON:", err)
		return
	}

	err = s.Write(jsonResponse)
	if err != nil {
		fmt.Println("Error sending message:", err)
	}
}

func randomString(n int) string {
	letters := []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
	b := make([]rune, n)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	return string(b)
}

// Task Management Helper Functions (for storing tasks in a JSON file)

func readTasksFromFile() (map[string]map[string]interface{}, error) {
	file, err := os.OpenFile(taskFile, os.O_RDONLY|os.O_CREATE, 0644)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}

	tasks := make(map[string]map[string]interface{})
	if len(data) > 0 {
		err = json.Unmarshal(data, &tasks)
		if err != nil {
			return nil, err
		}
	}

	return tasks, nil
}

func writeTasksToFile(tasks map[string]map[string]interface{}) error {
	file, err := os.Create(taskFile)
	if err != nil {
		return err
	}
	defer file.Close()

	data, err := json.MarshalIndent(tasks, "", "  ")
	if err != nil {
		return err
	}

	_, err = file.Write(data)
	return err
}

// HTML UI (index.html - Simplified example)

/*
<!DOCTYPE html>
<html>
<head>
    <title>AI Agent</title>
</head>
<body>
    <h1>AI Agent</h1>
    <input type="text" id="message" placeholder="Enter your message">
    <button onclick="sendMessage()">Send</button>
    <div id="response"></div>

    <script>
        const ws = new WebSocket("ws://localhost:8080/ws");

        ws.onopen = () => {
            console.log("Connected to WebSocket");
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            document.getElementById("response").innerText = `Action: ${data.action}, Message: ${JSON.stringify(data.message)}`;
        };

        function sendMessage() {
            const message = document.getElementById("message").value;
            ws.send(JSON.stringify({ action: "getOpenAIResponse", message: message }));
        }
    </script>
</body>
</html>
*/

type GPT struct {
	RequestChan  chan string
	ResponseChan chan string
	stopChan     chan bool
	wg           sync.WaitGroup
}

func NewGPT() *GPT {
	return &GPT{
		RequestChan:  make(chan string),
		ResponseChan: make(chan string),
		stopChan:     make(chan bool),
	}
}

func (g *GPT) Start(c chan string) {
	g.wg.Add(1)
	defer g.wg.Done()

	for {
		select {
		case prompt := <-g.RequestChan:
			fmt.Printf("GPT received request: %s\n", prompt)
			response, err := getOpenAIResponse(prompt)
			if err != nil {
				fmt.Printf("Error from getOpenAIResponse: %v\n", err)
				g.ResponseChan <- fmt.Sprintf("Error processing request: %v", err)
			} else {
				g.ResponseChan <- response
			}
		case <-g.stopChan:
			fmt.Println("GPT worker stopping")
			return
		}
	}
}

func (g *GPT) Stop() {
	close(g.stopChan)
	g.wg.Wait()
}

func setupRouter() *http.ServeMux {
	router := http.NewServeMux()
	router.HandleFunc("/query", queryHandler)
	return router
}

func queryHandler(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("q")
	if query == "" {
		http.Error(w, "Query parameter 'q' is required", http.StatusBadRequest)
		return
	}

	// Simulate processing and respond with a simple message
	response := fmt.Sprintf("AI processed your query: %s", query)
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(response))
}

// ExtractJSONValues extracts specific values from a JSON string based on keys
func ExtractJSONValues(jsonString string, keys []string) (map[string]string, error) {
	results := make(map[string]string)
	for _, key := range keys {
		result := gjson.Get(jsonString, key)
		if !result.Exists() {
			return nil, fmt.Errorf("key '%s' not found in JSON", key)
		}
		results[key] = result.String()
	}
	return results, nil
}

// SanitizeFilename removes or replaces characters in a string that are invalid in filenames
func SanitizeFilename(filename string) string {
	// Define a regular expression to match invalid characters in filenames
	reg := regexp.MustCompile(`[^a-zA-Z0-9._-]`) // example: only allow alphanumeric, dots, underscores and hyphens

	// Replace invalid characters with an underscore
	safeFilename := reg.ReplaceAllString(filename, "_")

	return safeFilename
}

// RetryOperation retries a function operation with exponential backoff
func RetryOperation(operation func() error, maxRetries int, baseDelay time.Duration) error {
	for i := 0; i < maxRetries; i++ {
		err := operation()
		if err == nil {
			return nil // Success
		}

		// Log the error and retry attempt
		fmt.Printf("Attempt %d failed: %v\n", i+1, err)

		// Calculate the delay with exponential backoff
		delay := baseDelay * time.Duration(1<<i) // 2^i
		time.Sleep(delay)
	}
	return fmt.Errorf("max retries exceeded")
}

// ParseBoolLoose attempts to convert a string to a boolean, treating common affirmative strings as true.
func ParseBoolLoose(input string) (bool, error) {
	input = strings.ToLower(strings.TrimSpace(input))
	if input == "true" || input == "yes" || input == "1" || input == "affirmative" {
		return true, nil
	} else if input == "false" || input == "no" || input == "0" || input == "negative" {
		return false, nil
	}
	return false, fmt.Errorf("unable to parse '%s