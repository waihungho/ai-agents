```go
/*
# AI Agent with MCP Interface in Golang

**Outline:**

This Go program implements an AI Agent with a My Control Panel (MCP) interface. The agent is designed to be a versatile personal assistant capable of performing a variety of advanced and trendy tasks.  It interacts with the user through a command-line MCP, allowing users to invoke different functionalities.

**Function Summary (20+ Unique Functions):**

1.  **Personalized News Curator:**  `agent news`:  Fetches and summarizes news articles based on user-defined interests, filtering out noise and presenting relevant information.
2.  **Creative Story Generator:** `agent story <genre>`: Generates original short stories in a specified genre (e.g., sci-fi, fantasy, mystery), leveraging AI storytelling techniques.
3.  **Personalized Recipe Recommendation:** `agent recipe <ingredients...>`: Recommends recipes based on available ingredients and user dietary preferences, considering nutritional value and cuisine types.
4.  **Smart Meeting Scheduler:** `agent schedule <participants...> <duration> <topic>`:  Analyzes participant calendars (simulated for this example), finds optimal meeting slots, and sends out invitations with agenda suggestions.
5.  **Sentiment Analyzer for Text:** `agent sentiment <text>`:  Analyzes the sentiment (positive, negative, neutral) of provided text, useful for understanding emotional tone in communication.
6.  **Trend Forecaster (Simulated):** `agent trend <topic>`: Predicts potential future trends related to a given topic based on simulated data analysis and pattern recognition.
7.  **Personalized Learning Path Generator:** `agent learn <skill>`: Creates a structured learning path with resources (articles, videos, courses) to acquire a new skill, tailored to the user's current knowledge level.
8.  **Code Snippet Generator:** `agent code <language> <task>`: Generates code snippets in a specified programming language to perform a given task, useful for quick prototyping and learning.
9.  **Style Guide Generator (Personalized):** `agent styleguide <theme>`: Generates a personalized style guide for writing, presentations, or even home decor, based on a chosen theme and user preferences.
10. **Mindfulness Prompt Generator:** `agent mindful`: Provides daily mindfulness prompts and exercises to encourage mental well-being and stress reduction.
11. **Habit Tracker and Analyzer:** `agent habit track <habit> <status>` / `agent habit analyze`: Tracks user habits and provides analysis on progress, consistency, and potential improvements.
12. **Personalized Travel Itinerary Planner:** `agent travel <destination> <duration> <interests...>`: Creates a travel itinerary to a given destination, considering duration, interests, budget (simulated), and suggesting attractions and activities.
13. **Creative Brainstorming Partner:** `agent brainstorm <topic>`:  Acts as a brainstorming partner, generating creative ideas and suggestions related to a given topic, pushing beyond conventional thinking.
14. **Personalized Fitness Plan Generator:** `agent fitness <goal> <level> <equipment...>`: Creates a personalized fitness plan based on fitness goals, current level, available equipment (simulated), and preferred workout styles.
15. **Language Translator with Contextual Understanding:** `agent translate <text> <target_language>`: Translates text to a target language, attempting to maintain contextual understanding beyond simple word-for-word translation.
16. **Meeting Minute Summarizer (Simulated Audio Input):** `agent minutes`: (Simulates audio input) Processes a simulated meeting audio transcript and generates concise meeting minutes, highlighting key decisions and action items.
17. **Personalized Music Playlist Generator (Genre/Mood Based):** `agent playlist <genre/mood>`: Generates a music playlist based on a specified genre or mood, discovering new music within user preferences.
18. **Art Style Transfer (Simulated):** `agent artstyle <image_path> <style>`: (Simulates image processing) Applies a specified art style (e.g., Van Gogh, Impressionism) to a given image, demonstrating creative image manipulation.
19. **Task Prioritization and Management:** `agent task add <task>` / `agent task list` / `agent task prioritize`: Manages a task list, allows adding tasks, listing tasks, and prioritizes tasks based on urgency and importance (simulated).
20. **Personalized Book Recommendation Engine:** `agent book <genre/author/theme>`: Recommends books based on genre, author, or theme preferences, suggesting reads that align with user taste.
21. **Dynamic Skill Assessment (Simulated):** `agent assess <skill>`: (Simulates assessment) Provides a simulated dynamic skill assessment based on user input and interaction, identifying strengths and areas for improvement in a chosen skill.
22. **Interactive Story/Game Generator:** `agent game <genre>`: Generates an interactive text-based story or game in a specified genre, allowing user choices to influence the narrative and outcome.

**MCP Interface:**

The MCP interface is command-line based. The user interacts with the agent by typing commands in the format: `agent <function> [arguments...]`.  The agent processes the command and arguments, performs the requested function, and displays the output to the console.

**Implementation Notes:**

*   This is a conceptual outline and a basic implementation.  Full AI functionality for each function would require integration with NLP libraries, machine learning models, and potentially external APIs.
*   For simplicity and demonstration purposes, many functions are implemented with placeholder logic or simulated outputs.
*   Error handling and input validation are included but can be further enhanced for a production-ready system.
*   The MCP interface is kept simple for clarity. A more sophisticated interface (GUI, web-based) could be built upon this foundation.

*/
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"time"
	"math/rand"
	"strconv"
)

// Agent struct represents the AI agent.
type Agent struct {
	userName string
	preferences map[string]interface{} // Placeholder for user preferences
	habits map[string]map[string][]string // Habit tracking data (habit -> date -> status)
	taskList []string
}

// NewAgent creates a new AI Agent instance.
func NewAgent(userName string) *Agent {
	return &Agent{
		userName:    userName,
		preferences: make(map[string]interface{}),
		habits:      make(map[string]map[string][]string),
		taskList:    []string{},
	}
}

// --- Agent Functions (Implementations with Placeholders or Simulations) ---

// 1. Personalized News Curator
func (a *Agent) PersonalizedNewsCurator() {
	fmt.Println("Fetching personalized news based on your interests...")
	interests := []string{"Technology", "World Affairs", "Science", "Space Exploration"} // Example interests - could be personalized
	fmt.Println("Your interests:", strings.Join(interests, ", "))

	newsHeadlines := []string{
		"[Tech] Breakthrough in Quantum Computing!",
		"[World] Geopolitical Tensions Rise in Region X",
		"[Science] New Study Reveals Surprising Link Between Diet and Longevity",
		"[Space] Upcoming Launch to Explore Jupiter's Moons",
		"[Tech] AI Agent Solves Complex Problem - Details Inside!", // Meta!
	}

	fmt.Println("\n--- Personalized News Headlines ---")
	for _, headline := range newsHeadlines {
		if containsInterest(headline, interests) { // Simulate interest filtering
			fmt.Println("- ", headline)
		}
	}
	fmt.Println("----------------------------------")
}

func containsInterest(headline string, interests []string) bool {
	headlineLower := strings.ToLower(headline)
	for _, interest := range interests {
		if strings.Contains(headlineLower, strings.ToLower(interest)) {
			return true
		}
	}
	return false
}


// 2. Creative Story Generator
func (a *Agent) CreativeStoryGenerator(genre string) {
	fmt.Printf("Generating a short story in the '%s' genre...\n", genre)

	storyPrompts := map[string][]string{
		"sci-fi":    {"A lone astronaut discovers...", "In the year 2347...", "The AI awakened and..."},
		"fantasy":   {"In a land of dragons and magic...", "The prophecy foretold...", "A hidden portal opened to..."},
		"mystery":   {"A strange disappearance in a quiet town...", "The detective received an anonymous letter...", "Secrets hidden for years began to surface..."},
		"default":   {"Once upon a time...", "In a world not unlike our own...", "Imagine a place where..."},
	}

	prompts, ok := storyPrompts[strings.ToLower(genre)]
	if !ok {
		prompts = storyPrompts["default"]
		fmt.Println("Genre not recognized, using default prompt.")
	}

	rand.Seed(time.Now().UnixNano())
	prompt := prompts[rand.Intn(len(prompts))]

	story := fmt.Sprintf("%s ... (Story continues - AI generated content would go here)", prompt)
	fmt.Println("\n--- Creative Story ---")
	fmt.Println(story)
	fmt.Println("----------------------")
}

// 3. Personalized Recipe Recommendation
func (a *Agent) PersonalizedRecipeRecommendation(ingredients []string) {
	fmt.Println("Recommending recipes based on ingredients:", strings.Join(ingredients, ", "))

	recipes := map[string][]string{
		"Pasta with Tomato Sauce": {"pasta", "tomato", "garlic", "onion", "olive oil"},
		"Chicken Stir-Fry":       {"chicken", "broccoli", "soy sauce", "ginger", "peppers"},
		"Vegetable Curry":        {"potatoes", "carrots", "peas", "coconut milk", "curry powder"},
		"Omelette":               {"eggs", "cheese", "mushrooms", "spinach"},
	}

	fmt.Println("\n--- Recipe Recommendations ---")
	for recipeName, recipeIngredients := range recipes {
		if containsAllIngredients(recipeIngredients, ingredients) {
			fmt.Println("- ", recipeName, "(Ingredients:", strings.Join(recipeIngredients, ", "), ")")
		}
	}
	fmt.Println("----------------------------")
}

func containsAllIngredients(recipeIngredients, availableIngredients []string) bool {
	for _, recipeIng := range recipeIngredients {
		found := false
		for _, availIng := range availableIngredients {
			if strings.ToLower(recipeIng) == strings.ToLower(availIng) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// 4. Smart Meeting Scheduler (Simulated)
func (a *Agent) SmartMeetingScheduler(participants []string, duration string, topic string) {
	fmt.Println("Scheduling meeting for:", strings.Join(participants, ", "), "Duration:", duration, ", Topic:", topic)

	// Simulate calendar checks - in real world, would integrate with calendar APIs
	availableSlots := []string{"Monday 10:00 AM", "Tuesday 2:00 PM", "Wednesday 11:00 AM"} // Simulated availability
	fmt.Println("Simulated available slots for participants:", availableSlots)

	if len(availableSlots) > 0 {
		fmt.Println("\n--- Suggested Meeting Slot ---")
		fmt.Println("Best available slot:", availableSlots[0]) // Simple suggestion - more sophisticated logic needed in real system
		fmt.Println("Invitation sent to participants (simulated).")
		fmt.Println("-----------------------------")
	} else {
		fmt.Println("No common free slots found. Please adjust participant availability.")
	}
}

// 5. Sentiment Analyzer for Text
func (a *Agent) SentimentAnalyzer(text string) {
	fmt.Println("Analyzing sentiment of text:", text)

	// Very basic sentiment simulation - real sentiment analysis is much more complex
	positiveKeywords := []string{"happy", "great", "amazing", "fantastic", "wonderful", "positive", "good"}
	negativeKeywords := []string{"sad", "bad", "terrible", "awful", "negative", "angry", "frustrated"}

	positiveCount := 0
	negativeCount := 0

	textLower := strings.ToLower(text)
	for _, keyword := range positiveKeywords {
		if strings.Contains(textLower, keyword) {
			positiveCount++
		}
	}
	for _, keyword := range negativeKeywords {
		if strings.Contains(textLower, keyword) {
			negativeCount++
		}
	}

	sentiment := "Neutral"
	if positiveCount > negativeCount {
		sentiment = "Positive"
	} else if negativeCount > positiveCount {
		sentiment = "Negative"
	}

	fmt.Println("\n--- Sentiment Analysis ---")
	fmt.Println("Text:", text)
	fmt.Println("Sentiment:", sentiment)
	fmt.Println("--------------------------")
}

// 6. Trend Forecaster (Simulated)
func (a *Agent) TrendForecaster(topic string) {
	fmt.Println("Forecasting trends for topic:", topic)

	// Simulate trend data and prediction - real trend forecasting uses complex algorithms and data analysis
	simulatedTrends := map[string][]string{
		"Technology": {"AI advancements in healthcare", "Rise of Web3 technologies", "Sustainable tech solutions"},
		"Fashion":    {"Sustainable and eco-friendly fashion", "Metaverse fashion and digital avatars", "Comfort and functionality focused clothing"},
		"Food":       {"Plant-based meat alternatives", "Personalized nutrition and diets", "Global cuisine fusion"},
		"default":    {"Continued focus on sustainability", "Increased personalization in services", "Growing importance of mental well-being"},
	}

	trends, ok := simulatedTrends[strings.ToLower(topic)]
	if !ok {
		trends = simulatedTrends["default"]
		fmt.Println("Topic not specific enough, using general trends.")
	}

	fmt.Println("\n--- Trend Forecast ---")
	fmt.Printf("Potential future trends for '%s':\n", topic)
	for i, trend := range trends {
		fmt.Printf("%d. %s\n", i+1, trend)
	}
	fmt.Println("-----------------------")
}

// 7. Personalized Learning Path Generator
func (a *Agent) PersonalizedLearningPathGenerator(skill string) {
	fmt.Println("Generating learning path for skill:", skill)

	// Simulated learning resources - real system would access curated learning platforms/APIs
	learningResources := map[string][]string{
		"programming": {"Online coding platforms (e.g., Coursera, Udemy, Codecademy)", "Programming tutorials on YouTube", "Documentation for specific languages", "Open-source projects on GitHub"},
		"data science": {"DataCamp courses", "Kaggle competitions", "Statistical learning textbooks", "Python data science libraries documentation"},
		"design":        {"UI/UX design courses on Skillshare", "Design blogs and articles", "Figma and Adobe XD tutorials", "Design inspiration websites (e.g., Dribbble, Behance)"},
		"default":       {"General online learning platforms (Coursera, edX, Khan Academy)", "YouTube educational channels", "Relevant Wikipedia articles", "Books and articles on the subject"},
	}

	resources, ok := learningResources[strings.ToLower(skill)]
	if !ok {
		resources = learningResources["default"]
		fmt.Println("Skill not specifically recognized, using general learning resources.")
	}

	fmt.Println("\n--- Personalized Learning Path for", skill, "---")
	fmt.Println("Recommended Resources:")
	for i, resource := range resources {
		fmt.Printf("%d. %s\n", i+1, resource)
	}
	fmt.Println("Start with foundational resources and gradually move to more advanced topics.")
	fmt.Println("--------------------------------------------------")
}

// 8. Code Snippet Generator
func (a *Agent) CodeSnippetGenerator(language string, task string) {
	fmt.Printf("Generating code snippet for %s task in %s...\n", task, language)

	// Very basic code snippet simulation - real code generation is much more complex
	snippets := map[string]map[string]string{
		"python": {
			"print hello world": "print(\"Hello, World!\")",
			"read file":         `with open("file.txt", "r") as f:
    content = f.read()
    print(content)`,
			"default":             "# Python code snippet for your task would go here",
		},
		"javascript": {
			"print hello world": "console.log(\"Hello, World!\");",
			"fetch data":        `fetch('https://api.example.com/data')
  .then(response => response.json())
  .then(data => console.log(data));`,
			"default":             "// Javascript code snippet for your task would go here",
		},
		"go": {
			"print hello world": `package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}`,
			"read file":         `package main

import (
	"fmt"
	"os"
	"io/ioutil"
)

func main() {
	content, err := ioutil.ReadFile("file.txt")
	if err != nil {
		fmt.Println("Error reading file:", err)
		return
	}
	fmt.Println(string(content))
}`,
			"default":             "// Go code snippet for your task would go here",
		},
		"default": {
			"default": "// Code snippet for your task would go here (language not specified or unsupported)",
		},
	}

	langSnippets, okLang := snippets[strings.ToLower(language)]
	snippet := ""
	if okLang {
		snippet, okTask := langSnippets[strings.ToLower(task)]
		if !okTask {
			snippet = langSnippets["default"]
			fmt.Println("Task not specifically recognized for", language, ", using default snippet.")
		}
	} else {
		snippet = snippets["default"]["default"]
		fmt.Println("Language not supported or recognized, using default snippet.")
	}

	fmt.Println("\n--- Code Snippet in", language, "---")
	fmt.Println(snippet)
	fmt.Println("--------------------------")
}


// 9. Style Guide Generator (Personalized)
func (a *Agent) StyleGuideGenerator(theme string) {
	fmt.Printf("Generating style guide based on theme: '%s'...\n", theme)

	styleThemes := map[string]map[string][]string{
		"professional": {
			"tone":      {"Formal", "Clear", "Concise", "Objective"},
			"formatting": {"Use headings and subheadings", "Bullet points for lists", "Consistent font and spacing"},
			"language":  {"Avoid jargon and slang", "Use precise vocabulary", "Maintain a respectful tone"},
		},
		"creative": {
			"tone":      {"Informal", "Engaging", "Vivid", "Expressive"},
			"formatting": {"Visually appealing layout", "Use of imagery and metaphors", "Varied sentence structure"},
			"language":  {"Embrace creativity and originality", "Use descriptive language", "Connect with the audience emotionally"},
		},
		"minimalist": {
			"tone":      {"Simple", "Direct", "Uncluttered", "Essential"},
			"formatting": {"Clean and spacious design", "Limited color palette", "Focus on key information"},
			"language":  {"Use short and impactful sentences", "Eliminate unnecessary words", "Prioritize clarity and efficiency"},
		},
		"default": {
			"tone":      {"Balanced", "Readable", "Organized", "Accessible"},
			"formatting": {"Clear structure", "Logical flow", "Easy to navigate"},
			"language":  {"Appropriate for target audience", "Correct grammar and spelling", "Consistent style"},
		},
	}

	themeStyles, ok := styleThemes[strings.ToLower(theme)]
	if !ok {
		themeStyles = styleThemes["default"]
		fmt.Println("Theme not recognized, using default style guide.")
	}

	fmt.Println("\n--- Style Guide for Theme:", theme, "---")
	fmt.Println("Tone:", strings.Join(themeStyles["tone"], ", "))
	fmt.Println("Formatting:", strings.Join(themeStyles["formatting"], ", "))
	fmt.Println("Language:", strings.Join(themeStyles["language"], ", "))
	fmt.Println("------------------------------------")
}

// 10. Mindfulness Prompt Generator
func (a *Agent) MindfulnessPromptGenerator() {
	prompts := []string{
		"Take a deep breath and notice the sensation of air entering and leaving your body.",
		"Observe your thoughts without judgment, like clouds passing by.",
		"Focus on the sounds around you, without labeling them as good or bad.",
		"Pay attention to the physical sensations in your body right now.",
		"Bring awareness to your emotions and acknowledge them without getting carried away.",
		"Practice gratitude for three things in your life today.",
		"Engage your senses fully in your current activity – what do you see, hear, smell, taste, touch?",
		"Imagine sending kindness and compassion to yourself and others.",
		"Notice any tension in your body and consciously try to release it.",
		"Reflect on a moment of joy or peace you experienced recently.",
	}

	rand.Seed(time.Now().UnixNano())
	prompt := prompts[rand.Intn(len(prompts))]

	fmt.Println("\n--- Mindfulness Prompt ---")
	fmt.Println(prompt)
	fmt.Println("--------------------------")
}

// 11. Habit Tracker and Analyzer
func (a *Agent) HabitTracker(habit string, status string) {
	today := time.Now().Format("2006-01-02") // YYYY-MM-DD format

	if a.habits[habit] == nil {
		a.habits[habit] = make(map[string][]string)
	}
	if a.habits[habit][today] == nil {
		a.habits[habit][today] = []string{}
	}

	a.habits[habit][today] = append(a.habits[habit][today], status) // Allow multiple statuses per day if needed

	fmt.Printf("Habit '%s' tracked for %s with status: %s\n", habit, today, status)
}

func (a *Agent) HabitAnalyzer() {
	if len(a.habits) == 0 {
		fmt.Println("No habits tracked yet.")
		return
	}

	fmt.Println("\n--- Habit Analysis ---")
	for habit, dailyData := range a.habits {
		fmt.Println("Habit:", habit)
		totalDays := len(dailyData)
		successfulDays := 0
		for _, statuses := range dailyData {
			for _, status := range statuses {
				if strings.ToLower(status) == "completed" || strings.ToLower(status) == "yes" || strings.ToLower(status) == "done" {
					successfulDays++
					break // Count only one successful status per day
				}
			}
		}
		successRate := float64(successfulDays) / float64(totalDays) * 100
		fmt.Printf("  Tracked for %d days, Successful days: %d, Success Rate: %.2f%%\n", totalDays, successfulDays, successRate)
	}
	fmt.Println("----------------------")
}


// 12. Personalized Travel Itinerary Planner
func (a *Agent) PersonalizedTravelItineraryPlanner(destination string, duration string, interests []string) {
	fmt.Println("Planning travel itinerary to:", destination, ", Duration:", duration, ", Interests:", strings.Join(interests, ", "))

	// Simulated travel data and itinerary generation - real system would use travel APIs, maps, etc.
	attractions := map[string][]string{
		"paris":     {"Eiffel Tower", "Louvre Museum", "Notre Dame Cathedral", "Seine River Cruise"},
		"tokyo":     {"Tokyo Skytree", "Senso-ji Temple", "Shibuya Crossing", "Tokyo National Museum"},
		"new york":  {"Statue of Liberty", "Central Park", "Times Square", "Metropolitan Museum of Art"},
		"default":   {"Local attractions", "Parks and nature spots", "Museums and historical sites", "Restaurants and cafes"},
	}

	cityAttractions, ok := attractions[strings.ToLower(destination)]
	if !ok {
		cityAttractions = attractions["default"]
		fmt.Println("Destination not specifically recognized, using general attraction suggestions.")
	}

	fmt.Println("\n--- Personalized Travel Itinerary for", destination, "---")
	fmt.Println("Duration:", duration)
	fmt.Println("Interests:", strings.Join(interests, ", "))
	fmt.Println("\nSuggested Itinerary:")

	days, err := strconv.Atoi(strings.TrimSuffix(duration, " days")) // Simple duration parsing
	if err != nil {
		days = 3 // Default duration if parsing fails
		fmt.Println("Invalid duration format. Assuming 3 days.")
	}

	for day := 1; day <= days; day++ {
		attractionIndex := (day - 1) % len(cityAttractions) // Cycle through attractions for each day
		fmt.Printf("Day %d: Visit %s\n", day, cityAttractions[attractionIndex])
	}
	fmt.Println("... (Detailed itinerary with timings, restaurants, etc. would be generated in a real system)")
	fmt.Println("--------------------------------------------------------")
}

// 13. Creative Brainstorming Partner
func (a *Agent) CreativeBrainstormingPartner(topic string) {
	fmt.Println("Brainstorming ideas for topic:", topic)

	brainstormingStarters := []string{
		"What if we approached this from a completely different angle?",
		"Let's think outside the box – what are some unconventional solutions?",
		"What are the limitations and how can we overcome them creatively?",
		"If we had unlimited resources, what would we do?",
		"Imagine this in a futuristic/past/fantasy setting – how would it change?",
		"What are the core principles involved and how else can we apply them?",
		"Let's consider the user experience – how can we make it more delightful/efficient?",
		"What are some unexpected combinations or analogies we can draw?",
		"What are the potential risks and how can we turn them into opportunities?",
		"Let's challenge the assumptions – are they really necessary?",
	}

	rand.Seed(time.Now().UnixNano())
	starter := brainstormingStarters[rand.Intn(len(brainstormingStarters))]

	fmt.Println("\n--- Creative Brainstorming Session for", topic, "---")
	fmt.Println("Starter thought:", starter)
	fmt.Println("\n...(AI would generate a series of related ideas, suggestions, and questions to stimulate creative thinking)")
	fmt.Println("\nExample ideas (AI generated - placeholders):")
	fmt.Println("- Idea 1: Innovative approach related to the topic")
	fmt.Println("- Idea 2: Unconventional solution exploring a different perspective")
	fmt.Println("- Idea 3: Concept focusing on a specific user need or problem")
	fmt.Println("---------------------------------------------------------")
}

// 14. Personalized Fitness Plan Generator
func (a *Agent) PersonalizedFitnessPlanGenerator(goal string, level string, equipment []string) {
	fmt.Println("Generating fitness plan for goal:", goal, ", Level:", level, ", Equipment:", strings.Join(equipment, ", "))

	// Simulated fitness plan generation - real system would use fitness databases, exercise science principles, etc.
	workoutTypes := map[string][]string{
		"weight loss":    {"Cardio", "Strength Training", "HIIT"},
		"muscle gain":    {"Strength Training", "Compound Exercises", "Progressive Overload"},
		"endurance":      {"Long Distance Cardio", "Interval Training", "Bodyweight Exercises"},
		"general fitness": {"Balanced Cardio and Strength", "Flexibility and Mobility"},
		"default":        {"Mix of Cardio and Strength", "Warm-up and Cool-down"},
	}

	levelAdjustments := map[string]map[string][]string{
		"beginner": {
			"exercises": {"Bodyweight exercises", "Walking/Jogging", "Basic strength training moves"},
			"duration":  {"Shorter workouts", "Focus on form over intensity"},
		},
		"intermediate": {
			"exercises": {"More challenging bodyweight exercises", "Weight training with moderate weights", "Varied cardio workouts"},
			"duration":  {"Moderate workout duration", "Increase intensity gradually"},
		},
		"advanced": {
			"exercises": {"Advanced weight training techniques", "High-intensity cardio", "Complex movements"},
			"duration":  {"Longer and more intense workouts", "Focus on pushing limits"},
		},
		"default": {
			"exercises": {"Variety of exercises to suit different levels"},
			"duration":  {"Moderate workout duration"},
		},
	}

	goalWorkouts, okGoal := workoutTypes[strings.ToLower(goal)]
	if !okGoal {
		goalWorkouts = workoutTypes["default"]
		fmt.Println("Goal not specifically recognized, using general fitness plan template.")
	}

	levelSettings, okLevel := levelAdjustments[strings.ToLower(level)]
	if !okLevel {
		levelSettings = levelAdjustments["default"]
		fmt.Println("Fitness level not recognized, using default level settings.")
	}

	fmt.Println("\n--- Personalized Fitness Plan for", goal, "---")
	fmt.Println("Level:", level)
	fmt.Println("Equipment:", strings.Join(equipment, ", "))
	fmt.Println("\nWorkout Types:", strings.Join(goalWorkouts, ", "))
	fmt.Println("Exercise Recommendations:", strings.Join(levelSettings["exercises"], ", "))
	fmt.Println("Duration Guidelines:", strings.Join(levelSettings["duration"], ", "))
	fmt.Println("\n...(Detailed weekly plan with specific exercises, sets, reps, etc. would be generated in a real system)")
	fmt.Println("-------------------------------------------------------------------")
}

// 15. Language Translator with Contextual Understanding
func (a *Agent) LanguageTranslator(text string, targetLanguage string) {
	fmt.Printf("Translating text to %s: '%s'\n", targetLanguage, text)

	// Very basic translation simulation - real translation uses complex NLP models and APIs
	simulatedTranslations := map[string]map[string]string{
		"english": {
			"hello world": "Hello, World!",
			"thank you":   "Thank you",
			"good morning": "Good morning",
		},
		"spanish": {
			"hello world": "¡Hola Mundo!",
			"thank you":   "Gracias",
			"good morning": "Buenos días",
		},
		"french": {
			"hello world": "Bonjour le monde!",
			"thank you":   "Merci",
			"good morning": "Bonjour",
		},
		"default": {
			"default": "(Translation in target language would go here)",
		},
	}

	langTranslations, okLang := simulatedTranslations[strings.ToLower(targetLanguage)]
	translation := ""
	if okLang {
		translation, okText := langTranslations[strings.ToLower(text)]
		if !okText {
			translation = langTranslations["default"]["default"]
			fmt.Println("Specific phrase not in simulated dictionary, using placeholder.")
		}
	} else {
		translation = simulatedTranslations["default"]["default"]
		fmt.Println("Target language not supported or recognized, using placeholder.")
	}


	fmt.Println("\n--- Language Translation ---")
	fmt.Println("Original Text:", text)
	fmt.Printf("Translation (%s): %s\n", targetLanguage, translation)
	fmt.Println("---------------------------")
	fmt.Println("(Note: This is a simplified simulation. Real translation would involve contextual analysis and more sophisticated NLP techniques.)")

}


// 16. Meeting Minute Summarizer (Simulated Audio Input)
func (a *Agent) MeetingMinuteSummarizer() {
	fmt.Println("Simulating processing meeting audio and generating minutes...")

	// Simulated meeting transcript
	transcript := `
Meeting started at 10:00 AM. Present: Alice, Bob, Carol.
Agenda: Project Alpha update, Budget discussion, Next steps.
Alice presented the progress on Project Alpha, highlighting key milestones achieved and upcoming deadlines.
Bob raised concerns about the current budget allocation and suggested exploring cost-saving measures.
Carol proposed focusing on marketing efforts for the next quarter.
Decision: To re-evaluate the budget and prioritize marketing initiatives. Action item: Bob to prepare a revised budget proposal by Friday.
Meeting adjourned at 10:45 AM.
`

	fmt.Println("\n--- Simulated Meeting Transcript ---")
	fmt.Println(transcript)

	// Basic keyword-based summarization simulation - real summarization is much more advanced
	summaryPoints := []string{
		"Project Alpha update presented by Alice.",
		"Budget concerns raised by Bob.",
		"Marketing focus proposed by Carol.",
		"Decision: Re-evaluate budget and prioritize marketing.",
		"Action item: Bob to revise budget proposal by Friday.",
	}

	fmt.Println("\n--- Meeting Minutes Summary ---")
	for _, point := range summaryPoints {
		fmt.Println("- ", point)
	}
	fmt.Println("-----------------------------")
}

// 17. Personalized Music Playlist Generator (Genre/Mood Based)
func (a *Agent) PersonalizedMusicPlaylistGenerator(genreMood string) {
	fmt.Printf("Generating music playlist for genre/mood: '%s'...\n", genreMood)

	// Simulated music data - real system would use music APIs (Spotify, Apple Music, etc.) and recommendation algorithms
	musicLibrary := map[string][]string{
		"pop":       {"Song A (Pop)", "Song B (Pop)", "Song C (Pop)", "Song D (Pop)"},
		"rock":      {"Song E (Rock)", "Song F (Rock)", "Song G (Rock)", "Song H (Rock)"},
		"jazz":      {"Song I (Jazz)", "Song J (Jazz)", "Song K (Jazz)", "Song L (Jazz)"},
		"relaxing":  {"Song M (Ambient)", "Song N (Classical)", "Song O (Instrumental)", "Song P (Chillout)"},
		"energetic": {"Song Q (Electronic)", "Song R (Pop-Punk)", "Song S (Upbeat)", "Song T (Dance)"},
		"default":   {"Song U (Various)", "Song V (Various)", "Song W (Various)", "Song X (Various)"},
	}

	playlist, ok := musicLibrary[strings.ToLower(genreMood)]
	if !ok {
		playlist = musicLibrary["default"]
		fmt.Println("Genre/Mood not specifically recognized, using a mix of genres.")
	}

	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(playlist), func(i, j int) { playlist[i], playlist[j] = playlist[j], playlist[i] }) // Shuffle playlist

	fmt.Println("\n--- Personalized Music Playlist for", genreMood, "---")
	fmt.Println("Generated Playlist:")
	for i := 0; i < 5 && i < len(playlist); i++ { // Display up to 5 songs from playlist
		fmt.Printf("%d. %s\n", i+1, playlist[i])
	}
	fmt.Println("...(Full playlist would be longer and potentially streamable in a real system)")
	fmt.Println("---------------------------------------------------")
}

// 18. Art Style Transfer (Simulated)
func (a *Agent) ArtStyleTransfer(imagePath string, style string) {
	fmt.Printf("Simulating art style transfer: Applying '%s' style to image '%s'...\n", style, imagePath)

	// Simulated art style transfer - real system would use image processing libraries and AI models
	artStyles := map[string]string{
		"van gogh":     "Van Gogh style (Starry Night)",
		"impressionism": "Impressionist style (Monet)",
		"cubism":        "Cubist style (Picasso)",
		"abstract":      "Abstract style (Kandinsky)",
		"default":       "Artistic style transformation",
	}

	selectedStyle, ok := artStyles[strings.ToLower(style)]
	if !ok {
		selectedStyle = artStyles["default"]
		fmt.Println("Art style not specifically recognized, using a general artistic transformation.")
	}

	fmt.Println("\n--- Simulated Art Style Transfer ---")
	fmt.Println("Input Image:", imagePath)
	fmt.Println("Applied Style:", selectedStyle)
	fmt.Println("\n...(Image processing and style transfer would be performed here in a real system)")
	fmt.Println("\nOutput (Simulated): Image with", selectedStyle, "applied to it.")
	fmt.Println("-------------------------------------")
}

// 19. Task Prioritization and Management
func (a *Agent) TaskAddTask(task string) {
	a.taskList = append(a.taskList, task)
	fmt.Printf("Task '%s' added to task list.\n", task)
}

func (a *Agent) TaskListTasks() {
	if len(a.taskList) == 0 {
		fmt.Println("Task list is empty.")
		return
	}
	fmt.Println("\n--- Task List ---")
	for i, task := range a.taskList {
		fmt.Printf("%d. %s\n", i+1, task)
	}
	fmt.Println("---------------")
}

func (a *Agent) TaskPrioritizeTasks() {
	if len(a.taskList) <= 1 {
		fmt.Println("Not enough tasks to prioritize.")
		return
	}

	// Very basic priority simulation - real prioritization would use more complex criteria (urgency, importance, deadlines etc.)
	fmt.Println("\n--- Task Prioritization (Simulated) ---")
	fmt.Println("Current Task List:")
	for i, task := range a.taskList {
		fmt.Printf("%d. %s\n", i+1, task)
	}

	// Simple priority heuristic: prioritize tasks based on length (shorter tasks first - can be replaced with more intelligent logic)
	sort.Slice(a.taskList, func(i, j int) bool {
		return len(a.taskList[i]) < len(a.taskList[j])
	})

	fmt.Println("\nPrioritized Task List (Simulated):")
	for i, task := range a.taskList {
		fmt.Printf("%d. %s (Priority: %d)\n", i+1, task, i+1) // Lower index = higher priority
	}
	fmt.Println("-------------------------------------")
}

// 20. Personalized Book Recommendation Engine
func (a *Agent) PersonalizedBookRecommendationEngine(genreAuthorTheme string) {
	fmt.Printf("Recommending books based on genre/author/theme: '%s'...\n", genreAuthorTheme)

	// Simulated book data - real system would use book APIs (Goodreads, Google Books, etc.) and recommendation algorithms
	bookRecommendations := map[string][]string{
		"sci-fi":        {"Book 1 (Sci-Fi)", "Book 2 (Sci-Fi)", "Book 3 (Sci-Fi)", "Book 4 (Sci-Fi)"},
		"fantasy":       {"Book 5 (Fantasy)", "Book 6 (Fantasy)", "Book 7 (Fantasy)", "Book 8 (Fantasy)"},
		"mystery":       {"Book 9 (Mystery)", "Book 10 (Mystery)", "Book 11 (Mystery)", "Book 12 (Mystery)"},
		"programming":   {"Book 13 (Programming)", "Book 14 (Programming)", "Book 15 (Programming)", "Book 16 (Programming)"},
		"self-help":     {"Book 17 (Self-Help)", "Book 18 (Self-Help)", "Book 19 (Self-Help)", "Book 20 (Self-Help)"},
		"default":       {"Book 21 (Various Genre)", "Book 22 (Various Genre)", "Book 23 (Various Genre)", "Book 24 (Various Genre)"},
	}

	recommendations, ok := bookRecommendations[strings.ToLower(genreAuthorTheme)]
	if !ok {
		recommendations = bookRecommendations["default"]
		fmt.Println("Genre/Author/Theme not specifically recognized, using a mix of genres.")
	}

	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(recommendations), func(i, j int) { recommendations[i], recommendations[j] = recommendations[j], recommendations[i] }) // Shuffle recommendations

	fmt.Println("\n--- Personalized Book Recommendations for", genreAuthorTheme, "---")
	fmt.Println("Recommended Books:")
	for i := 0; i < 3 && i < len(recommendations); i++ { // Display up to 3 recommendations
		fmt.Printf("%d. %s\n", i+1, recommendations[i])
	}
	fmt.Println("...(Full list of recommendations with descriptions, ratings, etc. would be provided in a real system)")
	fmt.Println("--------------------------------------------------------")
}

// 21. Dynamic Skill Assessment (Simulated)
func (a *Agent) DynamicSkillAssessment(skill string) {
	fmt.Printf("Simulating dynamic skill assessment for: '%s'...\n", skill)

	// Simulated skill assessment - real assessment would involve interactive tests, adaptive questions, performance analysis etc.
	assessmentQuestions := map[string][]string{
		"programming": {
			"Question 1: What is the difference between compiled and interpreted languages?",
			"Question 2: Explain the concept of object-oriented programming.",
			"Question 3: Write a simple function to reverse a string in Python (or your preferred language).",
		},
		"data science": {
			"Question 1: What are the steps involved in a typical data science project?",
			"Question 2: Explain different types of data visualizations and when to use them.",
			"Question 3: Describe a scenario where you would use regression vs. classification.",
		},
		"design": {
			"Question 1: What are the principles of good UI/UX design?",
			"Question 2: Explain the importance of user research in design.",
			"Question 3: Describe a design process for creating a mobile app interface.",
		},
		"default": {
			"Question 1: General knowledge question related to the skill.",
			"Question 2: Application-based question testing practical understanding.",
			"Question 3: Scenario-based question requiring problem-solving.",
		},
	}

	questions, ok := assessmentQuestions[strings.ToLower(skill)]
	if !ok {
		questions = assessmentQuestions["default"]
		fmt.Println("Skill not specifically recognized, using general assessment questions.")
	}

	fmt.Println("\n--- Dynamic Skill Assessment for", skill, "---")
	fmt.Println("Instructions: Please answer the following questions to the best of your ability.")
	for i, question := range questions {
		fmt.Printf("\nQuestion %d: %s\n", i+1, question)
		fmt.Print("Your Answer: ") // In real interactive assessment, would capture user input and evaluate it
		reader := bufio.NewReader(os.Stdin)
		_, _ = reader.ReadString('\n') // Simulate reading user input (for demonstration)
		fmt.Println("(Simulated Answer Received - In a real system, AI would evaluate the answer)")
	}

	fmt.Println("\n--- Assessment Results (Simulated) ---")
	fmt.Println("Based on your responses (simulated), your skill level in", skill, "is assessed as: Intermediate (Simulated)") // Simulated result
	fmt.Println("Areas of strength: (Simulated) Basic concepts understanding")
	fmt.Println("Areas for improvement: (Simulated) Practical application and advanced topics")
	fmt.Println("--------------------------------------------------------")
}


// 22. Interactive Story/Game Generator
func (a *Agent) InteractiveGameGenerator(genre string) {
	fmt.Printf("Generating interactive story/game in genre: '%s'...\n", genre)

	gameScenarios := map[string][]string{
		"adventure": {
			"You awaken in a mysterious forest. Paths diverge to the north and east. Which path do you choose? (north/east)",
			"You encounter a locked chest. Do you try to open it (open) or leave it (leave)?",
			"A friendly traveler offers you a ride. Do you accept (accept) or decline (decline)?",
		},
		"sci-fi": {
			"You are on a spaceship approaching a strange planet. Do you land (land) or orbit (orbit)?",
			"An alarm sounds - life support malfunction! Do you fix it (fix) or evacuate (evacuate)?",
			"You discover an alien artifact. Do you examine it (examine) or leave it untouched (leave)?",
		},
		"mystery": {
			"You find a cryptic clue at the crime scene. Do you investigate it further (investigate) or ignore it (ignore)?",
			"A suspicious character approaches you. Do you talk to them (talk) or avoid them (avoid)?",
			"You have a hunch about the culprit. Do you follow your hunch (follow) or gather more evidence (evidence)?",
		},
		"default": {
			"You are at a crossroads. Which way do you go? (left/right)",
			"You find an item. Do you pick it up (pick) or leave it (leave)?",
			"Someone asks for your help. Do you help (help) or refuse (refuse)?",
		},
	}

	scenarios, ok := gameScenarios[strings.ToLower(genre)]
	if !ok {
		scenarios = gameScenarios["default"]
		fmt.Println("Genre not specifically recognized, using default adventure scenario.")
	}

	fmt.Println("\n--- Interactive Story/Game in", genre, "---")
	fmt.Println("Welcome to the interactive story! Your choices will shape the narrative.")

	reader := bufio.NewReader(os.Stdin)
	scenarioIndex := 0
	for scenarioIndex < len(scenarios) {
		fmt.Println("\nScenario:", scenarios[scenarioIndex])
		fmt.Print("Your Choice: ")
		choice, _ := reader.ReadString('\n')
		choice = strings.TrimSpace(choice)

		fmt.Printf("You chose: %s\n", choice)
		scenarioIndex++ // Simple linear progression - real game would have branching paths based on choices

		// Simulate game logic based on choice (very basic example - would be more complex in a real game)
		if strings.ToLower(choice) == "north" && genre == "adventure" && scenarioIndex == 1 {
			fmt.Println("You travel north and find...") // Continue story based on choice
		} else if strings.ToLower(choice) == "open" && genre == "adventure" && scenarioIndex == 2 {
			fmt.Println("You open the chest and discover...")
		} // ... more complex game logic for different choices and scenarios ...

		if scenarioIndex >= len(scenarios) {
			fmt.Println("\n...To be continued (Game ended for demonstration purposes)...")
			break
		}
	}
	fmt.Println("----------------------------------------")
}


// --- MCP Interface ---

func main() {
	agent := NewAgent("User") // Initialize AI Agent

	fmt.Println("Welcome to the AI Agent MCP!")
	fmt.Println("Type 'help' to see available commands, 'exit' to quit.")

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("\nagent> ")
		commandStr, _ := reader.ReadString('\n')
		commandStr = strings.TrimSpace(commandStr)
		parts := strings.Fields(commandStr)

		if len(parts) == 0 {
			continue // Empty input
		}

		command := parts[0]
		args := parts[1:]

		switch command {
		case "help":
			printHelp()
		case "exit":
			fmt.Println("Exiting MCP. Goodbye!")
			return
		case "news":
			agent.PersonalizedNewsCurator()
		case "story":
			genre := "default"
			if len(args) > 0 {
				genre = strings.Join(args, " ") // Allow genre with spaces
			}
			agent.CreativeStoryGenerator(genre)
		case "recipe":
			agent.PersonalizedRecipeRecommendation(args)
		case "schedule":
			if len(args) < 3 {
				fmt.Println("Usage: schedule <participant1> <participant2> ... <duration> <topic>")
				continue
			}
			duration := args[len(args)-2] // Assumes duration is the second to last argument
			topic := args[len(args)-1]     // Assumes topic is the last argument
			participants := args[:len(args)-2] // Participants are the arguments before duration and topic
			agent.SmartMeetingScheduler(participants, duration, topic)
		case "sentiment":
			text := strings.Join(args, " ")
			if text == "" {
				fmt.Println("Usage: sentiment <text>")
				continue
			}
			agent.SentimentAnalyzer(text)
		case "trend":
			topic := strings.Join(args, " ")
			if topic == "" {
				fmt.Println("Usage: trend <topic>")
				continue
			}
			agent.TrendForecaster(topic)
		case "learn":
			skill := strings.Join(args, " ")
			if skill == "" {
				fmt.Println("Usage: learn <skill>")
				continue
			}
			agent.PersonalizedLearningPathGenerator(skill)
		case "code":
			if len(args) < 2 {
				fmt.Println("Usage: code <language> <task>")
				continue
			}
			language := args[0]
			task := strings.Join(args[1:], " ")
			agent.CodeSnippetGenerator(language, task)
		case "styleguide":
			theme := strings.Join(args, " ")
			if theme == "" {
				fmt.Println("Usage: styleguide <theme>")
				continue
			}
			agent.StyleGuideGenerator(theme)
		case "mindful":
			agent.MindfulnessPromptGenerator()
		case "habit":
			if len(args) < 2 {
				fmt.Println("Usage: habit track <habit> <status> OR habit analyze")
				continue
			}
			subCommand := args[0]
			if subCommand == "track" {
				if len(args) < 3 {
					fmt.Println("Usage: habit track <habit> <status>")
					continue
				}
				habitName := args[1]
				status := strings.Join(args[2:], " ")
				agent.HabitTracker(habitName, status)
			} else if subCommand == "analyze" {
				agent.HabitAnalyzer()
			} else {
				fmt.Println("Invalid habit subcommand. Use 'track' or 'analyze'.")
			}
		case "travel":
			if len(args) < 2 {
				fmt.Println("Usage: travel <destination> <duration> [interests...]")
				continue
			}
			destination := args[0]
			duration := args[1]
			interests := args[2:]
			agent.PersonalizedTravelItineraryPlanner(destination, duration, interests)
		case "brainstorm":
			topic := strings.Join(args, " ")
			if topic == "" {
				fmt.Println("Usage: brainstorm <topic>")
				continue
			}
			agent.CreativeBrainstormingPartner(topic)
		case "fitness":
			if len(args) < 3 {
				fmt.Println("Usage: fitness <goal> <level> [equipment...]")
				continue
			}
			goal := args[0]
			level := args[1]
			equipment := args[2:]
			agent.PersonalizedFitnessPlanGenerator(goal, level, equipment)
		case "translate":
			if len(args) < 2 {
				fmt.Println("Usage: translate <text> <target_language>")
				continue
			}
			text := strings.Join(args[:len(args)-1], " ") // Text can have spaces
			targetLanguage := args[len(args)-1]
			agent.LanguageTranslator(text, targetLanguage)
		case "minutes":
			agent.MeetingMinuteSummarizer()
		case "playlist":
			genreMood := strings.Join(args, " ")
			if genreMood == "" {
				fmt.Println("Usage: playlist <genre/mood>")
				continue
			}
			agent.PersonalizedMusicPlaylistGenerator(genreMood)
		case "artstyle":
			if len(args) < 2 {
				fmt.Println("Usage: artstyle <image_path> <style>")
				continue
			}
			imagePath := args[0]
			style := strings.Join(args[1:], " ")
			agent.ArtStyleTransfer(imagePath, style)
		case "task":
			if len(args) < 1 {
				fmt.Println("Usage: task add <task> | task list | task prioritize")
				continue
			}
			taskCommand := args[0]
			switch taskCommand {
			case "add":
				taskText := strings.Join(args[1:], " ")
				if taskText == "" {
					fmt.Println("Usage: task add <task>")
					continue
				}
				agent.TaskAddTask(taskText)
			case "list":
				agent.TaskListTasks()
			case "prioritize":
				agent.TaskPrioritizeTasks()
			default:
				fmt.Println("Invalid task subcommand. Use 'add', 'list', or 'prioritize'.")
			}
		case "book":
			genreAuthorTheme := strings.Join(args, " ")
			if genreAuthorTheme == "" {
				fmt.Println("Usage: book <genre/author/theme>")
				continue
			}
			agent.PersonalizedBookRecommendationEngine(genreAuthorTheme)
		case "assess":
			skill := strings.Join(args, " ")
			if skill == "" {
				fmt.Println("Usage: assess <skill>")
				continue
			}
			agent.DynamicSkillAssessment(skill)
		case "game":
			genre := strings.Join(args, " ")
			if genre == "" {
				genre = "adventure" // Default game genre
			}
			agent.InteractiveGameGenerator(genre)

		default:
			fmt.Println("Unknown command. Type 'help' for available commands.")
		}
	}
}

func printHelp() {
	fmt.Println("\n--- AI Agent MCP - Available Commands ---")
	fmt.Println("help                  - Show this help message")
	fmt.Println("exit                  - Exit the MCP")
	fmt.Println("news                  - Get personalized news headlines")
	fmt.Println("story <genre>         - Generate a creative short story in the given genre")
	fmt.Println("recipe <ingredients...> - Recommend recipes based on ingredients")
	fmt.Println("schedule <participants...> <duration> <topic> - Schedule a meeting")
	fmt.Println("sentiment <text>      - Analyze sentiment of text")
	fmt.Println("trend <topic>         - Forecast trends for a topic")
	fmt.Println("learn <skill>         - Generate a learning path for a skill")
	fmt.Println("code <language> <task> - Generate code snippet")
	fmt.Println("styleguide <theme>    - Generate a style guide for a theme")
	fmt.Println("mindful               - Get a mindfulness prompt")
	fmt.Println("habit track <habit> <status> - Track a habit")
	fmt.Println("habit analyze         - Analyze habit tracking data")
	fmt.Println("travel <destination> <duration> [interests...] - Plan travel itinerary")
	fmt.Println("brainstorm <topic>    - Brainstorm ideas for a topic")
	fmt.Println("fitness <goal> <level> [equipment...] - Generate a fitness plan")
	fmt.Println("translate <text> <target_language> - Translate text to another language")
	fmt.Println("minutes               - Summarize meeting minutes (simulated audio)")
	fmt.Println("playlist <genre/mood> - Generate a music playlist")
	fmt.Println("artstyle <image_path> <style> - Apply art style to an image (simulated)")
	fmt.Println("task add <task>       - Add a task to the task list")
	fmt.Println("task list             - List all tasks")
	fmt.Println("task prioritize       - Prioritize tasks")
	fmt.Println("book <genre/author/theme> - Recommend books")
	fmt.Println("assess <skill>        - Dynamic skill assessment (simulated)")
	fmt.Println("game [genre]          - Start an interactive story/game (default genre: adventure)")
	fmt.Println("------------------------------------------")
}

// --- Utility Functions ---
import "sort"
```