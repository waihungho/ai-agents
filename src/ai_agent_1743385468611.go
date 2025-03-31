```go
/*
Outline and Function Summary:

AI Agent Name: "Synergy" - Personal Growth & Creative Catalyst

Concept: Synergy is an AI agent designed to be a personal growth and creative catalyst. It leverages advanced AI techniques to understand user needs, provide personalized insights, stimulate creativity, and facilitate self-improvement.  It's designed to be a companion that helps users unlock their potential through diverse functionalities.

MCP (Message Passing Communication) Interface:  The agent communicates via channels, receiving commands and sending responses as string messages.

Functions (20+):

1.  PersonalizedGreeting: Generates a personalized greeting based on user history and current context (time of day, etc.).
2.  DailyInspirationQuote: Provides a curated inspirational quote each day to boost motivation.
3.  GoalSettingAssistant: Helps users define SMART goals and break them down into actionable steps.
4.  HabitTracker: Allows users to track habits, providing reminders and progress visualizations.
5.  MoodCheckIn: Prompts users for daily mood check-ins and provides insights into mood patterns.
6.  GuidedJournalingPrompt: Generates personalized journaling prompts to encourage self-reflection.
7.  CreativeBrainstorming: Facilitates brainstorming sessions for projects or ideas, offering diverse perspectives.
8.  IdeaIncubator: Takes initial ideas, expands upon them, and provides potential development paths.
9.  PersonalizedLearningPath: Recommends learning resources (articles, courses, videos) based on user interests and goals.
10. SkillGapAnalysis: Analyzes user skills and identifies areas for improvement based on their aspirations.
11. PersonalizedBookRecommendation: Recommends books based on user reading history, preferences, and current goals.
12. SummarizeArticle: Summarizes a given article or text into key points and insights.
13. ConceptExplainer: Explains complex concepts in a simplified and understandable manner.
14. LanguageTranslation: Provides real-time translation between different languages.
15. CreativeWritingPrompt: Generates creative writing prompts to stimulate imagination and writing skills.
16. PersonalizedMusicPlaylistGenerator: Creates music playlists based on user mood, activity, or genre preferences.
17. DreamInterpretationHint: Offers potential interpretations of user-described dreams (disclaimer: not professional advice).
18. TimeManagementOptimizer: Analyzes user schedules and provides suggestions for better time management and productivity.
19. StressReliefTechniqueSuggestion: Recommends personalized stress relief techniques based on user stress levels and preferences.
20. AffirmationGenerator: Generates personalized positive affirmations to boost self-esteem and positive thinking.
21. EthicalDilemmaSimulator: Presents ethical dilemmas and facilitates discussions to improve ethical reasoning skills.
22. FutureScenarioPlanning: Helps users explore potential future scenarios based on current trends and decisions.
*/

package main

import (
	"fmt"
	"math/rand"
	"strings"
	"time"
)

// AIAgent struct represents our Synergy AI Agent
type AIAgent struct {
	name          string
	userPreferences map[string]string // Example: Mood, LearningStyle, MusicGenre
	habitData     map[string][]bool   // Habit name -> daily completion (for last 7 days)
	moodHistory   []string          // Daily mood history
}

// NewAIAgent creates a new AIAgent instance
func NewAIAgent(name string) *AIAgent {
	return &AIAgent{
		name:          name,
		userPreferences: make(map[string]string),
		habitData:     make(map[string][]bool),
		moodHistory:   make([]string, 0),
	}
}

// Start initiates the AI Agent's message processing loop
func (agent *AIAgent) Start(commandChan <-chan string, responseChan chan<- string) {
	fmt.Printf("%s Agent '%s' started and listening for commands.\n", agent.name, agent.name)
	for command := range commandChan {
		response := agent.processCommand(command)
		responseChan <- response
	}
	fmt.Println(agent.name, "Agent shutting down.")
}

func (agent *AIAgent) processCommand(command string) string {
	parts := strings.SplitN(command, ":", 2)
	if len(parts) < 2 {
		return "Error: Invalid command format. Use 'command:arguments'"
	}
	commandName := parts[0]
	arguments := ""
	if len(parts) > 1 {
		arguments = parts[1]
	}

	switch commandName {
	case "PersonalizedGreeting":
		return agent.PersonalizedGreeting(arguments)
	case "DailyInspirationQuote":
		return agent.DailyInspirationQuote()
	case "GoalSettingAssistant":
		return agent.GoalSettingAssistant(arguments)
	case "HabitTracker":
		return agent.HabitTracker(arguments)
	case "MoodCheckIn":
		return agent.MoodCheckIn(arguments)
	case "GuidedJournalingPrompt":
		return agent.GuidedJournalingPrompt()
	case "CreativeBrainstorming":
		return agent.CreativeBrainstorming(arguments)
	case "IdeaIncubator":
		return agent.IdeaIncubator(arguments)
	case "PersonalizedLearningPath":
		return agent.PersonalizedLearningPath(arguments)
	case "SkillGapAnalysis":
		return agent.SkillGapAnalysis(arguments)
	case "PersonalizedBookRecommendation":
		return agent.PersonalizedBookRecommendation(arguments)
	case "SummarizeArticle":
		return agent.SummarizeArticle(arguments)
	case "ConceptExplainer":
		return agent.ConceptExplainer(arguments)
	case "LanguageTranslation":
		return agent.LanguageTranslation(arguments)
	case "CreativeWritingPrompt":
		return agent.CreativeWritingPrompt()
	case "PersonalizedMusicPlaylistGenerator":
		return agent.PersonalizedMusicPlaylistGenerator(arguments)
	case "DreamInterpretationHint":
		return agent.DreamInterpretationHint(arguments)
	case "TimeManagementOptimizer":
		return agent.TimeManagementOptimizer(arguments)
	case "StressReliefTechniqueSuggestion":
		return agent.StressReliefTechniqueSuggestion(arguments)
	case "AffirmationGenerator":
		return agent.AffirmationGenerator()
	case "EthicalDilemmaSimulator":
		return agent.EthicalDilemmaSimulator()
	case "FutureScenarioPlanning":
		return agent.FutureScenarioPlanning(arguments)
	case "SetPreference":
		return agent.SetPreference(arguments)
	case "GetPreference":
		return agent.GetPreference(arguments)
	default:
		return "Error: Unknown command: " + commandName
	}
}

// --- Function Implementations ---

// 1. PersonalizedGreeting: Generates a personalized greeting.
func (agent *AIAgent) PersonalizedGreeting(userName string) string {
	currentTime := time.Now()
	hour := currentTime.Hour()
	greeting := "Good day"
	if hour < 12 {
		greeting = "Good morning"
	} else if hour < 18 {
		greeting = "Good afternoon"
	} else {
		greeting = "Good evening"
	}

	namePart := "!"
	if userName != "" {
		namePart = ", " + userName + "!"
	}
	return fmt.Sprintf("%s%s Welcome back to Synergy.", greeting, namePart)
}

// 2. DailyInspirationQuote: Provides a curated inspirational quote.
func (agent *AIAgent) DailyInspirationQuote() string {
	quotes := []string{
		"The only way to do great work is to love what you do. - Steve Jobs",
		"Believe you can and you're halfway there. - Theodore Roosevelt",
		"The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
		"Strive not to be a success, but rather to be of value. - Albert Einstein",
		"The mind is everything. What you think you become. - Buddha",
	}
	rand.Seed(time.Now().UnixNano()) // Seed for random quote selection
	randomIndex := rand.Intn(len(quotes))
	return "Daily Inspiration: \"" + quotes[randomIndex] + "\""
}

// 3. GoalSettingAssistant: Helps users define SMART goals.
func (agent *AIAgent) GoalSettingAssistant(goalDescription string) string {
	if goalDescription == "" {
		return "Please provide a description of your goal for assistance."
	}
	// In a real agent, this would involve NLP to understand the goal,
	// and potentially guide the user through SMART goal framework.
	return fmt.Sprintf("Goal Setting Assistant:\nGoal: %s\nConsider making your goal Specific, Measurable, Achievable, Relevant, and Time-bound (SMART). Let's break it down further if needed.", goalDescription)
}

// 4. HabitTracker: Allows users to track habits.
func (agent *AIAgent) HabitTracker(arguments string) string {
	parts := strings.SplitN(arguments, ",", 2)
	if len(parts) < 2 {
		return "Habit Tracker: Invalid arguments. Use 'HabitTracker:action,habit_name' (action: 'add', 'track', 'view')"
	}
	action := strings.TrimSpace(parts[0])
	habitName := strings.TrimSpace(parts[1])

	switch action {
	case "add":
		if _, exists := agent.habitData[habitName]; exists {
			return fmt.Sprintf("Habit '%s' already exists.", habitName)
		}
		agent.habitData[habitName] = make([]bool, 7) // Initialize tracker for 7 days
		return fmt.Sprintf("Habit '%s' added. Start tracking!", habitName)
	case "track":
		if _, exists := agent.habitData[habitName]; !exists {
			return fmt.Sprintf("Habit '%s' not found. Add it first.", habitName)
		}
		today := time.Now().Weekday() // Sunday = 0, Monday = 1, ...
		dayIndex := (int(today) + 6) % 7 // Adjust to start from Monday (0) to Sunday (6) if needed, currently using day of week directly.
		agent.habitData[habitName][dayIndex] = true // Mark habit as completed today
		return fmt.Sprintf("Tracked '%s' for today. Keep it up!", habitName)
	case "view":
		if _, exists := agent.habitData[habitName]; !exists {
			return fmt.Sprintf("Habit '%s' not found.", habitName)
		}
		tracker := agent.habitData[habitName]
		report := fmt.Sprintf("Habit Tracker for '%s' (Last 7 days):\n", habitName)
		days := []string{"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"} // Assuming week starts Monday for display
		for i := 0; i < 7; i++ {
			status := "[ ]"
			if tracker[i] {
				status = "[X]"
			}
			report += fmt.Sprintf("%s: %s ", days[i], status)
		}
		return report
	default:
		return "Habit Tracker: Invalid action. Use 'add', 'track', or 'view'."
	}
}

// 5. MoodCheckIn: Prompts users for daily mood check-ins and provides insights.
func (agent *AIAgent) MoodCheckIn(mood string) string {
	if mood == "" {
		return "Mood Check-in: How are you feeling today? Please describe your mood (e.g., 'happy', 'stressed', 'calm')."
	}
	agent.moodHistory = append(agent.moodHistory, mood)
	// In a real agent, analyze mood history to provide insights.
	return fmt.Sprintf("Mood recorded as '%s'. Thanks for checking in. I'll analyze your mood patterns over time.", mood)
}

// 6. GuidedJournalingPrompt: Generates personalized journaling prompts.
func (agent *AIAgent) GuidedJournalingPrompt() string {
	prompts := []string{
		"What are you grateful for today?",
		"Describe a moment that made you smile recently.",
		"What is a challenge you are currently facing, and how are you approaching it?",
		"What is something you are proud of accomplishing?",
		"If you could give your younger self one piece of advice, what would it be?",
		"What are your goals for the next week?",
		"Describe a person who inspires you and why.",
	}
	rand.Seed(time.Now().UnixNano())
	randomIndex := rand.Intn(len(prompts))
	return "Journaling Prompt: " + prompts[randomIndex] + "\nTake some time to reflect and write about it."
}

// 7. CreativeBrainstorming: Facilitates brainstorming sessions.
func (agent *AIAgent) CreativeBrainstorming(topic string) string {
	if topic == "" {
		return "Creative Brainstorming: What topic would you like to brainstorm about?"
	}
	ideas := []string{
		fmt.Sprintf("Consider different perspectives on %s.", topic),
		fmt.Sprintf("What are some unconventional uses for %s?", topic),
		fmt.Sprintf("Imagine %s in a completely different context.", topic),
		fmt.Sprintf("What are the opposite ideas related to %s?", topic),
		fmt.Sprintf("How can you combine %s with something seemingly unrelated?", topic),
	}
	rand.Seed(time.Now().UnixNano())
	randomIndex := rand.Intn(len(ideas))
	return fmt.Sprintf("Creative Brainstorming for '%s':\n%s", topic, ideas[randomIndex])
}

// 8. IdeaIncubator: Takes initial ideas, expands upon them.
func (agent *AIAgent) IdeaIncubator(initialIdea string) string {
	if initialIdea == "" {
		return "Idea Incubator: What initial idea would you like to develop?"
	}
	expansionPoints := []string{
		"Explore the core components of this idea.",
		"Consider the potential audience or users.",
		"Think about the resources needed to develop this idea.",
		"Identify potential challenges and solutions.",
		"Visualize the final outcome of this idea.",
	}
	rand.Seed(time.Now().UnixNano())
	randomIndex := rand.Intn(len(expansionPoints))
	return fmt.Sprintf("Idea Incubator for '%s':\n%s", initialIdea, expansionPoints[randomIndex])
}

// 9. PersonalizedLearningPath: Recommends learning resources.
func (agent *AIAgent) PersonalizedLearningPath(topic string) string {
	if topic == "" {
		return "Personalized Learning Path: What topic are you interested in learning about?"
	}
	// In a real agent, this would involve querying a knowledge base or API
	// to find relevant resources based on the topic and user preferences.
	learningStyle := agent.userPreferences["LearningStyle"]
	resourceType := "articles and videos"
	if learningStyle == "visual" {
		resourceType = "videos and infographics"
	} else if learningStyle == "auditory" {
		resourceType = "podcasts and audio courses"
	} else if learningStyle == "kinesthetic" {
		resourceType = "interactive tutorials and projects"
	}

	return fmt.Sprintf("Personalized Learning Path for '%s':\nBased on your preferences, I recommend exploring %s on this topic. Search for introductory %s and online courses on platforms like Coursera, Udemy, or Khan Academy.", topic, resourceType, topic)
}

// 10. SkillGapAnalysis: Analyzes user skills and identifies areas for improvement.
func (agent *AIAgent) SkillGapAnalysis(aspirations string) string {
	if aspirations == "" {
		return "Skill Gap Analysis: What are your career or personal aspirations? (e.g., 'Become a software engineer', 'Learn to play guitar')"
	}
	// In a real agent, this would involve comparing user's current skills
	// with the skills required for their aspirations.
	return fmt.Sprintf("Skill Gap Analysis for '%s':\nTo achieve your aspiration, consider developing skills in [Skill 1], [Skill 2], and [Skill 3]. Focus on resources and projects that help you build these skills. Let's discuss specific skills further.", aspirations)
}

// 11. PersonalizedBookRecommendation: Recommends books based on preferences.
func (agent *AIAgent) PersonalizedBookRecommendation(genreOrTopic string) string {
	if genreOrTopic == "" {
		return "Personalized Book Recommendation: What genre or topic are you interested in reading about?"
	}
	// In a real agent, this would use a book recommendation API or database.
	return fmt.Sprintf("Personalized Book Recommendation for '%s':\nBased on your interest in '%s', I recommend checking out '[Book Title 1]', '[Book Title 2]', and '[Book Title 3]'. These are highly rated books in this area.", genreOrTopic, genreOrTopic)
}

// 12. SummarizeArticle: Summarizes a given article or text.
func (agent *AIAgent) SummarizeArticle(text string) string {
	if text == "" {
		return "Summarize Article: Please provide the text you want me to summarize."
	}
	// In a real agent, this would use NLP summarization techniques.
	summary := "This is a placeholder summary. In a real implementation, I would use advanced NLP to summarize the provided text. Key points include: [Point 1], [Point 2], [Point 3]."
	return "Article Summary:\n" + summary
}

// 13. ConceptExplainer: Explains complex concepts in a simplified manner.
func (agent *AIAgent) ConceptExplainer(concept string) string {
	if concept == "" {
		return "Concept Explainer: What concept would you like me to explain?"
	}
	// In a real agent, this would access a knowledge base or use NLP to simplify explanations.
	explanation := fmt.Sprintf("Concept Explanation for '%s':\n(Simplified explanation of %s would go here). Imagine it like this: [Analogy or simple example]. In essence, %s is about...", concept, concept, concept)
	return explanation
}

// 14. LanguageTranslation: Provides real-time translation.
func (agent *AIAgent) LanguageTranslation(arguments string) string {
	parts := strings.SplitN(arguments, ",", 3)
	if len(parts) < 3 {
		return "Language Translation: Invalid arguments. Use 'LanguageTranslation:text,source_language,target_language' (e.g., 'LanguageTranslation:Hello,en,fr')"
	}
	textToTranslate := strings.TrimSpace(parts[0])
	sourceLang := strings.TrimSpace(parts[1])
	targetLang := strings.TrimSpace(parts[2])

	// In a real agent, this would use a translation API.
	translatedText := fmt.Sprintf("(Placeholder translation) '%s' in %s is (approximately) '%s' in %s.", textToTranslate, sourceLang, "[Translated Text]", targetLang)
	return "Translation:\n" + translatedText
}

// 15. CreativeWritingPrompt: Generates creative writing prompts.
func (agent *AIAgent) CreativeWritingPrompt() string {
	prompts := []string{
		"Write a story about a sentient cloud.",
		"Imagine a world where books are outlawed. Write about a book smuggler.",
		"Describe a conversation between two objects in a room when no one is around.",
		"Write a poem about the sound of silence.",
		"Start a story with the sentence: 'The rain tasted like regret.'",
	}
	rand.Seed(time.Now().UnixNano())
	randomIndex := rand.Intn(len(prompts))
	return "Creative Writing Prompt: " + prompts[randomIndex] + "\nLet your imagination flow!"
}

// 16. PersonalizedMusicPlaylistGenerator: Creates music playlists.
func (agent *AIAgent) PersonalizedMusicPlaylistGenerator(moodOrGenre string) string {
	if moodOrGenre == "" {
		return "Personalized Music Playlist Generator: What mood or genre are you in the mood for?"
	}
	// In a real agent, this would use a music API (Spotify, Apple Music) to create playlists.
	musicGenre := agent.userPreferences["MusicGenre"]
	if moodOrGenre != "" {
		musicGenre = moodOrGenre // Override with specific request
	}

	return fmt.Sprintf("Personalized Music Playlist for '%s' mood/genre:\n(Placeholder playlist) I've generated a playlist with songs in the '%s' genre. Consider checking out: [Song 1], [Song 2], [Song 3]... on your favorite music platform.", musicGenre, musicGenre)
}

// 17. DreamInterpretationHint: Offers potential interpretations of dreams.
func (agent *AIAgent) DreamInterpretationHint(dreamDescription string) string {
	if dreamDescription == "" {
		return "Dream Interpretation Hint: Describe your dream to get potential interpretations (Disclaimer: This is not professional dream analysis)."
	}
	// In a real agent, this would use a symbolic dream dictionary or NLP to find patterns.
	interpretation := "(Disclaimer: Dream interpretations are subjective and for entertainment purposes only. Not professional advice.) Based on your dream description, some symbolic interpretations might include: [Symbol 1 meaning], [Symbol 2 meaning]. Consider reflecting on these themes in your waking life."
	return "Dream Interpretation Hint:\n" + interpretation
}

// 18. TimeManagementOptimizer: Analyzes schedules for better time management.
func (agent *AIAgent) TimeManagementOptimizer(scheduleDetails string) string {
	if scheduleDetails == "" {
		return "Time Management Optimizer: Please provide details about your typical daily or weekly schedule to get optimization suggestions."
	}
	// In a real agent, this would parse schedule data and suggest improvements.
	suggestions := "Time Management Optimization Suggestions:\n(Based on your schedule details) Consider time-blocking for focused tasks. Prioritize tasks based on importance and urgency (Eisenhower Matrix). Schedule breaks to avoid burnout. Review your schedule regularly and adjust as needed."
	return suggestions
}

// 19. StressReliefTechniqueSuggestion: Recommends stress relief techniques.
func (agent *AIAgent) StressReliefTechniqueSuggestion() string {
	stressLevel := agent.userPreferences["StressLevel"] // Example preference
	if stressLevel == "" {
		stressLevel = "moderate" // Default
	}

	techniques := []string{
		"Try deep breathing exercises for 5 minutes.",
		"Engage in a short mindfulness meditation session.",
		"Take a brisk walk or do some light exercise.",
		"Listen to calming music or nature sounds.",
		"Practice progressive muscle relaxation.",
	}
	rand.Seed(time.Now().UnixNano())
	randomIndex := rand.Intn(len(techniques))

	return fmt.Sprintf("Stress Relief Technique Suggestion (Based on stress level: '%s'):\nI recommend you try: %s", stressLevel, techniques[randomIndex])
}

// 20. AffirmationGenerator: Generates personalized positive affirmations.
func (agent *AIAgent) AffirmationGenerator() string {
	affirmations := []string{
		"I am capable and strong.",
		"I am worthy of love and happiness.",
		"I embrace challenges as opportunities for growth.",
		"I am in control of my thoughts and feelings.",
		"I am confident in my abilities.",
		"I am grateful for all the good in my life.",
	}
	rand.Seed(time.Now().UnixNano())
	randomIndex := rand.Intn(len(affirmations))
	return "Affirmation: " + affirmations[randomIndex] + "\nRepeat this affirmation to yourself throughout the day."
}

// 21. EthicalDilemmaSimulator: Presents ethical dilemmas for discussion.
func (agent *AIAgent) EthicalDilemmaSimulator() string {
	dilemmas := []string{
		"You witness a colleague stealing office supplies. Do you report them, even if it might harm your work relationship?",
		"You find a wallet with a significant amount of cash and no identification except a photo of a family. What do you do?",
		"You are asked to complete a task at work that you believe is unethical but not illegal. Do you refuse, risking your job?",
		"You discover a flaw in a product your company is about to release that could pose a minor safety risk to consumers. Do you speak up and potentially delay the launch?",
	}
	rand.Seed(time.Now().UnixNano())
	randomIndex := rand.Intn(len(dilemmas))
	return "Ethical Dilemma:\n" + dilemmas[randomIndex] + "\nLet's discuss the ethical considerations and potential courses of action."
}

// 22. FutureScenarioPlanning: Helps explore future scenarios based on trends.
func (agent *AIAgent) FutureScenarioPlanning(trendOrTopic string) string {
	if trendOrTopic == "" {
		return "Future Scenario Planning: What trend or topic would you like to explore future scenarios for? (e.g., 'AI in healthcare', 'Climate change impact on cities')"
	}
	scenarios := []string{
		fmt.Sprintf("Best-Case Scenario for '%s': Imagine the most positive outcomes and breakthroughs related to %s. What does the world look like in 10 years?", trendOrTopic, trendOrTopic),
		fmt.Sprintf("Worst-Case Scenario for '%s': Consider the most negative potential consequences and challenges related to %s. What are the biggest risks and pitfalls?", trendOrTopic, trendOrTopic),
		fmt.Sprintf("Plausible Scenario for '%s': Think about a realistic and likely future based on current trends and trajectories of %s. What is a balanced and probable outlook?", trendOrTopic, trendOrTopic),
		fmt.Sprintf("Wildcard Scenario for '%s': Consider unexpected and disruptive events that could drastically change the future of %s. What are some 'black swan' possibilities?", trendOrTopic, trendOrTopic),
	}
	rand.Seed(time.Now().UnixNano())
	randomIndex := rand.Intn(len(scenarios))
	return fmt.Sprintf("Future Scenario Planning for '%s':\n%s", trendOrTopic, scenarios[randomIndex])
}

// --- Preference Management Functions ---

// SetPreference: Allows setting user preferences.
func (agent *AIAgent) SetPreference(arguments string) string {
	parts := strings.SplitN(arguments, ",", 2)
	if len(parts) < 2 {
		return "SetPreference: Invalid arguments. Use 'SetPreference:preference_name,preference_value'"
	}
	prefName := strings.TrimSpace(parts[0])
	prefValue := strings.TrimSpace(parts[1])
	agent.userPreferences[prefName] = prefValue
	return fmt.Sprintf("Preference '%s' set to '%s'.", prefName, prefValue)
}

// GetPreference: Retrieves a user preference.
func (agent *AIAgent) GetPreference(prefName string) string {
	if value, exists := agent.userPreferences[prefName]; exists {
		return fmt.Sprintf("Preference '%s' is set to '%s'.", prefName, value)
	}
	return fmt.Sprintf("Preference '%s' not found.", prefName)
}


func main() {
	commandChan := make(chan string)
	responseChan := make(chan string)

	synergyAgent := NewAIAgent("Synergy")
	go synergyAgent.Start(commandChan, responseChan)

	// Example interaction loop
	fmt.Println("Type commands for Synergy Agent (e.g., 'PersonalizedGreeting:Alice', 'Help', 'Exit'):")
	for {
		fmt.Print("> ")
		var commandInput string
		fmt.Scanln(&commandInput)

		if strings.ToLower(commandInput) == "exit" {
			close(commandChan) // Signal agent to shutdown
			break
		}

		commandChan <- commandInput
		response := <-responseChan
		fmt.Println("< ", response)
	}

	fmt.Println("Exiting main program.")
}
```