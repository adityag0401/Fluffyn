# Step 1: Install dependencies (run in Colab)
!pip install faiss-cpu google-generativeai sentence-transformers textblob


import json
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from google.colab import files
import re
import random
import time
from datetime import datetime
from textblob import TextBlob


# Step 2: Upload your JSON knowledge base file
print("🐾 Please upload your JSON knowledge base for Fluffyn:")
uploaded = files.upload()
json_path = list(uploaded.keys())[0]
with open(json_path, "r", encoding="utf-8") as f:
    knowledge = json.load(f)


# Step 3: Process & flatten JSON into text chunks for embedding
chunks = []
chunk_texts = []
for key, value in knowledge.items():
    # Convert the value dict into a simple text block
    text = f"{key}: " + " ".join([f"{k}: {v}" for k, v in value.items()])
    chunks.append((key, text))
    chunk_texts.append(text)


# Step 4: Embed the chunks using a SentenceTransformer model
print("🔄 Loading AI models and processing knowledge base...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(chunk_texts, convert_to_numpy=True).astype('float32')


# Step 5: Create a FAISS index and add embeddings
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


# Step 6: Setup Gemini LLM API
GEMINI_API_KEY = "AIzaSyDyp2Mk44mJWvfEpllRoj-SmofFljf84GI"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')


# Step 7: Enhanced text processing and spell correction
def correct_spelling_and_grammar(text):
    """Correct spelling and basic grammar using TextBlob"""
    try:
        blob = TextBlob(text)
        corrected = str(blob.correct())
        return corrected
    except:
        return text


def normalize_text(text):
    """Enhanced normalize text for better matching"""
    # Correct spelling first
    corrected_text = correct_spelling_and_grammar(text)

    # Handle common variations and typos
    variations = {
        r'\bfluffi\b': 'fluffyn',
        r'\bfluffy\b': 'fluffyn',
        r'\bflufin\b': 'fluffyn',
        r'\bthx\b': 'thank you',
        r'\bu\b': 'you',
        r'\bur\b': 'your',
        r'\br\b': 'are',
        r'\bgr8\b': 'great',
        r'\b2\b': 'to',
        r'\b4\b': 'for',
        # Fix common breed name issues
        r'\border collie\b': 'border collie',
        r'\bgerman shepard\b': 'german shepherd',
        r'\bchiwawa\b': 'chihuahua',
        r'\blabrodor\b': 'labrador',
        r'\bgolden retreiver\b': 'golden retriever',
        # Fix common dog-related terms
        r'\bbitting\b': 'biting',
        r'\bbarking\b': 'barking',
        r'\btraine\b': 'training',
        r'\bfeeding\b': 'feeding'
    }

    for pattern, replacement in variations.items():
        corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)

    return corrected_text.strip()


# Step 8: Enhanced conversation pattern matching with better knowledge base integration
def check_knowledge_base_relevance(user_input):
    """Check if user input might be relevant to the knowledge base"""
    normalized_input = normalize_text(user_input).lower()

    # Check if any chunk text contains similar words or concepts
    relevant_chunks = retrieve_relevant_chunks(normalized_input, top_k=3)

    if relevant_chunks:
        # If we found relevant chunks with reasonable similarity, it's knowledge base related
        query_embedding = embedder.encode([normalized_input], convert_to_numpy=True).astype('float32')
        distances, indices = index.search(query_embedding, 3)

        # If the closest match has a reasonable distance (not too far), consider it relevant
        if len(distances[0]) > 0 and distances[0][0] < 1.5:  # Adjust threshold as needed
            return True

    return False


def extract_context_from_history(chat_history):
    """Extract relevant context from recent chat history"""
    if not chat_history:
        return None

    # Look at the last few exchanges for context
    recent_topics = []
    mentioned_breeds = []

    for entry in chat_history[-5:]:  # Last 5 exchanges
        user_msg = entry.get('user', '').lower()
        assistant_msg = entry.get('assistant', '').lower()

        # Extract breed names mentioned in conversation
        common_breeds = [
            'german shepherd', 'golden retriever', 'labrador', 'pug', 'bulldog',
            'boxer', 'cocker spaniel', 'beagle', 'husky', 'chihuahua', 'poodle',
            'rottweiler', 'dalmatian', 'mastiff', 'terrier', 'spaniel', 'retriever',
            'border collie', 'tibetan mastiff', 'shih tzu', 'pomeranian'
        ]

        for breed in common_breeds:
            if breed in user_msg or breed in assistant_msg:
                mentioned_breeds.append(breed)

        # Extract topics discussed
        if any(word in assistant_msg for word in ['temperament', 'personality', 'behavior']):
            recent_topics.append('temperament')
        if any(word in assistant_msg for word in ['training', 'trainability']):
            recent_topics.append('training')
        if any(word in assistant_msg for word in ['health', 'medical', 'test']):
            recent_topics.append('health')
        if any(word in assistant_msg for word in ['grooming', 'care', 'brushing']):
            recent_topics.append('grooming')
        if any(word in assistant_msg for word in ['price', 'cost', 'range', 'budget']):
            recent_topics.append('price')

    return {
        'breeds_mentioned': list(set(mentioned_breeds)),
        'topics_discussed': list(set(recent_topics)),
        'last_response': chat_history[-1].get('assistant', '') if chat_history else ''
    }


def is_follow_up_question(user_input, chat_history):
    """Enhanced follow-up question detection"""
    if not chat_history:
        return False

    user_input_lower = user_input.lower().strip()

    # Common follow-up question patterns
    follow_up_patterns = [
        r'\b(what about|how about|tell me about|what\'?s)\b.*\b(their|its|the)\b',
        r'\b(their|its|the)\b.*\b(temperament|personality|behavior|training|health|care|grooming|exercise|diet|size|price|cost)\b',
        r'\b(how|what|when|where|why|can|do|are|is)\b.*\b(they|it|them)\b',
        r'\b(more about|details about|information about)\b',
        r'^\b(temperament|personality|behavior|training|health|care|grooming|exercise|diet|feeding|size|weight|price|cost|lifespan|life span)\b',
        r'\bwhat\'?s (their|its|the)\b',
        r'\bhow (big|small|much|often)\b.*\b(they|it)\b',
        r'\bdo they\b',
        r'\bare they\b',
        r'\bcan they\b',
        r'\btell.*about\b',
        r'\b(good|suitable).for\b.\bowner\b',
        r'\bprice.*range\b',
        r'\bcost.*less\b',
        r'\bbudget\b',
        r'\bbiting\b.*\bpotential\b'
    ]

    for pattern in follow_up_patterns:
        if re.search(pattern, user_input_lower):
            return True

    # If the input mentions a breed that was recently discussed
    if chat_history:
        context_info = extract_context_from_history(chat_history)
        if context_info and context_info['breeds_mentioned']:
            for breed in context_info['breeds_mentioned']:
                if breed.lower() in user_input_lower:
                    return True

    return False


def classify_user_input(user_input, chat_history=None):
    """Enhanced classify user input with better pet-related detection"""
    # Normalize and correct the input
    normalized_input = normalize_text(user_input)
    user_input_lower = normalized_input.lower().strip()

    # Check if this is a follow-up question first
    if is_follow_up_question(user_input, chat_history):
        return "pet_related_followup"

    # Greeting patterns (more comprehensive)
    greeting_patterns = [
        r'\b(hi|hello|hey|hola|hii|helo|hai)\b',
        r'\b(good morning|good afternoon|good evening|good night)\b',
        r'\bhow are you\b',
        r'\bwhat\'?s up\b',
        r'\bgreetings?\b',
        r'\byo\b',
        r'\bsup\b',
        r'\bhey there\b'
    ]

    # Thank you patterns (expanded)
    thank_patterns = [
        r'\b(thank you|thanks|thank u|thx|thanx|thnx)\b',
        r'\b(thanks a lot|thank you so much)\b',
        r'\b(much appreciated|appreciate it|appreciated)\b',
        r'\b(ty|tysm)\b',
        r'\bgrateful\b'
    ]

    # Sorry patterns
    sorry_patterns = [
        r'\b(sorry|sory|apologize|apologies|my bad|excuse me)\b',
        r'\bi\'?m sorry\b',
        r'\bforgive me\b'
    ]

    # Fluffyn/Company specific patterns
    company_patterns = [
        r'\b(fluffyn|about fluffyn|what is fluffyn|tell me about fluffyn)\b',
        r'\b(company|business|platform)\b',
        r'\b(mission|about us|who are you)\b'
    ]

    # Enhanced pet-related keywords (more comprehensive)
    pet_keywords = [
        'dog', 'cat', 'pet', 'animal', 'puppy', 'kitten', 'breed', 'training',
        'feeding', 'health', 'veterinarian', 'vet', 'grooming', 'care', 'behavior',
        'exercise', 'toys', 'food', 'medicine', 'vaccination', 'shelter',
        'adoption', 'rescue', 'walk', 'leash', 'collar', 'treats', 'litter', 'cage',
        'bird', 'fish', 'rabbit', 'hamster', 'guinea pig', 'canine', 'feline',
        'paws', 'tail', 'fur', 'whiskers', 'bark', 'meow', 'bite', 'scratch',
        'play', 'sleep', 'eat', 'drink', 'sick', 'healthy', 'weight', 'size',
        'age', 'temperament', 'personality', 'socialization', 'housetraining',
        'potty', 'commands', 'tricks', 'obedience', 'aggressive', 'friendly',
        'energy', 'active', 'lazy', 'calm', 'playful', 'gentle', 'loyal',
        'first time owner', 'beginner', 'price', 'cost', 'budget', 'range',
        'affordable', 'expensive', 'cheap', 'biting', 'potential', 'lifespan',
        'life span', 'years', 'living', 'apartment', 'house', 'family'
    ]

    # Dog-specific keywords
    dog_keywords = [
        'Rottweiler', 'Lhasa Apso', 'Beagle', 'American Bulldog', 'Bull Mastiff',
        'Chow Chow', 'German Shepherd', 'Alaskan Malamute', 'Siberian Husky', 'Corgi',
        'Chihuahua', 'Tibetan Mastiff', 'Pomeranian', 'Golden Retriever', 'French Mastiff',
        'English Mastiff', 'Saint Bernard', 'Cane Corso', 'Shih Tzu', 'Border Collie',
        'Great Dane', 'Cocker Spaniel', 'French Bulldog', 'Labrador Retriever',
        'English Bulldog', 'Shiba Inu', 'Pug', 'Neapolitan Mastiff', 'Maltese', 'Poodle',
        'Doberman Pinscher', 'Brazilian Mastiff', 'Boxer', 'Dachshund',

        # ✅ Added more dog breeds
        'Akita', 'Australian Shepherd', 'Basenji', 'Basset Hound', 'Belgian Malinois',
        'Bernese Mountain Dog', 'Bloodhound', 'Boston Terrier', 'Bull Terrier',
        'Dalmatian', 'Irish Setter', 'Jack Russell Terrier', 'Newfoundland',
        'Papillon', 'Pointer', 'Portuguese Water Dog', 'Samoyed', 'Staffordshire Bull Terrier',
        'Weimaraner', 'Whippet', 'Yorkshire Terrier'
    ]

    # Cat-specific keywords
    cat_keywords = [
        'Exotic Shorthair', 'Bengal', 'Ragdoll', 'Siberian Cat', 'Russian Blue',
        'Scottish Fold', 'Siamese', 'American Shorthair', 'Maine Coon',
        'British Shorthair', 'Persian',

        # ✅ Added more cat breeds
        'Abyssinian', 'Balinese', 'Birman', 'Burmese', 'Chartreux',
        'Cornish Rex', 'Devon Rex', 'Egyptian Mau', 'Himalayan',
        'Norwegian Forest Cat', 'Oriental Shorthair', 'Savannah Cat',
        'Selkirk Rex', 'Singapura', 'Sphynx', 'Tonkinese', 'Turkish Angora',
        'Turkish Van', 'Manx', 'Somali'
    ]


    # Check patterns in order of priority
    for pattern in greeting_patterns:
        if re.search(pattern, user_input_lower):
            return "greeting"

    for pattern in thank_patterns:
        if re.search(pattern, user_input_lower):
            return "thank"

    for pattern in sorry_patterns:
        if re.search(pattern, user_input_lower):
            return "sorry"

    for pattern in company_patterns:
        if re.search(pattern, user_input_lower):
            return "company_info"

    # Check if input contains pet-related keywords (including specific breeds)
    all_pet_keywords = pet_keywords + dog_keywords + cat_keywords
    if any(keyword in user_input_lower for keyword in all_pet_keywords):
        return "pet_related"

    # Check if the question might be relevant to our knowledge base
    if check_knowledge_base_relevance(user_input):
        return "pet_related"

    return "unrelated"


# Step 9: Define retrieval function using FAISS
def retrieve_relevant_chunks(query, top_k=3):
    """Retrieve relevant chunks with improved query preprocessing"""
    normalized_query = normalize_text(query)
    query_embedding = embedder.encode([normalized_query], convert_to_numpy=True).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return [chunk_texts[i] for i in indices[0] if i < len(chunk_texts)]


def retrieve_relevant_chunks_with_context(query, context_info=None, top_k=3):
    """Retrieve relevant chunks with context from chat history"""
    # If we have context about breeds mentioned, include them in the query
    enhanced_query = normalize_text(query)

    if context_info and context_info['breeds_mentioned']:
        # Add the most recently mentioned breed to the query
        recent_breed = context_info['breeds_mentioned'][-1]
        enhanced_query = f"{recent_breed} {enhanced_query}"

    query_embedding = embedder.encode([enhanced_query], convert_to_numpy=True).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return [chunk_texts[i] for i in indices[0] if i < len(chunk_texts)]


# Step 10: Generate follow-up questions
def generate_follow_up_questions(user_input, response_type, context_info=None):
    """Generate relevant follow-up questions based on user input and context"""
    normalized_input = normalize_text(user_input).lower()

    # Default general questions
    general_questions = [
        "What other pet topics would you like to explore?",
        "Do you need help choosing the right breed for your lifestyle?",
        "Would you like to know about pet care essentials?"
    ]

    # If we have breed context, generate breed-specific questions
    if context_info and context_info['breeds_mentioned']:
        breed = context_info['breeds_mentioned'][-1]
        breed_questions = [
            f"Would you like to know about {breed.title()}'s grooming requirements?",
            f"Are you curious about {breed.title()}'s exercise needs?",
            f"Do you want to learn about {breed.title()}'s health concerns?"
        ]
        return breed_questions

    # Topic-specific follow-up questions
    if any(word in normalized_input for word in ['training', 'train']):
        return [
            "Would you like specific training techniques for your breed?",
            "Are you interested in puppy training tips?",
            "Do you need help with behavioral issues?"
        ]

    elif any(word in normalized_input for word in ['health', 'medical', 'test']):
        return [
            "Would you like to know about preventive health care?",
            "Are you curious about vaccination schedules?",
            "Do you need information about common health problems?"
        ]

    elif any(word in normalized_input for word in ['food', 'feeding', 'diet']):
        return [
            "Would you like to know about feeding schedules?",
            "Are you interested in nutrition requirements by age?",
            "Do you need help choosing the right food brand?"
        ]

    elif any(word in normalized_input for word in ['price', 'cost', 'budget', 'affordable']):
        return [
            "Would you like to know about ongoing pet care costs?",
            "Are you interested in budget-friendly pet care tips?",
            "Do you need information about pet insurance options?"
        ]

    elif any(word in normalized_input for word in ['first time', 'beginner', 'owner']):
        return [
            "Would you like a beginner's guide to pet ownership?",
            "Are you interested in essential supplies for new pets?",
            "Do you need tips for preparing your home for a pet?"
        ]

    # Check if specific breeds were mentioned
    dog_breeds = ['german shepherd', 'golden retriever', 'labrador', 'pug', 'bulldog',
                  'boxer', 'beagle', 'husky', 'chihuahua', 'poodle', 'border collie',
                  'tibetan mastiff', 'shih tzu', 'pomeranian']

    for breed in dog_breeds:
        if breed in normalized_input:
            return [
                f"Would you like to know more about {breed.title()}'s temperament?",
                f"Are you curious about {breed.title()}'s exercise requirements?",
                f"Do you want to learn about caring for a {breed.title()}?"
            ]

    return general_questions


# Step 11: Enhanced response generation functions
def generate_greeting_response():
    """Generate personalized greeting responses"""
    current_hour = datetime.now().hour
    time_greeting = "Good morning" if current_hour < 12 else "Good afternoon" if current_hour < 17 else "Good evening"

    greetings = [
        f"{time_greeting}! 🐾 I'm Fluffyn, your AI pet care companion! How can I help you find your perfect furry friend today?",
        f"Hello there! 🐕 Welcome to Fluffyn! I'm here to help you with pets and pet care!",
        f"Hey! 🐱 Fluffyn here! Whether you're looking for pet advice or care tips, I'm here to help!",
        f"{time_greeting}! 🐾 Welcome to Fluffyn - where every pet finds their perfect home! How can I assist you today?"
    ]
    return random.choice(greetings)


def generate_thank_response():
    """Generate varied thank you responses"""
    responses = [
        "You're absolutely welcome! 🐾 At Fluffyn, we're always happy to help connect you with your perfect pet companion!",
        "My pleasure! 🐕 That's what Fluffyn is all about - making pet parenting joyful and simple! Anything else I can help with?",
        "Glad I could help! 🐱 Your future pets are lucky to have such a caring owner! Feel free to ask me anything else!",
        "You're very welcome! 🐾 Remember, Fluffyn is here for all your pet needs - caring for and celebrating your furry friends!"
    ]
    return random.choice(responses)


def generate_sorry_response():
    """Generate understanding sorry responses"""
    responses = [
        "No worries at all! 🐾 At Fluffyn, we're here to make everything easy. How can I help you with your pet needs today?",
        "That's perfectly fine! 🐕 No need to apologize - I'm here to assist with any pet-related questions!",
        "No problem whatsoever! 🐱 Let's focus on finding you the perfect pet care advice at Fluffyn!",
        "All good! 🐾 That's what I'm here for - to make your pet journey smooth and enjoyable!"
    ]
    return random.choice(responses)


def generate_company_info_response():
    """Generate detailed company information responses"""
    responses = [
        """🐾 Welcome to Fluffyn! 🐾

At Fluffyn, we're more than just a pet platform — we're a community of pet lovers committed to connecting furry friends with their forever homes!

Our Mission: To make pet parenting joyful, simple, and heartwarming — one pet, one care tip, and one tail wag at a time.

What We Do:
🐶 Help you find your new best friend - Connect with adorable pets waiting to join your family
💡 Provide expert pet care advice - From feeding to training to health tips
❤ Build a caring community - Where every pet deserves love and every human deserves companionship

We believe every pet deserves a loving home and every human deserves a loyal companion! How can I help you today? 🐾""",

        """🌟 About Fluffyn - Your Pet Paradise! 🌟

We're passionate pet lovers who built Fluffyn to help people find, care for, and celebrate their perfect pet match!

Whether you're a first-time pet parent or part of a seasoned fur family, Fluffyn helps you at every stage:

✨ Discover your perfect companion through our pet community
✨ Get expert pet care advice and tips
✨ Learn about pet health, training, and behavior
✨ Join a community that celebrates the joy of pet parenting

Ready to start your pet journey? Ask me about pet care, training, or health advice! 🐾"""
    ]
    return random.choice(responses)


def generate_pet_related_llm_response(user_query, chat_history=None):
    """Generate comprehensive pet-related responses with chat history context - for questions not in knowledge base"""
    normalized_query = normalize_text(user_query)

    # Include chat history in the prompt for context
    history_context = ""
    if chat_history and len(chat_history) > 1:
        recent_history = chat_history[-4:]  # Last 4 exchanges
        history_context = "\n\nPrevious conversation context:\n"
        for entry in recent_history:
            history_context += f"User: {entry['user']}\nFluffyn: {entry['assistant'][:100]}...\n"

    prompt = f"""You are Fluffyn, the friendly AI assistant for pet care and pet advice platform.

    {history_context}

    Current question: {normalized_query}

    This question is about pets (dogs, cats, or other animals) but doesn't seem to match specific information in my knowledge base. Provide a comprehensive, helpful response that:
    • Gives detailed, accurate pet care advice based on general veterinary knowledge
    • Covers dogs, cats, and other common pets as relevant
    • Uses a warm, caring tone
    • Includes practical tips and considerations
    • References previous conversation if relevant
    • Ends with an offer to help further

    Make the response informative (200-300 words) and include relevant emojis.
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"I'd love to help you with that! 🐾 However, I'm experiencing a technical issue right now. For immediate assistance with pet care, please feel free to ask me other pet-related questions!"


def rag_gemini_answer_with_context(user_query, chat_history=None, top_k=5):
    """Enhanced RAG function with context-aware retrieval and chat history"""
    normalized_query = normalize_text(user_query)

    # Extract context from chat history
    context_info = extract_context_from_history(chat_history)

    # Use context-aware retrieval
    if context_info:
        relevant_chunks = retrieve_relevant_chunks_with_context(normalized_query, context_info, top_k)
    else:
        relevant_chunks = retrieve_relevant_chunks(normalized_query, top_k)

    context = "\n\n".join(relevant_chunks)

    # Build enhanced history context
    history_context = ""
    if chat_history and len(chat_history) > 1:
        recent_history = chat_history[-4:]  # Last 4 exchanges
        history_context = "\n\nPrevious conversation context:\n"
        for entry in recent_history:
            history_context += f"User: {entry['user']}\nFluffyn: {entry['assistant'][:150]}...\n"

    # Add context information to the prompt
    context_prompt = ""
    if context_info:
        if context_info['breeds_mentioned']:
            context_prompt += f"\nBreeds previously discussed: {', '.join(context_info['breeds_mentioned'])}"
        if context_info['topics_discussed']:
            context_prompt += f"\nTopics previously covered: {', '.join(context_info['topics_discussed'])}"

    prompt = f"""You are Fluffyn, the expert AI assistant for pet care and pet advice.

    IMPORTANT: Use the knowledge base information provided below to answer the user's question. The knowledge base contains specific information about pets, breeds, care instructions, and other pet-related topics.When user asks about pet costs, always prefer giving price in Indian Rupees (₹).


    CONTEXT AWARENESS: Pay attention to the conversation history and context. If the user is asking a follow-up question about a breed or topic we discussed earlier, make sure to reference that information and continue the conversation naturally.

    Format your response with:
    • Clear, detailed explanations based on the knowledge base
    • Practical step-by-step advice from the knowledge base
    • Additional helpful context and tips
    • Warm, caring tone with emojis
    • Natural continuation of previous conversation if relevant
    • Answer the specific question asked without getting confused

    Aim for 200-300 words and be precise in your response.

    [Fluffyn Knowledge Base]
    {context}

    {context_prompt}

    {history_context}

    [User's Current Question]
    {normalized_query}

    [Your Expert Response - Based on Knowledge Base with Context]
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"I'd love to help you with that pet question! 🐾 While I'm experiencing a technical issue accessing my full knowledge base, I'm still here to assist. Please try rephrasing your question, and I'll do my best to help!"


def rag_gemini_answer(user_query, chat_history=None, top_k=5):
    """Enhanced RAG function with Fluffyn branding and chat history - prioritizes knowledge base"""
    return rag_gemini_answer_with_context(user_query, chat_history, top_k)


# Step 12: Enhanced conversation handler with chat history
class UserSession:
    def _init_(self):
        self.start_time = datetime.now()
        self.interaction_count = 0
        self.topics_discussed = set()
        self.chat_history = []

    def log_interaction(self, topic, user_input, assistant_response):
        self.interaction_count += 1
        self.topics_discussed.add(topic)
        self.chat_history.append({
            'user': user_input,
            'assistant': assistant_response,
            'topic': topic,
            'timestamp': datetime.now()
        })

        # Keep only last 20 exchanges to manage memory
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]


def handle_conversation(user_input, session):
    """Enhanced conversation handler with session tracking, chat history, and follow-up questions"""
    # Correct and normalize input
    normalized_input = normalize_text(user_input)
    input_type = classify_user_input(normalized_input, session.chat_history)

    # Extract context for follow-up questions
    context_info = extract_context_from_history(session.chat_history)

    response = ""

    if input_type == "greeting":
        response = generate_greeting_response()

    elif input_type == "thank":
        response = generate_thank_response()

    elif input_type == "sorry":
        response = generate_sorry_response()

    elif input_type == "company_info":
        response = generate_company_info_response()

    elif input_type == "pet_related" or input_type == "pet_related_followup":
        # Always try to use RAG with context for pet-related questions
        relevant_chunks = retrieve_relevant_chunks(normalized_input, top_k=5)

        # Use context-aware RAG if we have any relevant chunks
        if relevant_chunks:
            response = rag_gemini_answer_with_context(normalized_input, session.chat_history, top_k=5)
        else:
            # Only use general LLM if no relevant chunks found
            response = generate_pet_related_llm_response(normalized_input, session.chat_history)

    else:  # unrelated
        unrelated_responses = [
            "I'm Fluffyn, your pet care assistant! 🐾 I specialize in helping with pets and pet care. You can ask me about pet care tips, training, health advice, or anything related to your furry friends! What would you like to know?",
            "Hi! I'm Fluffyn! 🐕🐱 I'd love to help you with pet-related questions, care advice, or training tips. What pet topic interests you?",
            "That's outside my pet expertise, but I'm here to help with all things furry and feathered! 🐾 Ask me about pet care, training, health, or behavior!",
            "I'm focused on making your pet journey amazing! 🐾 Whether you want pet care advice, training tips, or health guidance, I'm here to help. What can I assist you with today?"
        ]
        response = random.choice(unrelated_responses)

    # Generate follow-up questions
    follow_up_questions = generate_follow_up_questions(normalized_input, input_type, context_info)

    # Add follow-up questions to response
    if follow_up_questions:
        response += f"\n\n🤔 You might also be interested in:\n"
        for i, question in enumerate(follow_up_questions[:3], 1):  # Limit to 3 questions
            response += f"{i}. {question}\n"

    # Log the interaction with chat history
    session.log_interaction(input_type, user_input, response)

    return response


# Step 13: Enhanced chat interface with welcome message
def display_welcome():
    print("=" * 70)
    print("🐾 WELCOME TO FLUFFYN - WHERE PETS FIND THEIR FOREVER HOMES! 🐾")
    print("=" * 70)
    print("Hi! I'm Fluffyn, your friendly AI pet care assistant! 🐕🐱")
    print()
    print("🌟 What I can help you with:")
    print("   🐶 Expert pet care and training advice")
    print("   💡 Health and nutrition guidance")
    print("   🏠 Pet behavior and training tips")
    print("   💰 Price ranges and budget-friendly recommendations")
    print("   ❓ Learn about Fluffyn's mission")
    print("   💭 Remember our conversation context")
    print()
    print("✨ I understand typos and casual language, so chat naturally!")
    print("📝 Type 'exit', 'quit', or 'bye' to end our conversation.")
    print("🔄 I'll provide follow-up questions to keep our conversation going!")
    print("-" * 70)


def main_chat_loop():
    """Main enhanced chat loop with history tracking and follow-up questions"""
    display_welcome()
    session = UserSession()

    while True:
        try:
            user_input = input("\n🐾 You: ").strip()

            # Handle exit commands
            exit_commands = ['exit', 'quit', 'bye', 'goodbye', 'see you', 'cya']
            if any(cmd in user_input.lower() for cmd in exit_commands):
                farewell_messages = [
                    f"Goodbye! 🐾 Thanks for chatting with Fluffyn! You had {session.interaction_count} great interactions. Come back anytime for pet advice!",
                    f"See you later! 🐕🐱 It was wonderful helping you today! Remember, Fluffyn is always here for all your pet needs. Take care!",
                    f"Bye! 🐾 Your pets are lucky to have someone who cares so much! Visit Fluffyn anytime for more pet care advice!",
                    f"Farewell! 🐾 Thanks for choosing Fluffyn - Where every pet finds their perfect care. Until next time, keep spreading pet love!"
                ]
                print(f"\nFluffyn: {random.choice(farewell_messages)}")
                break

            # Handle empty input
            if not user_input:
                encouragements = [
                    "I'm here and ready to help! 🐾 What would you like to know about pets or pet care?",
                    "Feel free to ask me anything about pet care, training, or health! 🐕🐱",
                    "I'm listening! Ask me about pets, behavior, training, or anything else pet-related! 🐾"
                ]
                print(f"\nFluffyn: {random.choice(encouragements)}")
                continue

            # Generate response with follow-up questions
            answer = handle_conversation(user_input, session)
            print(f"\nFluffyn: {answer}")

            # Provide helpful suggestions periodically
            if session.interaction_count % 7 == 0 and session.interaction_count > 0:
                suggestions = [
                    "\n💡 Pro tip: You can ask me about specific pet breeds, training tips, or health concerns!",
                    "\n🌟 Did you know? I can remember our conversation and provide contextual advice!",
                    "\n✨ Feel free to ask follow-up questions - I remember what we discussed!",
                    "\n🎯 You can also ask about budget-friendly pets or breeds good for first-time owners!"
                ]
                print(random.choice(suggestions))

        except KeyboardInterrupt:
            print(f"\n\nFluffyn: Thanks for visiting Fluffyn! 🐾 Come back anytime!")
            break
        except Exception as e:
            print(f"\nFluffyn: Oops! I encountered a small hiccup: {str(e)}")
            print("No worries though - please try again! I'm here to help with all your pet questions! 🐾")


# Step 14: Start the enhanced chat experience
if _name_ == "_main_":
    print("🔄 Initializing Fluffyn AI Assistant...")
    time.sleep(1)
    print("✅ Ready to help!")
    main_chat_loop()
    print("\n🐾 Thank you for choosing Fluffyn - Where every pet finds their perfect home! 🐾")
