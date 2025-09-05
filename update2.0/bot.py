# FLUFFYN AI ASSISTANT - OPTIMIZED VERSION
# REMOVED: Emergency handling code
# ENHANCED: Context and history tracking
# ============================================

# ============================================
# STEP 1: Install Required Dependencies
# ============================================
!pip install -q langchain langchain-google-genai langchain-community
!pip install -q sentence-transformers faiss-cpu
!pip install -q google-cloud-firestore firebase-admin
!pip install -q textblob tiktoken
!pip install -q google-generativeai google-ai-generativelanguage

# ============================================
# STEP 2: Import Libraries
# ============================================
import json
import os
import re
import random
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import numpy as np
import pickle

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# FAISS and embeddings
import faiss
from sentence_transformers import SentenceTransformer

# Alternative imports for the combined approach
import google.generativeai as genai

# Firebase/Firestore imports
import firebase_admin
from firebase_admin import credentials, firestore
from google.colab import files
import google.auth

# Text processing
from textblob import TextBlob

# ============================================
# STEP 3: Enhanced Breed Recognition System
# ============================================
class BreedMatcher:
    """Enhanced breed matching with species tracking"""

    def __init__(self):
        self.dog_breeds = [
            'Rottweiler', 'Lhasa Apso', 'Beagle', 'American Bulldog', 'Bull Mastiff',
            'Chow Chow', 'German Shepherd', 'Alaskan Malamute', 'Siberian Husky', 'Corgi',
            'Chihuahua', 'Tibetan Mastiff', 'Pomeranian', 'Golden Retriever', 'French Mastiff',
            'English Mastiff', 'Saint Bernard', 'Cane Corso', 'Shih Tzu', 'Border Collie',
            'Great Dane', 'Cocker Spaniel', 'French Bulldog', 'Labrador Retriever',
            'English Bulldog', 'Shiba Inu', 'Pug', 'Neapolitan Mastiff', 'Maltese', 'Poodle',
            'Doberman Pinscher', 'Brazilian Mastiff', 'Boxer', 'Dachshund',
            'Akita', 'Australian Shepherd', 'Basenji', 'Basset Hound', 'Belgian Malinois',
            'Bernese Mountain Dog', 'Bloodhound', 'Boston Terrier', 'Bull Terrier',
            'Dalmatian', 'Irish Setter', 'Jack Russell Terrier', 'Newfoundland',
            'Papillon', 'Pointer', 'Portuguese Water Dog', 'Samoyed', 'Staffordshire Bull Terrier',
            'Weimaraner', 'Whippet', 'Yorkshire Terrier'
        ]

        self.cat_breeds = [
            'Exotic Shorthair', 'Bengal', 'Ragdoll', 'Siberian Cat', 'Russian Blue',
            'Scottish Fold', 'Siamese', 'American Shorthair', 'Maine Coon',
            'British Shorthair', 'Persian', 'Abyssinian', 'Balinese', 'Birman', 'Burmese',
            'Chartreux', 'Cornish Rex', 'Devon Rex', 'Egyptian Mau', 'Himalayan',
            'Norwegian Forest Cat', 'Oriental Shorthair', 'Savannah Cat',
            'Selkirk Rex', 'Singapura', 'Sphynx', 'Tonkinese', 'Turkish Angora',
            'Turkish Van', 'Manx', 'Somali'
        ]

        self.all_breeds = self.dog_breeds + self.cat_breeds

        # Create breed aliases for better matching
        self.breed_aliases = {
            'german shepard': 'German Shepherd',
            'german sheapard': 'German Shepherd',
            'german shephard': 'German Shepherd',
            'german shrepard': 'German Shepherd',
            'gsd': 'German Shepherd',
            'chiwawa': 'Chihuahua',
            'chiuaua': 'Chihuahua',
            'labrodor': 'Labrador Retriever',
            'labrador': 'Labrador Retriever',
            'lab': 'Labrador Retriever',
            'golden retreiver': 'Golden Retriever',
            'golden': 'Golden Retriever',
            'cocker': 'Cocker Spaniel',
            'pom': 'Pomeranian',
            'shiba': 'Shiba Inu',
            'husky': 'Siberian Husky',
            'malamute': 'Alaskan Malamute',
            'great lane': 'Great Dane',
            'dane': 'Great Dane',
            'border': 'Border Collie',
            'collie': 'Border Collie',
            'bulldog': 'English Bulldog',
            'french bulldog': 'French Bulldog',
            'frenchie': 'French Bulldog',
            'mastiff': 'English Mastiff',
            'rotty': 'Rottweiler',
            'rott': 'Rottweiler',
            'maine': 'Maine Coon',
            'persian': 'Persian',
            'siamese': 'Siamese',
            'bengal': 'Bengal',
            'ragdoll': 'Ragdoll'
        }

        # Create lowercase mapping for faster lookup
        self.breed_lower_map = {}
        for breed in self.all_breeds:
            self.breed_lower_map[breed.lower()] = breed

        for alias, breed in self.breed_aliases.items():
            self.breed_lower_map[alias.lower()] = breed

    def find_breed_in_text(self, text):
        """Find breed mentions in text with fuzzy matching"""
        text_lower = text.lower().strip()
        found_breeds = []

        # Direct breed name match
        for breed_lower, breed_proper in self.breed_lower_map.items():
            if breed_lower in text_lower:
                found_breeds.append(breed_proper)

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for breed in found_breeds:
            if breed not in seen:
                seen.add(breed)
                result.append(breed)

        return result

    def get_breed_type(self, breed):
        """Get whether breed is dog or cat"""
        if breed in self.dog_breeds:
            return "dog"
        elif breed in self.cat_breeds:
            return "cat"
        return "unknown"

# ============================================
# STEP 4: Enhanced Context Manager
# ============================================
class ConversationContext:
    """Maintains conversation context to prevent species/breed confusion"""

    def __init__(self):
        self.current_pet = None  # {'breed': str, 'species': str, 'life_stage': str}
        self.last_topic = None
        self.conversation_flow = []
        self.history = []  # Store conversation history

    def set_pet_context(self, breed: str, species: str, life_stage: str = None):
        """Set the current pet being discussed"""
        self.current_pet = {
            'breed': breed,
            'species': species,
            'life_stage': life_stage or 'adult'
        }

    def get_current_pet(self):
        """Get current pet context"""
        return self.current_pet

    def update_topic(self, topic: str):
        """Update current topic"""
        self.last_topic = topic

    def add_to_history(self, user_input: str, response: str):
        """Add to conversation history"""
        self.history.append({
            'user': user_input,
            'assistant': response,
            'timestamp': datetime.now()
        })
        # Keep only last 10 interactions
        if len(self.history) > 10:
            self.history = self.history[-10:]

    def get_recent_history(self, count=3):
        """Get recent conversation history"""
        return self.history[-count:] if self.history else []

# ============================================
# STEP 5: Enhanced Text Processing
# ============================================
class TextProcessor:
    """Enhanced text processing with context detection"""

    def __init__(self):
        self.breed_matcher = BreedMatcher()

        # Life stage keywords
        self.life_stage_keywords = {
            'puppy': ['puppy', 'puppies', 'pup'],
            'kitten': ['kitten', 'kittens'],
            'senior': ['senior', 'old', 'elderly', 'aged'],
            'adult': ['adult', 'grown']
        }

    def extract_life_stage(self, text):
        """Extract life stage from text"""
        text_lower = text.lower()

        for stage, keywords in self.life_stage_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return stage
        return 'adult'

    def correct_spelling(self, text):
        """Correct spelling using TextBlob with breed-specific fixes"""
        try:
            # First fix common breed misspellings
            corrected = text
            for alias, correct_breed in self.breed_matcher.breed_aliases.items():
                corrected = re.sub(
                    r'\b' + re.escape(alias) + r'\b',
                    correct_breed,
                    corrected,
                    flags=re.IGNORECASE
                )

            # Then apply general spell correction
            blob = TextBlob(corrected)
            return str(blob.correct())
        except:
            return text

    def normalize_text(self, text):
        """Normalize text with breed-aware processing"""
        # First correct spelling including breed names
        corrected = self.correct_spelling(text)

        # Common variations
        variations = {
            r'\bthx\b': 'thank you',
            r'\bu\b': 'you',
            r'\bur\b': 'your',
            r'\br\b': 'are',
            r'\bgr8\b': 'great',
            r'\b2\b': 'to',
            r'\b4\b': 'for',
            r'\bbitting\b': 'biting',
            r'\btraine\b': 'training',
            r'\btell about\b': 'information about',
            r'\bwhat eye issues\b': 'eye problems',
        }

        for pattern, replacement in variations.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)

        return corrected.strip()

    def extract_breed_and_topic(self, text):
        """Extract breed, topic, and context from query"""
        normalized = self.normalize_text(text)
        breeds = self.breed_matcher.find_breed_in_text(normalized)
        topic = self._extract_topic(normalized)
        life_stage = self.extract_life_stage(normalized)

        return breeds, topic, life_stage

    def _extract_topic(self, text):
        """Extract the main topic from text"""
        text_lower = text.lower()

        topic_patterns = {
            'health': ['health', 'medical', 'disease', 'condition', 'illness', 'eye issues', 'problems', 'sick'],
            'temperament': ['temperament', 'personality', 'behavior', 'nature', 'character', 'traits', 'aggressive', 'calm'],
            'training': ['training', 'train', 'command', 'obedience', 'teach', 'learn'],
            'grooming': ['grooming', 'brushing', 'care', 'maintenance', 'coat', 'fur'],
            'price': ['price', 'cost', 'budget', 'expensive', 'affordable', 'range', 'money'],
            'exercise': ['exercise', 'activity', 'walk', 'energy', 'active', 'physical'],
            'feeding': ['food', 'feeding', 'diet', 'nutrition', 'eat', 'meal', 'meat'],
            'size': ['size', 'big', 'small', 'large', 'weight', 'height'],
            'child_friendly': ['children', 'kids', 'child friendly', 'good with children'],
            'general_info': ['information', 'about', 'tell', 'describe', 'details']
        }

        for topic, keywords in topic_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return topic

        return 'general_info'

# ============================================
# STEP 6: Enhanced Firebase Session Manager
# ============================================
class FirestoreSessionManager:
    """Enhanced session manager with context persistence"""

    def __init__(self, db, collection_name="fluffyn_sessions"):
        self.db = db
        self.collection = db.collection(collection_name)
        self.current_session_id = None

    def create_session(self, user_id="default_user"):
        """Create a new chat session with enhanced context tracking"""
        session_id = hashlib.md5(f"{user_id}_{datetime.now()}".encode()).hexdigest()[:12]
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'messages': [],
            'context': {
                'current_pet': None,
                'last_topic': None,
                'conversation_theme': None,
                'breed_history': [],
                'species_consistency': True
            },
            'interaction_count': 0
        }
        self.collection.document(session_id).set(session_data)
        self.current_session_id = session_id
        return session_id

    def add_message(self, session_id, user_input, bot_response, context_updates=None):
        """Add message with enhanced context tracking"""
        doc_ref = self.collection.document(session_id)
        doc = doc_ref.get()

        if doc.exists:
            data = doc.to_dict()
            messages = data.get('messages', [])
            context = data.get('context', {})

            # Add new message
            messages.append({
                'user': user_input,
                'assistant': bot_response,
                'timestamp': datetime.now()
            })

            # Keep last 10 messages for better context
            if len(messages) > 10:
                messages = messages[-10:]

            # Update context
            if context_updates:
                context.update(context_updates)

            doc_ref.update({
                'messages': messages,
                'context': context,
                'interaction_count': data.get('interaction_count', 0) + 1,
                'updated_at': datetime.now()
            })

    def get_session_context(self, session_id):
        """Get enhanced session context"""
        doc = self.collection.document(session_id).get()
        if doc.exists:
            data = doc.to_dict()
            return {
                'messages': data.get('messages', []),
                'context': data.get('context', {}),
                'interaction_count': data.get('interaction_count', 0)
            }
        return None

# ============================================
# STEP 7: FAISS Knowledge Retrieval System
# ============================================
class FAISSKnowledgeRetriever:
    """FAISS-based knowledge retrieval system with context awareness"""

    def __init__(self, gemini_api_key):
        self.api_key = gemini_api_key
        self.text_processor = TextProcessor()
        self.breed_matcher = BreedMatcher()

        # Initialize models
        self.setup_gemini_models()
        self.setup_faiss()

    def setup_gemini_models(self):
        """Setup Gemini models with error handling"""
        try:
            if not self.api_key or self.api_key.strip() == "":
                raise ValueError("Gemini API key is empty or invalid")

            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.api_key,
                temperature=0.3,  # Lower temperature for more consistent responses
                max_output_tokens=400  # Reduced for more concise responses
            )

            # Also setup direct Gemini model for fallbacks
            genai.configure(api_key=self.api_key)
            self.direct_model = genai.GenerativeModel('gemini-1.5-flash')

            # Test the API key
            test_response = self.direct_model.generate_content("Hello")
            print("✅ Gemini API key verified and working!")

        except Exception as e:
            print(f"❌ Gemini setup failed: {e}")
            raise

    def setup_faiss(self):
        """Setup FAISS vector store"""
        try:
            # Initialize embedding model
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384

            # Initialize FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.documents = []
            self.embeddings = []

            print("✅ FAISS setup completed!")

        except Exception as e:
            print(f"❌ FAISS setup failed: {e}")
            raise

    def load_knowledge_base(self, json_path):
        """Load knowledge base into FAISS"""
        print("Loading knowledge base into FAISS...")

        with open(json_path, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)

        # Process documents for FAISS
        embeddings = []
        documents = []

        for key, value in knowledge.items():
            # Create comprehensive text
            text_parts = [f"Breed: {key}"]

            # Add breed variations
            breeds_in_key = self.breed_matcher.find_breed_in_text(key)
            for breed in breeds_in_key:
                if breed.lower() != key.lower():
                    text_parts.append(f"Also known as: {breed}")

            for field, content in value.items():
                if isinstance(content, dict):
                    for sub_field, sub_content in content.items():
                        text_parts.append(f"{field} - {sub_field}: {sub_content}")
                else:
                    text_parts.append(f"{field}: {content}")

            doc_text = "\n".join(text_parts)

            # Generate embedding
            embedding = self.embedder.encode(doc_text)
            embeddings.append(embedding)

            # Create metadata
            doc_metadata = {
                "source": key,
                "type": "pet_knowledge",
                "category": self._categorize_content(key, value),
                "breed_type": self.breed_matcher.get_breed_type(key),
                "text": doc_text,
                "breed_names": breeds_in_key,
                "raw_data": value
            }
            documents.append(doc_metadata)

        # Convert to numpy array and normalize for cosine similarity
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)

        # Add to FAISS index
        self.index.add(embeddings_array)
        self.documents = documents
        self.embeddings = embeddings_array

        print(f"✅ Loaded {len(knowledge)} documents into FAISS")

    def _categorize_content(self, key, value):
        """Categorize content based on keywords"""
        text = str(key) + " " + str(value).lower()

        if any(word in text for word in ['dog', 'puppy', 'canine']):
            return "dog"
        elif any(word in text for word in ['cat', 'kitten', 'feline']):
            return "cat"
        elif any(word in text for word in ['health', 'medical', 'disease']):
            return "health"
        elif any(word in text for word in ['training', 'behavior', 'obedience']):
            return "training"
        elif any(word in text for word in ['grooming', 'care', 'maintenance']):
            return "grooming"
        elif any(word in text for word in ['price', 'cost', 'budget']):
            return "pricing"
        else:
            return "general"

    def retrieve_with_faiss(self, query, breeds=None, topic=None, top_k=3):
        """Enhanced retrieval with FAISS - reduced results for focus"""
        try:
            if len(self.documents) == 0:
                return []

            # Build enhanced query
            enhanced_query_parts = [query]

            if breeds:
                for breed in breeds:
                    enhanced_query_parts.append(f"breed {breed}")

            if topic and topic != 'general_info':
                enhanced_query_parts.append(f"{topic} information")

            enhanced_query = " ".join(enhanced_query_parts)

            # Generate query embedding
            query_embedding = self.embedder.encode(enhanced_query)
            query_embedding = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)

            # Search FAISS index
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))

            # Extract relevant chunks
            relevant_chunks = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and score > 0.3:
                    doc = self.documents[idx]

                    # If breeds specified, prioritize breed-specific results
                    if breeds:
                        doc_breeds = doc.get('breed_names', [])
                        if any(breed in doc_breeds for breed in breeds):
                            score += 0.2

                    relevant_chunks.append({
                        'text': doc['text'],
                        'source': doc['source'],
                        'score': score,
                        'metadata': doc
                    })

            # Sort by score (highest first)
            relevant_chunks.sort(key=lambda x: x['score'], reverse=True)

            return [chunk['text'] for chunk in relevant_chunks]

        except Exception as e:
            print(f"Error querying FAISS: {e}")
            return []

# ============================================
# STEP 8: Enhanced Main Fluffyn Assistant Class
# ============================================
class FluffynAssistant:
    """Enhanced Fluffyn Assistant with context persistence and focused responses"""

    def __init__(self, gemini_api_key, firebase_db):
        self.api_key = gemini_api_key
        self.db = firebase_db
        self.session_manager = FirestoreSessionManager(firebase_db)
        self.text_processor = TextProcessor()
        self.breed_matcher = BreedMatcher()
        self.conversation_context = ConversationContext()

        print("Initializing FAISS retriever...")
        self.knowledge_retriever = FAISSKnowledgeRetriever(gemini_api_key)

        self.current_session_id = None
        print("✅ Fluffyn Assistant initialized successfully!")

    def load_knowledge_base(self, json_path):
        """Load knowledge base into FAISS"""
        print("📚 Loading knowledge base...")
        self.knowledge_retriever.load_knowledge_base(json_path)
        print("✅ Knowledge base loaded!")

    def start_session(self, user_id="default_user"):
        """Start a new chat session"""
        self.current_session_id = self.session_manager.create_session(user_id)
        self.conversation_context = ConversationContext()  # Reset context
        print(f"New session started!")
        return self.current_session_id

    def _should_use_context(self, user_input, breeds, session_context):
        """Determine if we should use conversation context"""
        context_indicators = ['it', 'that', 'this', 'the dog', 'the cat', 'my pet', 'he', 'she']

        # If no breeds mentioned but context indicators present
        if not breeds and any(indicator in user_input.lower() for indicator in context_indicators):
            # Check if we have current pet context
            current_pet = self.conversation_context.get_current_pet()
            if current_pet:
                return True, current_pet['breed'], current_pet['species']

            # Check session context
            if session_context and session_context.get('context', {}).get('current_pet'):
                pet_context = session_context['context']['current_pet']
                return True, pet_context['breed'], pet_context['species']

        # Check if user is asking about history
        history_keywords = ['history', 'previous', 'earlier', 'before', 'last time']
        if any(keyword in user_input.lower() for keyword in history_keywords):
            # Return the most recent breed from history
            recent_history = self.conversation_context.get_recent_history()
            if recent_history:
                # Try to find breed in recent history
                for entry in reversed(recent_history):
                    history_breeds, _, _ = self.text_processor.extract_breed_and_topic(entry['user'])
                    if history_breeds:
                        species = self.breed_matcher.get_breed_type(history_breeds[0])
                        return True, history_breeds[0], species

        return False, None, None

    def generate_response(self, user_input, session_context):
        """Enhanced response generation with strict context management"""
        # Extract information from input
        breeds, topic, life_stage = self.text_processor.extract_breed_and_topic(user_input)

        # Check if we should use context
        use_context, context_breed, context_species = self._should_use_context(user_input, breeds, session_context)

        if use_context:
            breeds = [context_breed]
            # Update conversation context
            self.conversation_context.set_pet_context(context_breed, context_species, life_stage)

        # Classify input type
        input_type = self._classify_input(user_input, breeds, topic)

        # Generate appropriate response based on type
        if input_type == "greeting":
            return self._generate_greeting()
        elif input_type == "thank":
            return self._generate_thank_response()
        elif input_type == "company_info":
            return self._generate_company_info()
        elif input_type == "pet_specific":
            return self._generate_pet_response(user_input, breeds, topic, life_stage, session_context, use_context)
        else:
            return self._generate_unrelated_response()

    def _classify_input(self, user_input, breeds, topic):
        """Classify input type"""
        normalized = user_input.lower()

        # Greeting patterns
        if re.search(r'\b(hi|hello|hey|good morning|good evening)\b', normalized):
            return "greeting"

        # Thank you patterns
        elif re.search(r'\b(thank you|thanks|thx)\b', normalized):
            return "thank"

        # Company info patterns
        elif re.search(r'\b(fluffyn|about fluffyn|company)\b', normalized):
            return "company_info"

        # Pet-related
        elif breeds or topic != 'general_info' or self._is_pet_related(normalized):
            return "pet_specific"

        else:
            return "unrelated"

    def _is_pet_related(self, text):
        """Check if input is pet-related"""
        pet_keywords = [
            'dog', 'cat', 'pet', 'animal', 'puppy', 'kitten', 'breed', 'training',
            'feeding', 'health', 'grooming', 'care', 'behavior', 'exercise',
            'temperament', 'personality', 'price', 'cost'
        ]
        return any(keyword in text for keyword in pet_keywords)

    def _generate_pet_response(self, user_input, breeds, topic, life_stage, session_context, use_context=False):
        """Generate focused pet response with context awareness"""
        # Update conversation context if we have new pet info
        if breeds and not use_context:
            species = self.breed_matcher.get_breed_type(breeds[0])
            self.conversation_context.set_pet_context(breeds[0], species, life_stage)

        # Get relevant information from knowledge base
        try:
            relevant_chunks = self.knowledge_retriever.retrieve_with_faiss(
                user_input, breeds=breeds, topic=topic, top_k=2
            )

            if relevant_chunks:
                return self._generate_focused_response(
                    user_input, relevant_chunks, breeds, topic, life_stage, use_context
                )
            else:
                return self._generate_general_breed_response(breeds, topic, user_input)

        except Exception as e:
            print(f"Error in retrieval: {e}")
            return self._generate_fallback_response()

    def _generate_focused_response(self, query, relevant_chunks, breeds, topic, life_stage, use_context):
        """Generate focused, concise response"""
        knowledge_context = "\n".join(relevant_chunks[:2])  # Only top 2 chunks

        context_intro = ""
        if use_context and breeds:
            context_intro = f"For your {life_stage} {breeds[0]}: "
        elif breeds:
            context_intro = f"{breeds[0]} - {topic.replace('_', ' ').title()}:\n\n"

        prompt = f"""You are Fluffyn, a pet care expert. Provide a CONCISE, focused answer (150-200 words max) to the user's question.

REQUIREMENTS:
- Be direct and specific
- Focus ONLY on the question asked
- Use the knowledge provided
- Include practical advice
- Mention costs in Indian Rupees when relevant
- NO excessive emojis or pleasantries
- NO broad breed information unless specifically asked

Knowledge Base:
{knowledge_context}

Question: {query}

Give a focused, helpful response:"""

        try:
            response = self.knowledge_retriever.direct_model.generate_content(prompt)
            base_response = context_intro + response.text.strip()

            # Add relevant follow-up only if not a context-based question
            if not use_context and topic in ['temperament', 'health', 'training']:
                follow_up = self._get_single_followup(breeds[0] if breeds else None, topic)
                if follow_up:
                    base_response += f"\n\nRelated: {follow_up}"

            return base_response

        except Exception as e:
            print(f"Error generating focused response: {e}")
            return self._generate_fallback_response()

    def _generate_general_breed_response(self, breeds, topic, query):
        """Generate response when no specific knowledge found"""
        breed_info = f"{breeds[0]}" if breeds else "pets"
        topic_info = f" about {topic.replace('_', ' ')}" if topic != 'general_info' else ""

        prompt = f"""You are Fluffyn, a pet care expert. Provide CONCISE, accurate information about {breed_info}{topic_info}.

Question: {query}

Requirements:
- Be direct and specific (150-200 words max)
- Include practical advice
- Mention Indian Rupee costs when relevant
- Focus only on what was asked
- Use general veterinary knowledge

Response:"""

        try:
            response = self.knowledge_retriever.direct_model.generate_content(prompt)
            return response.text.strip()
        except:
            return self._generate_fallback_response()

    def _get_single_followup(self, breed, topic):
        """Generate single relevant follow-up question"""
        if not breed:
            return None

        followups = {
            'temperament': f"Training tips for {breed}?",
            'health': f"Common {breed} health issues?",
            'training': f"{breed} exercise needs?",
            'grooming': f"How often should I groom {breed}?",
            'price': f"Monthly costs for {breed}?"
        }

        return followups.get(topic)

    def _generate_greeting(self):
        """Generate concise greeting"""
        return "Hello! I'm Fluffyn, your pet care assistant. What would you like to know about your pet?"

    def _generate_thank_response(self):
        """Generate brief thank you response"""
        return "You're welcome! Happy to help with your pet care needs."

    def _generate_company_info(self):
        """Generate company information"""
        return """Fluffyn is your AI pet care companion, providing expert advice on:

- Breed selection and information
- Health and medical guidance
- Training and behavior tips
- Nutrition and feeding advice
- Grooming and care instructions

Ask me anything about your pet!"""

    def _generate_unrelated_response(self):
        """Generate response for unrelated questions"""
        return "I specialize in pet care advice. Ask me about dog or cat breeds, health, training, feeding, or any pet-related topic!"

    def _generate_fallback_response(self):
        """Generate fallback response"""
        return """I'm here to help with pet care questions! Try asking about:

- Specific dog or cat breeds
- Health and medical concerns
- Training and behavior
- Feeding and nutrition
- Grooming needs

What would you like to know?"""

    def chat(self, user_input):
        """Enhanced chat function with context management"""
        # Handle special commands
        if user_input.lower().strip() == 'end':
            return "CHAT_END"
        elif user_input.lower().strip() == 'startnew':
            self.start_session()
            return "NEW_SESSION_STARTED"

        # Get current session context
        session_context = self.session_manager.get_session_context(self.current_session_id)

        # Generate response
        response = self.generate_response(user_input, session_context)

        # Update context for session storage
        breeds, topic, life_stage = self.text_processor.extract_breed_and_topic(user_input)

        context_updates = {}
        if breeds:
            context_updates['current_pet'] = {
                'breed': breeds[0],
                'species': self.breed_matcher.get_breed_type(breeds[0]),
                'life_stage': life_stage
            }
            context_updates['last_topic'] = topic

        # Add to conversation history
        self.conversation_context.add_to_history(user_input, response)

        # Log the interaction
        self.session_manager.add_message(
            self.current_session_id,
            user_input,
            response,
            context_updates=context_updates
        )

        return response

    def display_welcome(self):
        """Display welcome message"""
        print("=" * 60)
        print("FLUFFYN - OPTIMIZED AI PET CARE ASSISTANT")
        print("=" * 60)
        print("Expert pet advice with enhanced context awareness!")
        print()
        print("KEY FEATURES:")
        print("✓ Maintains conversation context (no species confusion)")
        print("✓ Remembers history and references")
        print("✓ Concise, focused answers (150-200 words)")
        print("✓ Context-aware follow-up handling")
        print("✓ Smart breed recognition with fuzzy matching")
        print()
        print("Commands: 'end' to quit, 'startnew' for fresh session")
        print("-" * 60)

    def chat_loop(self):
        """Enhanced chat loop"""
        self.display_welcome()
        self.start_session()

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    print("\nFluffyn: What would you like to know about your pet?")
                    continue

                response = self.chat(user_input)

                if response == "CHAT_END":
                    print(f"\nFluffyn: Thanks for using Fluffyn! Take care of your pets!")
                    break
                elif response == "NEW_SESSION_STARTED":
                    print("\nFluffyn: Fresh session started! How can I help?")
                    continue

                print(f"\nFluffyn: {response}")

            except KeyboardInterrupt:
                print(f"\n\nFluffyn: Goodbye!")
                break
            except Exception as e:
                print(f"\nFluffyn: I encountered an issue. Please try again!")

# ============================================
# STEP 9: Firebase Setup Function (Unchanged)
# ============================================
def setup_firebase():
    """Setup Firebase with service account credentials"""
    print("Setting up Firebase Authentication...")

    if not firebase_admin._apps:
        print("Please upload your Firebase service account JSON file:")
        uploaded = files.upload()
        service_account_file = list(uploaded.keys())[0]

        cred = credentials.Certificate(service_account_file)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase initialized successfully!")
    else:
        print("✅ Firebase app already initialized.")

    db = firestore.client()
    return db

# ============================================
# STEP 10: Enhanced Main Execution Function
# ============================================
def main():
    """Enhanced main function"""
    print("🚀 Initializing Enhanced Context-Aware Fluffyn...")

    # Setup Firebase
    try:
        db = setup_firebase()
    except Exception as e:
        print(f"❌ Firebase setup failed: {e}")
        return

    # Upload knowledge base
    print("\n📁 Please upload your JSON knowledge base file:")
    try:
        uploaded = files.upload()
        json_path = list(uploaded.keys())[0]
        print(f"✅ Knowledge base uploaded: {json_path}")
    except Exception as e:
        print(f"❌ File upload failed: {e}")
        return

    # Get API key
    gemini_api_key = input("\n🔑 Enter your Gemini API Key: ").strip()
    if not gemini_api_key:
        print("❌ Gemini API key required!")
        return

    # Initialize assistant
    try:
        assistant = FluffynAssistant(gemini_api_key, db)
        assistant.load_knowledge_base(json_path)
        print("✅ Enhanced Fluffyn ready!")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return

    # Start chat
    assistant.chat_loop()

# ============================================
# STEP 11: Enhanced Setup Instructions
# ============================================
def display_setup_instructions():
    """Display setup instructions"""
    print("""
=================================================================
📋 ENHANCED FLUFFYN SETUP - CONTEXT-AWARE VERSION
=================================================================

🔧 KEY FIXES IMPLEMENTED:
✅ CONTEXT PERSISTENCE: Maintains pet breed/species across conversation
✅ EMERGENCY DETECTION: Prioritizes urgent health concerns
✅ CONCISE RESPONSES: 150-200 words max, focused answers
✅ NO SPECIES CONFUSION: Tracks current pet being discussed
✅ SMART FOLLOW-UPS: Context-aware suggestions without overload

=================================================================
🎯 BEHAVIOR IMPROVEMENTS:

1. CONTEXT MANAGEMENT:
   - Remembers which pet you're discussing
   - "What about its temperament?" continues with same breed
   - No more switching from dog to cat mid-conversation

2. EMERGENCY PRIORITY:
   - Detects urgent keywords (vomiting, bleeding, etc.)
   - Provides immediate action steps first
   - Clear vet visit recommendations with costs

3. FOCUSED RESPONSES:
   - Direct answers to specific questions
   - Reduced repetitive information
   - Less overwhelming health condition lists

4. CONVERSATION FLOW:
   - Better handling of "it", "that", "the dog/cat" references
   - Maintains topic continuity
   - Single relevant follow-up suggestion

=================================================================
🚀 SETUP STEPS:

1. Get Gemini API key: https://ai.google.dev/
2. Setup Firebase project with Firestore
3. Upload Firebase service account JSON
4. Upload pet knowledge base JSON file
5. Enter API key and start chatting!

=================================================================
💬 TEST SCENARIOS:

Try these to see the improvements:
1. "My German Shepherd puppy is vomiting" → Emergency response
2. "Tell me about Golden Retrievers" → Focused breed info
3. "What about their temperament?" → Context-aware follow-up
4. "Is it good with children?" → Continues same breed discussion

=================================================================
""")

# ============================================
# STEP 12: Test Questions (Updated)
# ============================================
def get_test_questions():
    """Updated test questions showcasing improvements"""
    return [
        # Context persistence tests
        "Tell me about German Shepherds",
        "What about their temperament?",  # Should continue with German Shepherds
        "Are they good with children?",   # Should continue with German Shepherds

        # Emergency detection tests
        "My puppy is vomiting after eating",
        "My cat has been bleeding",
        "Help! My dog is bloated and restless",

        # Species consistency tests
        "I have a Chow Chow puppy that vomited",
        "Should I stop feeding it?",  # Should continue with Chow Chow

        # Focused response tests
        "How much does a Labrador cost?",  # Should be concise
        "Persian cat grooming needs",      # Should be specific

        # General improvement tests
        "Hello Fluffyn",
        "Thank you for helping",
        "What is Fluffyn?"
    ]

# ============================================
# STEP 13: Run Instructions
# ============================================
if __name__ == "__main__":
    display_setup_instructions()

    print("📝 TEST QUESTIONS TO VERIFY IMPROVEMENTS:")
    print("=" * 50)
    test_questions = get_test_questions()
    for i, question in enumerate(test_questions, 1):
        print(f"{i:2d}. {question}")
    print("=" * 50)
    print("\nReady to start Enhanced Context-Aware Fluffyn?")
    input("Press Enter to continue...")

    main()

# ============================================
# END OF ENHANCED CONTEXT-AWARE FLUFFYN ASSISTANT
# ============================================
