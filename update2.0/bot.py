# ============================================
# FLUFFYN AI ASSISTANT - GOOGLE COLAB VERSION
# FIXED: Enhanced Breed Recognition & Knowledge Retrieval
# ============================================

# ============================================
# STEP 1: Install Required Dependencies
# ============================================
!pip install -q langchain langchain-google-genai langchain-community
!pip install -q chromadb sentence-transformers
!pip install -q google-cloud-firestore firebase-admin
!pip install -q textblob faiss-cpu tiktoken
!pip install -q google-generativeai google-ai-generativelanguage
!pip install -q pinecone-client  # Using the correct package name

# ============================================
# STEP 2: Import Libraries
# ============================================
import json
import os
import re
import random
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib
import numpy as np

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Alternative imports for the combined approach
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Fixed Pinecone imports - using the correct modern syntax
from pinecone import Pinecone, ServerlessSpec

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
    """Enhanced breed matching with fuzzy matching and aliases"""
    
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
            'great lane': 'Great Dane',  # Fixed the issue from logs
            'dane': 'Great Dane',
            'border': 'Border Collie',
            'collie': 'Border Collie',
            'bulldog': 'English Bulldog',
            'french bulldog': 'French Bulldog',
            'frenchie': 'French Bulldog',
            'mastiff': 'English Mastiff',
            'rotty': 'Rottweiler',
            'rott': 'Rottweiler',
            'pocket spanish': 'Cocker Spaniel',  # Another fix from logs
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
# STEP 4: Enhanced Text Processing
# ============================================
class TextProcessor:
    """Enhanced text processing with spell correction and normalization"""

    def __init__(self):
        self.breed_matcher = BreedMatcher()

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
        """Extract breed and topic from query"""
        normalized = self.normalize_text(text)
        breeds = self.breed_matcher.find_breed_in_text(normalized)
        
        # Extract topic/intent
        topic = self._extract_topic(normalized)
        
        return breeds, topic

    def _extract_topic(self, text):
        """Extract the main topic from text"""
        text_lower = text.lower()
        
        topic_patterns = {
            'temperament': ['temperament', 'personality', 'behavior', 'nature', 'character', 'traits'],
            'training': ['training', 'train', 'command', 'obedience', 'teach', 'learn'],
            'health': ['health', 'medical', 'disease', 'condition', 'illness', 'eye issues', 'problems'],
            'grooming': ['grooming', 'brushing', 'care', 'maintenance', 'coat', 'fur'],
            'price': ['price', 'cost', 'budget', 'expensive', 'affordable', 'range', 'money'],
            'exercise': ['exercise', 'activity', 'walk', 'energy', 'active', 'physical'],
            'feeding': ['food', 'feeding', 'diet', 'nutrition', 'eat', 'meal'],
            'size': ['size', 'big', 'small', 'large', 'weight', 'height'],
            'general_info': ['information', 'about', 'tell', 'describe', 'details']
        }
        
        for topic, keywords in topic_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return topic
        
        return 'general_info'

# ============================================
# STEP 5: Firebase Session Manager (Unchanged)
# ============================================
class FirestoreSessionManager:
    """Manage chat sessions using Firestore with enhanced context tracking"""

    def __init__(self, db, collection_name="fluffyn_sessions"):
        self.db = db
        self.collection = db.collection(collection_name)
        self.current_session_id = None

    def create_session(self, user_id="default_user"):
        """Create a new chat session"""
        session_id = hashlib.md5(f"{user_id}_{datetime.now()}".encode()).hexdigest()[:12]
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'messages': [],
            'context': {
                'breeds_mentioned': [],
                'topics_discussed': [],
                'last_question_type': None,
                'conversation_theme': None
            },
            'interaction_count': 0
        }
        self.collection.document(session_id).set(session_data)
        self.current_session_id = session_id
        return session_id

    def add_message(self, session_id, user_input, bot_response, question_type=None, context_updates=None):
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
                'timestamp': datetime.now(),
                'question_type': question_type
            })

            # Keep last 15 messages for context
            if len(messages) > 15:
                messages = messages[-15:]

            # Update context
            if context_updates:
                for key, value in context_updates.items():
                    if key in context and isinstance(context[key], list):
                        context[key].extend(value if isinstance(value, list) else [value])
                        context[key] = list(set(context[key]))  # Remove duplicates
                    else:
                        context[key] = value

            # Update last question type
            context['last_question_type'] = question_type

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
                'messages': data.get('messages', [])[-10:],  # Last 10 messages
                'context': data.get('context', {}),
                'interaction_count': data.get('interaction_count', 0)
            }
        return None

# ============================================
# STEP 6: Enhanced Pinecone Knowledge Retrieval System
# ============================================
class PineconeKnowledgeRetriever:
    """Enhanced Pinecone retrieval with better breed matching"""

    def __init__(self, gemini_api_key, pinecone_api_key, pinecone_env="us-east-1"):
        self.api_key = gemini_api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_env = pinecone_env
        self.text_processor = TextProcessor()
        self.breed_matcher = BreedMatcher()
        
        # Initialize Pinecone with modern syntax
        self.setup_pinecone()

        # Initialize Gemini models with retry logic
        self.setup_gemini_models()

    def setup_pinecone(self):
        """Setup Pinecone vector database with updated syntax"""
        try:
            # Initialize Pinecone with the new v3 syntax
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Create index if it doesn't exist
            self.index_name = "fluffyn-pet-knowledge"
            
            # Check if index exists and create if not
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"Creating index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # SentenceTransformer dimension for all-MiniLM-L6-v2
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.pinecone_env
                    )
                )
                print("Index created successfully!")
                # Wait for index to be ready
                time.sleep(10)
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            # Initialize embedding model
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Pinecone setup completed!")
        
        except Exception as e:
            print(f"❌ Pinecone setup failed: {e}")
            raise

    def setup_gemini_models(self):
        """Setup Gemini models with error handling"""
        try:
            # Verify API key is not empty
            if not self.api_key or self.api_key.strip() == "":
                raise ValueError("Gemini API key is empty or invalid")
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.api_key,
                temperature=0.7,
                max_output_tokens=500
            )

            # Also setup direct Gemini model for fallbacks
            genai.configure(api_key=self.api_key)
            self.direct_model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Test the API key by making a simple call
            test_response = self.direct_model.generate_content("Hello")
            print("✅ Gemini API key verified and working!")
            
        except Exception as e:
            print(f"❌ Gemini setup failed: {e}")
            print("Please check your API key and try again.")
            raise

    def load_knowledge_base(self, json_path):
        """Load knowledge base into Pinecone with enhanced breed indexing"""
        print("Loading knowledge base...")
        with open(json_path, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)

        # Process documents for Pinecone in smaller batches
        vectors_to_upsert = []
        
        for key, value in knowledge.items():
            # Create comprehensive text with breed emphasis
            text_parts = [f"Breed: {key}"]

            # Add breed variations for better matching
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
            embedding = self.embedder.encode(doc_text).tolist()

            # Create enhanced metadata
            metadata = {
                "source": key,
                "type": "pet_knowledge",
                "category": self._categorize_content(key, value),
                "breed_type": self.breed_matcher.get_breed_type(key),
                "text": doc_text[:1500],  # Store more text for reference
                "breed_names": breeds_in_key  # Store recognized breed names
            }

            vectors_to_upsert.append({
                "id": key.replace(" ", "_").lower(),
                "values": embedding,
                "metadata": metadata
            })

            # Upsert in batches of 50 to avoid rate limits
            if len(vectors_to_upsert) >= 50:
                try:
                    self.index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []
                    time.sleep(0.1)  # Small delay to avoid rate limits
                except Exception as e:
                    print(f"Error upserting batch: {e}")
                    time.sleep(1)

        # Upsert any remaining vectors
        if vectors_to_upsert:
            try:
                self.index.upsert(vectors=vectors_to_upsert)
            except Exception as e:
                print(f"Error upserting final batch: {e}")

        print(f"✅ Loaded {len(knowledge)} documents into Pinecone")
        
        # Wait for indexing to complete
        time.sleep(2)

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

    def retrieve_with_pinecone(self, query, breeds=None, topic=None, top_k=5):
        """Enhanced retrieval with breed-specific matching"""
        try:
            # Build enhanced query
            enhanced_query_parts = []
            
            # Add the original query
            enhanced_query_parts.append(query)
            
            # Add breed information if available
            if breeds:
                for breed in breeds:
                    enhanced_query_parts.append(f"breed {breed}")
            
            # Add topic information if available
            if topic and topic != 'general_info':
                enhanced_query_parts.append(f"{topic} information")
            
            enhanced_query = " ".join(enhanced_query_parts)
            
            # Generate query embedding
            query_embedding = self.embedder.encode(enhanced_query).tolist()

            # Query Pinecone with filters if breed is specified
            filter_dict = {}
            if breeds:
                # Try to match breeds in metadata
                filter_dict = {
                    "$or": [
                        {"source": {"$in": breeds}},
                        {"breed_names": {"$in": breeds}}
                    ]
                }

            # Query with or without filters
            if filter_dict:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict
                )
            else:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )

            # Extract relevant chunks with better scoring
            relevant_chunks = []
            for match in results.matches:
                if match.score > 0.3:  # Lower threshold for better recall
                    relevant_chunks.append(match.metadata.get('text', ''))

            # If no good results with filters, try without filters
            if not relevant_chunks and filter_dict:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
                for match in results.matches:
                    if match.score > 0.3:
                        relevant_chunks.append(match.metadata.get('text', ''))

            return relevant_chunks

        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []

# ============================================
# STEP 7: Enhanced Conversation Handler
# ============================================
class ContextAwareConversationHandler:
    """Enhanced conversation handler with better breed recognition"""

    def __init__(self, knowledge_retriever, session_manager):
        self.retriever = knowledge_retriever
        self.session_manager = session_manager
        self.text_processor = TextProcessor()
        self.breed_matcher = BreedMatcher()

    def classify_input(self, user_input, context=None):
        """Enhanced input classification"""
        normalized = self.text_processor.normalize_text(user_input).lower()

        # Greeting patterns
        if re.search(r'\b(hi|hello|hey|good morning|good evening|hola|hai)\b', normalized):
            return "greeting"

        # Thank you patterns
        elif re.search(r'\b(thank you|thanks|thx|appreciated|grateful)\b', normalized):
            return "thank"

        # Company info patterns
        elif re.search(r'\b(fluffyn|about fluffyn|company|mission|what is fluffyn)\b', normalized):
            return "company_info"

        # Check for breed mentions
        breeds = self.breed_matcher.find_breed_in_text(normalized)
        if breeds:
            return "pet_breed_specific"

        # General pet-related
        elif self._is_pet_related(normalized, context):
            return "pet_general"

        else:
            return "unrelated"

    def _is_pet_related(self, text, context):
        """Check if input is pet-related"""
        pet_keywords = [
            'dog', 'cat', 'pet', 'animal', 'puppy', 'kitten', 'breed', 'training',
            'feeding', 'health', 'grooming', 'care', 'behavior', 'exercise',
            'temperament', 'personality', 'price', 'cost', 'budget'
        ]

        return any(keyword in text for keyword in pet_keywords)

# ============================================
# STEP 8: Enhanced Main Fluffyn Assistant Class
# ============================================
class FluffynAssistant:
    """Enhanced Fluffyn Assistant with improved breed handling"""

    def __init__(self, gemini_api_key, pinecone_api_key, pinecone_env, firebase_db):
        self.api_key = gemini_api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_env = pinecone_env
        self.db = firebase_db
        self.session_manager = FirestoreSessionManager(firebase_db)
        self.text_processor = TextProcessor()
        self.breed_matcher = BreedMatcher()
        
        print("Initializing Pinecone retriever...")
        self.knowledge_retriever = PineconeKnowledgeRetriever(
            gemini_api_key, pinecone_api_key, pinecone_env
        )
        
        print("Setting up conversation handler...")
        self.conversation_handler = ContextAwareConversationHandler(
            self.knowledge_retriever, self.session_manager
        )
        
        self.current_session_id = None
        print("✅ Fluffyn Assistant initialized successfully!")

    def load_knowledge_base(self, json_path):
        """Load knowledge base into Pinecone"""
        print("📚 Loading knowledge base into Pinecone...")
        self.knowledge_retriever.load_knowledge_base(json_path)
        print("✅ Knowledge base loaded successfully!")

    def start_session(self, user_id="default_user"):
        """Start a new chat session"""
        self.current_session_id = self.session_manager.create_session(user_id)
        print(f"📝 Session started: {self.current_session_id}")
        return self.current_session_id

    def generate_response(self, user_input, context):
        """Enhanced response generation with better breed handling"""
        # Extract breeds and topics from input
        breeds, topic = self.text_processor.extract_breed_and_topic(user_input)
        
        input_type = self.conversation_handler.classify_input(user_input, context['context'] if context else None)

        if input_type == "greeting":
            return self._generate_greeting()

        elif input_type == "thank":
            return self._generate_thank_response()

        elif input_type == "company_info":
            return self._generate_company_info()

        elif input_type in ["pet_breed_specific", "pet_general"]:
            return self._generate_pet_response(user_input, context, breeds, topic)

        else:  # unrelated
            return self._generate_unrelated_response()

    def _generate_pet_response(self, user_input, session_context, breeds, topic):
        """Enhanced pet response generation"""
        context = session_context['context'] if session_context else {}
        chat_history = session_context['messages'] if session_context else []

        # If no breeds detected but we have context, use context breeds
        if not breeds and context and context.get('breeds_mentioned'):
            breeds = context['breeds_mentioned'][-1:]

        # Try Pinecone retrieval with breed and topic context
        try:
            relevant_chunks = self.knowledge_retriever.retrieve_with_pinecone(
                user_input, breeds=breeds, topic=topic, top_k=5
            )

            if relevant_chunks:
                return self._generate_rag_response(
                    user_input, relevant_chunks, chat_history, context, breeds, topic
                )
            else:
                # Fallback to general breed information if we have breeds
                if breeds:
                    return self._generate_breed_fallback_response(breeds[0], topic, user_input)
                else:
                    return self._generate_general_pet_response(user_input, chat_history)

        except Exception as e:
            print(f"Error in retrieval: {e}")
            return self._generate_fallback_response()

    def _generate_rag_response(self, query, relevant_chunks, chat_history, context, breeds, topic):
        """Generate enhanced RAG response"""
        knowledge_context = "\n\n".join(relevant_chunks)

        # Build conversation history context
        history_context = ""
        if chat_history and len(chat_history) > 1:
            recent_history = chat_history[-2:]
            history_context = "\n\nPrevious conversation:\n"
            for entry in recent_history:
                history_context += f"User: {entry['user']}\nFluffyn: {entry['assistant'][:100]}...\n"

        # Build context info
        context_info = ""
        if breeds:
            context_info += f"\nBreeds in question: {', '.join(breeds)}"
        if topic:
            context_info += f"\nTopic focus: {topic}"

        prompt = f"""You are Fluffyn, the expert AI pet care assistant. Answer the user's question using the knowledge base information provided.

CRITICAL INSTRUCTIONS:
1. Base your answer primarily on the knowledge base information
2. Be specific and detailed (200-400 words)
3. Use warm, caring tone with appropriate emojis (🐕🐱🐾💕)
4. When discussing costs, provide prices in Indian Rupees (₹)
5. Be accurate and helpful
6. Answer the specific question directly

[Knowledge Base Information]
{knowledge_context}

{context_info}
{history_context}

[Current Question]
{query}

[Your Expert Response]
"""

        try:
            response = self.knowledge_retriever.direct_model.generate_content(prompt)
            base_response = response.text.strip()

            # Add follow-up questions
            follow_ups = self._generate_followup_questions(query, breeds, topic)
            if follow_ups:
                base_response += f"\n\n🤔 You might also want to know:\n"
                for i, q in enumerate(follow_ups[:3], 1):
                    base_response += f"{i}. {q}\n"

            return base_response

        except Exception as e:
            print(f"Error generating response: {e}")
            return self._generate_fallback_response()

    def _generate_breed_fallback_response(self, breed, topic, user_input):
        """Generate breed-specific fallback when no knowledge base match"""
        breed_type = self.breed_matcher.get_breed_type(breed)
        
        prompt = f"""You are Fluffyn, a friendly pet care assistant. The user is asking about {breed} ({breed_type}), specifically about {topic if topic != 'general_info' else 'general information'}.

User question: {user_input}

Provide helpful, accurate information about {breed} based on general veterinary and pet care knowledge. Be specific about this breed and include practical advice. Use 200-300 words with appropriate emojis. Include Indian Rupee (₹) pricing where relevant.
"""

        try:
            response = self.knowledge_retriever.direct_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating breed fallback: {e}")
            return self._generate_fallback_response()

    def _generate_general_pet_response(self, query, chat_history):
        """Generate general pet advice when no specific knowledge found"""
        history_context = ""
        if chat_history and len(chat_history) > 1:
            recent_history = chat_history[-2:]
            history_context = "\n\nPrevious conversation:\n"
            for entry in recent_history:
                history_context += f"User: {entry['user']}\nFluffyn: {entry['assistant'][:100]}...\n"

        prompt = f"""You are Fluffyn, a friendly pet care assistant. The user is asking about pets, but I don't have specific information in my knowledge base for this question.

{history_context}

Current question: {query}

Provide helpful, general pet care advice based on common veterinary knowledge. Be warm, caring, and include practical tips. Use 150-250 words and appropriate emojis. Include Indian Rupee (₹) pricing where relevant.
"""

        try:
            response = self.knowledge_retriever.direct_model.generate_content(prompt)
            return response.text.strip()
        except:
            return self._generate_fallback_response()

    def _generate_greeting(self):
        """Generate greeting response"""
        current_hour = datetime.now().hour
        time_greeting = "Good morning" if current_hour < 12 else "Good afternoon" if current_hour < 17 else "Good evening"

        greetings = [
            f"{time_greeting}! 🐾 I'm Fluffyn, your AI pet care companion! How can I help you with your pet needs today?",
            f"Hello! 🐕 Welcome to Fluffyn! I'm here to help with all your pet care questions and advice!",
            f"Hey there! 🐱 Fluffyn here, ready to assist with pets, breeds, training, and care tips!"
        ]
        return random.choice(greetings)

    def _generate_thank_response(self):
        """Generate thank you response"""
        responses = [
            "You're very welcome! 🐾 Fluffyn is always here to help with your pet care journey!",
            "My pleasure! 🐕 That's what Fluffyn is for - making pet parenting easier and more joyful!",
            "Glad I could help! 🐱 Your pets are lucky to have such a caring owner!"
        ]
        return random.choice(responses)

    def _generate_company_info(self):
        """Generate company information"""
        return """🐾 Welcome to Fluffyn! 🐾

We're your trusted AI-powered pet care companion, dedicated to making pet ownership joyful and stress-free!

Our Mission: Connecting pets with loving homes and supporting pet parents with expert care advice.

What Fluffyn Offers:
🐶 Expert pet care guidance and breed information
💡 Health, nutrition, and training advice
❤ Support for first-time and experienced pet owners
🏠 Personalized recommendations for your lifestyle

Whether you're choosing your first pet or caring for a long-time companion, Fluffyn is here to help every step of the way! How can I assist you today?"""

    def _generate_unrelated_response(self):
        """Generate response for unrelated questions"""
        responses = [
            "I'm Fluffyn, your pet care specialist! 🐾 I focus on helping with pets, breeds, training, health, and care advice. What would you like to know about pets?",
            "Hi! I'm here to help with all things pet-related! 🐕🐱 Ask me about pet care, training, health, or choosing the right breed for you!",
            "I specialize in pet care and advice! 🐾 Whether you need help with training, health, grooming, or choosing a pet, I'm here to help!"
        ]
        return random.choice(responses)

    def _generate_fallback_response(self):
        """Generate fallback response"""
        return """I'd love to help you with that! 🐾 While I'm having a moment accessing my knowledge base, I'm still here for all your pet care needs.

Please feel free to ask me about:
• Specific dog or cat breeds
• Pet training and behavior tips
• Health and nutrition advice
• Grooming and care instructions

What would you like to know about pets?"""

    def _generate_followup_questions(self, user_input, breeds, topic):
        """Generate contextual follow-up questions"""
        follow_ups = []

        # Breed-specific questions
        if breeds:
            breed = breeds[0]
            if topic == 'health':
                follow_ups = [
                    f"What are common health issues in {breed}?",
                    f"What preventive care does {breed} need?",
                    f"How often should {breed} visit the vet?"
                ]
            elif topic == 'training':
                follow_ups = [
                    f"What training techniques work best for {breed}?",
                    f"How intelligent is {breed}?",
                    f"Are {breed} easy to house train?"
                ]
            elif topic == 'grooming':
                follow_ups = [
                    f"How often should I groom {breed}?",
                    f"What grooming tools do I need for {breed}?",
                    f"Does {breed} shed a lot?"
                ]
            elif topic == 'price':
                follow_ups = [
                    f"What are the ongoing costs for {breed}?",
                    f"How much does {breed} food cost monthly?",
                    f"What are veterinary costs for {breed}?"
                ]
            else:
                follow_ups = [
                    f"What's the temperament of {breed}?",
                    f"What are {breed} exercise needs?",
                    f"Is {breed} good with children?"
                ]

        # General topic questions
        elif topic == 'health':
            follow_ups = [
                "What are signs of a healthy pet?",
                "How often should pets visit the vet?",
                "What vaccinations do pets need?"
            ]
        elif topic == 'training':
            follow_ups = [
                "What are basic training commands?",
                "How long does training usually take?",
                "What are common training mistakes?"
            ]

        return follow_ups[:3]  # Return max 3 questions

    def chat(self, user_input):
        """Enhanced chat function with better breed handling"""
        # Get current session context
        session_context = self.session_manager.get_session_context(self.current_session_id)

        # Extract breeds and topics for context tracking
        breeds, topic = self.text_processor.extract_breed_and_topic(user_input)

        # Generate response
        response = self.generate_response(user_input, session_context)

        # Extract context updates
        context_updates = {}
        if breeds:
            context_updates['breeds_mentioned'] = breeds
        if topic and topic != 'general_info':
            context_updates['topics_discussed'] = [topic]

        # Log the interaction
        input_type = self.conversation_handler.classify_input(
            user_input, session_context['context'] if session_context else None
        )
        
        self.session_manager.add_message(
            self.current_session_id,
            user_input,
            response,
            question_type=input_type,
            context_updates=context_updates
        )

        return response

    def display_welcome(self):
        """Display welcome message"""
        print("=" * 70)
        print("🐾 FLUFFYN - AI PET CARE ASSISTANT (Enhanced & Fixed) 🐾")
        print("=" * 70)
        print("Hi! I'm Fluffyn, your intelligent pet care companion!")
        print()
        print("🌟 What makes me special:")
        print("   🧠 Enhanced breed recognition for all supported breeds")
        print("   🔍 Extensive pet knowledge base with Pinecone search")
        print("   💬 Smart context-aware conversations")
        print("   🎯 Precise answers for breed-specific questions")
        print("   📱 Secure session storage with Firebase")
        print("   ⚡ Fast response times with error handling")
        print()
        print("🐕 Supported Dog Breeds:")
        dog_list = ", ".join(self.breed_matcher.dog_breeds[:10]) + "... and many more!"
        print(f"   {dog_list}")
        print()
        print("🐱 Supported Cat Breeds:")
        cat_list = ", ".join(self.breed_matcher.cat_breeds[:10]) + "... and many more!"
        print(f"   {cat_list}")
        print()
        print("Ask me about breeds, training, health, grooming, costs, or anything pet-related!")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("-" * 70)

    def chat_loop(self):
        """Enhanced chat loop with better error handling"""
        self.display_welcome()
        self.start_session()

        while True:
            try:
                user_input = input("\n🐾 You: ").strip()

                # Check for exit
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    session_context = self.session_manager.get_session_context(self.current_session_id)
                    interaction_count = session_context['interaction_count'] if session_context else 0
                    print(f"\nFluffyn: Goodbye! 🐾 Thanks for {interaction_count} great interactions! Come back anytime for pet advice!")
                    break

                # Skip empty input
                if not user_input:
                    print("\nFluffyn: I'm here to help! Ask me anything about pets!")
                    continue

                # Get response
                start_time = time.time()
                response = self.chat(user_input)
                end_time = time.time()

                print(f"\nFluffyn: {response}")
                print(f"⏱ Response time: {end_time - start_time:.2f} seconds")

            except KeyboardInterrupt:
                print(f"\n\nFluffyn: Thanks for choosing Fluffyn! 🐾")
                break
            except Exception as e:
                print(f"\nFluffyn: I encountered an issue: {str(e)}")
                print("Please try again! I'm here to help with pet questions!")

# ============================================
# STEP 9: Firebase Setup Function (Unchanged)
# ============================================
def setup_firebase():
    """Setup Firebase with service account credentials"""
    print("🔐 Setting up Firebase Authentication...")

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
    """Enhanced main function with better error handling"""
    print("🚀 Initializing Enhanced Fluffyn AI Assistant...")

    # Setup Firebase
    try:
        db = setup_firebase()
    except Exception as e:
        print(f"❌ Firebase setup failed: {e}")
        print("Please check your Firebase credentials and try again.")
        return

    # Upload knowledge base
    print("\n📁 Please upload your JSON knowledge base file:")
    try:
        uploaded = files.upload()
        json_path = list(uploaded.keys())[0]
        print(f"✅ Knowledge base file uploaded: {json_path}")
    except Exception as e:
        print(f"❌ File upload failed: {e}")
        return

    # Get API keys with validation
    gemini_api_key = input("\n🔑 Enter your Gemini API Key: ").strip()
    if not gemini_api_key:
        print("❌ Gemini API key is required!")
        return

    pinecone_api_key = input("🔑 Enter your Pinecone API Key: ").strip()
    if not pinecone_api_key:
        print("❌ Pinecone API key is required!")
        return

    pinecone_env = input("🌍 Enter your Pinecone Environment (default: us-east-1): ").strip()
    if not pinecone_env:
        pinecone_env = "us-east-1"

    print("\n🔧 Validating API keys and setting up services...")

    # Initialize assistant with error handling
    try:
        assistant = FluffynAssistant(gemini_api_key, pinecone_api_key, pinecone_env, db)
        print("✅ Assistant initialized successfully!")
    except Exception as e:
        print(f"❌ Assistant initialization failed: {e}")
        print("Please check your API keys and try again.")
        return

    # Load knowledge base with error handling
    try:
        assistant.load_knowledge_base(json_path)
        print("✅ Knowledge base loaded successfully!")
    except Exception as e:
        print(f"❌ Knowledge base loading failed: {e}")
        return

    # Start chat loop
    print("\n🎉 Everything is ready! Starting Fluffyn...")
    assistant.chat_loop()

# ============================================
# STEP 11: Enhanced Colab Setup Instructions
# ============================================
def display_colab_instructions():
    """Display enhanced setup instructions"""
    print("""
=================================================================
📋 GOOGLE COLAB SETUP INSTRUCTIONS - FLUFFYN ENHANCED
=================================================================

🔧 FIXES IMPLEMENTED:
✅ Enhanced breed recognition with fuzzy matching
✅ Better API key validation and error handling  
✅ Improved Pinecone indexing and retrieval
✅ Fixed breed name variations and common misspellings
✅ Enhanced context awareness for follow-up questions
✅ Better fallback responses when knowledge base fails

=================================================================
🚀 SETUP STEPS:

1. 🔑 GET API KEYS:
   • Gemini API: https://ai.google.dev/ → Get API Key
   • Pinecone API: https://www.pinecone.io/ → Sign up → Get API Key

2. 🔥 SETUP FIREBASE:
   • Firebase Console: https://console.firebase.google.com/
   • Create project → Project Settings → Service Accounts
   • Generate new private key (JSON file)

3. 📊 PREPARE KNOWLEDGE BASE:
   • JSON file with pet breed information
   • Format: {"Breed Name": {"field": "value", ...}}

4. ▶️ RUN THE CODE:
   • Execute all cells in order
   • Upload Firebase JSON when prompted
   • Upload knowledge base JSON when prompted
   • Enter valid API keys
   • Start chatting!

=================================================================
🐾 BREED SUPPORT:
• All 55+ dog breeds fully supported with aliases
• All 25+ cat breeds fully supported  
• Smart breed name matching (handles misspellings)
• Context-aware follow-up questions

=================================================================
""")

# ============================================
# STEP 12: Run Instructions
# ============================================
if __name__ == "__main__":
    # Display setup instructions first
    display_colab_instructions()
    
    # Ask user if they want to proceed
    proceed = input("Ready to start Fluffyn Enhanced? Press Enter to continue: ").strip()
    
    # Run the main function
    main()

# ============================================
# END OF ENHANCED FLUFFYN AI ASSISTANT
# ============================================
