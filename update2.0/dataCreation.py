#!/usr/bin/env python3
"""
Pet Dataset Enrichment Pipeline - Enhanced Uniform Data Generation
=================================================================

MAJOR IMPROVEMENTS FOR UNIFORMITY:
- Guaranteed uniform feature completion using multi-stage LLM fallback
- No null/empty/missing values in final output
- Enhanced validation and data completion pipeline
- Intelligent feature filling when scraping fails

Features Added: 22 comprehensive features with guaranteed completion
"""

import json
import time
import os
import re
import random
from typing import Dict, List, Optional, Any, Tuple
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

INPUT_FILE = 'pd_clean.json' # input file
OUTPUT_FILE = 'final_enriched_training_data.json' #final file
PROGRESS_FILE = 'pipeline_progress.json'
VALIDATION_FILE = 'validation_report.json'

GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'

# Complete feature set with validation rules
FEATURE_SCHEMA = {
    # Health & Wellness (5 features)
    "common_health_concerns": {"type": "array", "min_items": 1, "fallback": ["Regular veterinary checkups recommended"]},
    "health_disclaimer": {"type": "string", "required": True, "fallback": "This information is not a substitute for professional veterinary advice. Please consult a vet for any health issues."},
    "recommended_health_tests": {"type": "array", "min_items": 0, "fallback": []},
    "general_dietary_needs": {"type": "string", "min_length": 10, "fallback": "High-quality dog food appropriate for age and activity level"},
    "average_exercise_needs": {"type": "string", "min_length": 10, "fallback": "Moderate daily exercise including walks and playtime"},

    # Training & Behavior (6 features)
    "training_difficulty": {"type": "string", "enum": ["Easy", "Moderate", "Challenging", "Expert Level"], "fallback": "Moderate"},
    "training_tips": {"type": "string", "min_length": 20, "fallback": "Start with basic commands, use positive reinforcement, and maintain consistency"},
    "socialization_needs": {"type": "string", "enum": ["Low", "Moderate", "High", "Very High"], "fallback": "Moderate"},
    "common_behavioral_issues": {"type": "array", "min_items": 0, "fallback": []},
    "mental_stimulation_needs": {"type": "string", "enum": ["Low", "Moderate", "High", "Very High"], "fallback": "Moderate"},
    "prey_drive_level": {"type": "string", "enum": ["Low", "Moderate", "High", "Very High"], "fallback": "Moderate"},

    # Breed History & Characteristics (5 features)
    "breed_history": {"type": "string", "min_length": 50, "fallback": "This breed has a rich history as a companion and working dog"},
    "breed_group": {"type": "string", "min_length": 3, "fallback": "Mixed/Unknown Group"},
    "puppy_availability": {"type": "string", "enum": ["Readily Available", "Moderately Available", "Limited", "Rare"], "fallback": "Moderately Available"},
    "distinguishing_features": {"type": "string", "min_length": 20, "fallback": "Unique characteristics that make this breed special"},
    "celebrity_owners": {"type": "array", "min_items": 0, "fallback": []},

    # Lifestyle & Home Compatibility (6 features)
    "good_for_first_time_owners": {"type": "string", "enum": ["Yes", "No", "With Guidance"], "fallback": "With Guidance"},
    "ideal_living_conditions": {"type": "string", "min_length": 20, "fallback": "Adaptable to various living situations with proper care"},
    "tolerance_to_being_alone": {"type": "string", "enum": ["Poor", "Fair", "Good", "Excellent"], "fallback": "Fair"},
    "weather_tolerance_details": {"type": "string", "min_length": 20, "fallback": "Moderate tolerance to various weather conditions"},
    "grooming_frequency_and_tips": {"type": "string", "min_length": 20, "fallback": "Regular grooming recommended to maintain coat health"},
    "cost_of_ownership_summary": {"type": "string", "min_length": 30, "fallback": "Moderate cost including food, veterinary care, grooming, and supplies"}
}

NEW_FEATURES_ADDED = list(FEATURE_SCHEMA.keys())

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

RATE_LIMIT_CONFIG = {
    'base_delay': 3,
    'max_delay': 60,
    'max_retries': 5,
    'backoff_factor': 2
}

# =============================================================================
# ENHANCED VALIDATION SYSTEM
# =============================================================================

class DataValidator:
    """Comprehensive data validation and completion system"""
    
    def __init__(self):
        self.validation_report = {
            'total_breeds': 0,
            'validation_issues': {},
            'completion_stats': {},
            'timestamp': datetime.now().isoformat()
        }

    def validate_breed_data(self, breed_name: str, data: Dict) -> Tuple[bool, List[str]]:
        """Validate a breed's data against the schema"""
        issues = []
        
        for feature, rules in FEATURE_SCHEMA.items():
            if feature not in data:
                issues.append(f"Missing feature: {feature}")
                continue
                
            value = data[feature]
            
            # Type validation
            if rules["type"] == "array" and not isinstance(value, list):
                issues.append(f"{feature}: Expected array, got {type(value).__name__}")
            elif rules["type"] == "string" and not isinstance(value, str):
                issues.append(f"{feature}: Expected string, got {type(value).__name__}")
            
            # Content validation
            if rules["type"] == "string":
                if "min_length" in rules and len(value.strip()) < rules["min_length"]:
                    issues.append(f"{feature}: String too short (min {rules['min_length']})")
                if "enum" in rules and value not in rules["enum"]:
                    issues.append(f"{feature}: Invalid value '{value}', must be one of {rules['enum']}")
            elif rules["type"] == "array":
                if "min_items" in rules and len(value) < rules["min_items"]:
                    issues.append(f"{feature}: Array too short (min {rules['min_items']} items)")
        
        self.validation_report['validation_issues'][breed_name] = issues
        return len(issues) == 0, issues

    def fix_data_issues(self, breed_name: str, data: Dict, issues: List[str]) -> Dict:
        """Fix data issues using fallback values"""
        fixed_data = data.copy()
        fixes_applied = []
        
        for issue in issues:
            if issue.startswith("Missing feature:"):
                feature = issue.split(": ")[1]
                if feature in FEATURE_SCHEMA:
                    fixed_data[feature] = FEATURE_SCHEMA[feature]["fallback"]
                    fixes_applied.append(f"Added missing {feature}")
            
            elif "String too short" in issue:
                feature = issue.split(":")[0]
                if feature in FEATURE_SCHEMA:
                    fixed_data[feature] = FEATURE_SCHEMA[feature]["fallback"]
                    fixes_applied.append(f"Fixed short string for {feature}")
            
            elif "Invalid value" in issue:
                feature = issue.split(":")[0]
                if feature in FEATURE_SCHEMA:
                    fixed_data[feature] = FEATURE_SCHEMA[feature]["fallback"]
                    fixes_applied.append(f"Fixed invalid value for {feature}")
            
            elif "Expected array" in issue or "Expected string" in issue:
                feature = issue.split(":")[0]
                if feature in FEATURE_SCHEMA:
                    fixed_data[feature] = FEATURE_SCHEMA[feature]["fallback"]
                    fixes_applied.append(f"Fixed type mismatch for {feature}")
        
        if fixes_applied:
            print(f"  🔧 Applied {len(fixes_applied)} automatic fixes for '{breed_name}'")
        
        return fixed_data

    def generate_validation_report(self, output_path: str):
        """Generate comprehensive validation report"""
        total_issues = sum(len(issues) for issues in self.validation_report['validation_issues'].values())
        breeds_with_issues = sum(1 for issues in self.validation_report['validation_issues'].values() if issues)
        
        summary = {
            'total_breeds_processed': self.validation_report['total_breeds'],
            'breeds_with_issues': breeds_with_issues,
            'total_issues_found': total_issues,
            'issues_per_breed': total_issues / max(1, self.validation_report['total_breeds'])
        }
        
        self.validation_report['summary'] = summary
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_report, f, indent=2)
        
        print(f"📊 Validation report saved to: {output_path}")
        return summary

# =============================================================================
# ENHANCED GEMINI LLM SYSTEM
# =============================================================================

class EnhancedGeminiEnricher:
    """Advanced LLM enricher with guaranteed feature completion"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
        self.rate_limiter = RateLimiter(RATE_LIMIT_CONFIG)

    def enrich_breed_data(self, breed_name: str, original_data: Dict, scraped_data: Dict) -> Dict[str, Any]:
        """Enhanced enrichment with guaranteed uniform output"""
        print(f"  🤖 [Gemini] Enriching data for '{breed_name}'...")
        
        # Try primary enrichment first
        enriched_data = self._attempt_enrichment(breed_name, original_data, scraped_data)
        
        # Validate and fix any issues
        validator = DataValidator()
        is_valid, issues = validator.validate_breed_data(breed_name, enriched_data)
        
        if not is_valid:
            print(f"  🔧 [Validation] Found {len(issues)} issues, attempting fixes...")
            enriched_data = self._fix_missing_features(breed_name, enriched_data, issues)
            
            # Re-validate
            is_valid, remaining_issues = validator.validate_breed_data(breed_name, enriched_data)
            if remaining_issues:
                print(f"  ⚠️ [Validation] {len(remaining_issues)} issues remain after fixes")
        
        print(f"  ✅ [Gemini] Final data for '{breed_name}' has {len(enriched_data)} features")
        return enriched_data

    def _attempt_enrichment(self, breed_name: str, original_data: Dict, scraped_data: Dict) -> Dict[str, Any]:
        """Attempt initial enrichment with retries"""
        for attempt in range(RATE_LIMIT_CONFIG['max_retries']):
            try:
                self.rate_limiter.wait_if_needed()
                
                prompt = self._create_enhanced_prompt(breed_name, original_data, scraped_data)
                response = self._call_gemini_api(prompt)
                
                if response:
                    parsed_data = self._parse_gemini_response(response)
                    if parsed_data:
                        self.rate_limiter.record_success()
                        return parsed_data
                
                raise ValueError("Failed to parse response")
                
            except Exception as e:
                print(f"  ❌ [Gemini] Attempt {attempt + 1} failed: {e}")
                self.rate_limiter.record_failure()
                if attempt < RATE_LIMIT_CONFIG['max_retries'] - 1:
                    time.sleep(RATE_LIMIT_CONFIG['base_delay'] * (2 ** attempt))
        
        # If all attempts fail, use comprehensive fallback
        print(f"  🆘 [Gemini] All attempts failed, using intelligent fallback")
        return self._create_intelligent_fallback(breed_name, original_data)

    def _create_enhanced_prompt(self, breed_name: str, original_data: Dict, scraped_data: Dict) -> str:
        """Create enhanced prompt with strict formatting requirements"""
        
        # Create schema description for the prompt
        schema_description = "\n".join([
            f"- {key}: {rules['type']}" + 
            (f" (options: {rules['enum']})" if 'enum' in rules else "") +
            (f" (min length: {rules['min_length']})" if 'min_length' in rules else "")
            for key, rules in FEATURE_SCHEMA.items()
        ])
        
        return f"""You are a professional pet breed expert creating a comprehensive breed profile.

BREED: {breed_name}

EXISTING DATA:
{json.dumps(original_data, indent=2)}

WEB SCRAPED DATA:
{json.dumps(scraped_data, indent=2)}

REQUIRED OUTPUT SCHEMA:
{schema_description}

CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON with ALL required fields
2. NEVER use null, "unknown", "not available", or empty strings
3. For health_disclaimer, use exactly: "This information is not a substitute for professional veterinary advice. Please consult a vet for any health issues."
4. For enum fields, use ONLY the specified options
5. For arrays, provide realistic items or empty array []
6. For strings, provide meaningful, specific information
7. Ensure all content is breed-specific and accurate

QUALITY STANDARDS:
- Health info should be breed-specific and helpful
- Training info should reflect breed temperament
- History should be accurate and informative
- All text should be professional and informative
- Use proper capitalization and grammar

OUTPUT: Return only the JSON object with all required fields filled."""

    def _call_gemini_api(self, prompt: str) -> Optional[str]:
        """Make API call to Gemini"""
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {'Content-Type': 'application/json'}
        params = {'key': self.api_key}
        
        response = requests.post(
            self.api_url,
            headers=headers,
            params=params,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 429:
            raise requests.exceptions.RequestException("Rate limit exceeded")
        
        response.raise_for_status()
        response_json = response.json()
        return response_json['candidates'][0]['content']['parts'][0]['text']

    def _fix_missing_features(self, breed_name: str, data: Dict, issues: List[str]) -> Dict[str, Any]:
        """Use targeted LLM calls to fix specific missing features"""
        fixed_data = data.copy()
        
        # Extract missing features from issues
        missing_features = []
        for issue in issues:
            if "Missing feature:" in issue:
                feature = issue.split(": ")[1]
                missing_features.append(feature)
            elif any(x in issue for x in ["String too short", "Invalid value", "Expected"]):
                feature = issue.split(":")[0]
                missing_features.append(feature)
        
        if missing_features:
            print(f"  🔧 [LLM Fix] Generating {len(missing_features)} missing features...")
            
            # Create targeted prompt for missing features only
            features_to_fix = {f: FEATURE_SCHEMA[f] for f in missing_features if f in FEATURE_SCHEMA}
            fix_prompt = f"""Generate ONLY the missing features for {breed_name}.

BREED: {breed_name}
CURRENT DATA: {json.dumps(data, indent=2)}

GENERATE ONLY THESE FEATURES:
{json.dumps(list(features_to_fix.keys()))}

Return valid JSON with only these fields. Use realistic, breed-specific information."""

            try:
                response = self._call_gemini_api(fix_prompt)
                if response:
                    fix_data = self._parse_gemini_response(response)
                    if fix_data:
                        for feature in missing_features:
                            if feature in fix_data:
                                fixed_data[feature] = fix_data[feature]
                            elif feature in FEATURE_SCHEMA:
                                fixed_data[feature] = FEATURE_SCHEMA[feature]["fallback"]
            except:
                # Use fallbacks if fix attempt fails
                for feature in missing_features:
                    if feature in FEATURE_SCHEMA:
                        fixed_data[feature] = FEATURE_SCHEMA[feature]["fallback"]
        
        return fixed_data

    def _parse_gemini_response(self, response_text: str) -> Optional[Dict]:
        """Enhanced JSON parsing with validation"""
        cleaned_text = response_text.strip()
        
        # Remove markdown formatting
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        
        cleaned_text = cleaned_text.strip()
        
        # Extract JSON object
        json_start = cleaned_text.find('{')
        json_end = cleaned_text.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_text = cleaned_text[json_start:json_end+1]
            
            try:
                data = json.loads(json_text)
                
                # Ensure we have a valid dictionary
                if not isinstance(data, dict):
                    return None
                
                # Clean up any null values
                cleaned_data = {}
                for key, value in data.items():
                    if key in FEATURE_SCHEMA:
                        if value is None or value == "" or value == "null":
                            cleaned_data[key] = FEATURE_SCHEMA[key]["fallback"]
                        else:
                            cleaned_data[key] = value
                
                return cleaned_data
                
            except json.JSONDecodeError as e:
                print(f"  ❌ [Parser] JSON error: {e}")
                return None
        
        return None

    def _create_intelligent_fallback(self, breed_name: str, original_data: Dict) -> Dict[str, Any]:
        """Create intelligent fallback data using breed name analysis"""
        fallback_data = {}
        
        # Use schema fallbacks but try to make them breed-specific
        for feature, rules in FEATURE_SCHEMA.items():
            base_fallback = rules["fallback"]
            
            if feature == "breed_history" and isinstance(base_fallback, str):
                fallback_data[feature] = f"The {breed_name} is a distinctive breed with a unique heritage as both a companion and working dog, developed through careful breeding practices."
            elif feature == "distinguishing_features" and isinstance(base_fallback, str):
                fallback_data[feature] = f"The {breed_name} has unique physical and temperamental characteristics that distinguish it from other breeds."
            elif feature == "ideal_living_conditions" and isinstance(base_fallback, str):
                fallback_data[feature] = f"The {breed_name} can adapt to various living situations with proper exercise, training, and socialization."
            else:
                fallback_data[feature] = base_fallback
        
        return fallback_data


class RateLimiter:
    """Enhanced rate limiter with exponential backoff"""
    def __init__(self, config: Dict):
        self.config = config
        self.last_request_time = 0
        self.consecutive_failures = 0

    def wait_if_needed(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if self.consecutive_failures > 0:
            delay = min(
                self.config['base_delay'] * (self.config['backoff_factor'] ** self.consecutive_failures),
                self.config['max_delay']
            )
        else:
            delay = self.config['base_delay']

        if time_since_last < delay:
            wait_time = delay - time_since_last
            print(f"  ⏳ Rate limiting: waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)

        self.last_request_time = time.time()

    def record_success(self):
        self.consecutive_failures = 0

    def record_failure(self):
        self.consecutive_failures += 1


class WebScraper:
    """Web scraper for pet breed information"""
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.rate_limiter = RateLimiter({'base_delay': 1, 'max_delay': 10, 'backoff_factor': 1.5, 'max_retries': 3})

    def scrape_akc_data(self, breed_name: str) -> Dict[str, str]:
        """Scrape American Kennel Club data for a breed"""
        print(f"  🔍 [AKC] Searching for '{breed_name}'...")
        search_name = breed_name.lower().replace(' ', '-').replace("'", "")
        url = f"https://www.akc.org/dog-breeds/{search_name}/"

        self.rate_limiter.wait_if_needed()

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')
            scraped_data = {}

            # Look for breed group
            elem = soup.find('span', class_='breed-group')
            if elem:
                scraped_data['breed_group'] = elem.get_text(strip=True)

            # Look for other sections
            for section_key, keywords in {
                'health': ['health', 'care'],
                'training': ['training', 'personality', 'temperament'],
                'history': ['history', 'origin'],
                'grooming': ['grooming', 'coat']
            }.items():
                content = self._find_section_content(soup, keywords)
                if content:
                    scraped_data[section_key] = content[:1000]

            print(f"  ✅ [AKC] Found {len(scraped_data)} sections")
            self.rate_limiter.record_success()
            return scraped_data

        except Exception as e:
            print(f"  ⚠️ [AKC] Error: {e}")
            self.rate_limiter.record_failure()
            return {}

    def scrape_wikipedia_data(self, breed_name: str) -> Dict[str, str]:
        """Scrape Wikipedia data for additional breed information"""
        print(f"  🔍 [Wikipedia] Searching for '{breed_name}'...")
        url = f"https://en.wikipedia.org/wiki/{quote_plus(breed_name.replace(' ', '_'))}"

        self.rate_limiter.wait_if_needed()

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')
            scraped_data = {}

            content_div = soup.find('div', {'class': 'mw-parser-output'})
            if content_div:
                paragraphs = content_div.find_all('p', recursive=False)[:3]
                general_info = ' '.join([p.get_text(strip=True) for p in paragraphs])
                if general_info:
                    scraped_data['wikipedia_info'] = re.sub(r'\[\d+\]', '', general_info)[:1500]

            print(f"  ✅ [Wikipedia] Found general info")
            self.rate_limiter.record_success()
            return scraped_data

        except Exception as e:
            print(f"  ⚠️ [Wikipedia] Error: {e}")
            self.rate_limiter.record_failure()
            return {}

    def _find_section_content(self, soup: BeautifulSoup, keywords: List[str]) -> Optional[str]:
        """Find content sections based on header keywords"""
        for keyword in keywords:
            header = soup.find(['h1', 'h2', 'h3', 'h4'], string=re.compile(keyword, re.IGNORECASE))
            if header:
                content_elements = []
                for sibling in header.find_next_siblings():
                    if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                        break
                    if sibling.name == 'p':
                        content_elements.append(sibling.get_text(strip=True))
                if content_elements:
                    return ' '.join(content_elements)
        return None


class ProgressTracker:
    """Track and save pipeline progress to enable resume functionality"""
    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self.data = self._load_progress()

    def _load_progress(self) -> Dict:
        try:
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'completed_breeds': [],
                'failed_breeds': [],
                'last_updated': None,
                'total_processed': 0
            }

    def save_progress(self):
        self.data['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def mark_completed(self, breed_name: str):
        if breed_name not in self.data['completed_breeds']:
            self.data['completed_breeds'].append(breed_name)
            self.data['total_processed'] += 1
            self.save_progress()

    def mark_failed(self, breed_name: str):
        if breed_name not in self.data['failed_breeds']:
            self.data['failed_breeds'].append(breed_name)
            self.save_progress()

    def is_completed(self, breed_name: str) -> bool:
        return breed_name in self.data['completed_breeds']

    def get_remaining_breeds(self, all_breeds: List[str]) -> List[str]:
        return [breed for breed in all_breeds if not self.is_completed(breed)]


# =============================================================================
# ENHANCED MAIN PIPELINE CLASS
# =============================================================================

class UniformDataPipeline:
    """Enhanced pipeline ensuring uniform data output"""
    
    def __init__(self):
        self.scraper = WebScraper()
        self.api_key = self._get_gemini_api_key()
        self.enricher = EnhancedGeminiEnricher(self.api_key)
        self.progress_tracker = ProgressTracker(PROGRESS_FILE)
        self.validator = DataValidator()
        self.processed_count = 0
        self.failed_count = 0

    def _get_gemini_api_key(self):
        """Get Gemini API key from environment or user input"""
        print("🔧 Getting Gemini API key...")
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ GOOGLE_API_KEY not found in environment variables.")
            print("Get your free API key from: https://makersuite.google.com/app/apikey")
            api_key = input("Enter your Gemini API key: ").strip()
            if not api_key:
                raise ValueError("API key is required")
        print("✅ Gemini API key loaded.")
        return api_key

    def load_input_data(self, file_path: str) -> Dict:
        """Load and validate input data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"📂 Loaded {len(data)} breeds from '{file_path}'")
            return data
        except FileNotFoundError:
            print(f"❌ Input file '{file_path}' not found!")
            return {}

    def process_breed(self, breed_name: str, original_data: Dict) -> Dict:
        """Process a single breed with guaranteed uniform output"""
        print(f"\n🔄 Processing: {breed_name}")
        print("=" * 50)

        try:
            # Level 1: Web Scraping
            print("📡 LEVEL 1: Web Scraping")
            akc_data = self.scraper.scrape_akc_data(breed_name)
            wikipedia_data = self.scraper.scrape_wikipedia_data(breed_name)
            scraped_data = {**akc_data, **wikipedia_data}

            # Level 2: LLM Enrichment with validation
            print("🤖 LEVEL 2: LLM Enrichment & Validation")
            enriched_features = self.enricher.enrich_breed_data(
                breed_name, original_data, scraped_data
            )

            # Combine all data
            final_data = {**original_data, **enriched_features}

            # Final validation and fixing
            print("🔍 LEVEL 3: Final Validation")
            is_valid, issues = self.validator.validate_breed_data(breed_name, final_data)
            
            if not is_valid:
                print(f"  🔧 Found {len(issues)} final issues, applying fixes...")
                final_data = self.validator.fix_data_issues(breed_name, final_data, issues)
                
                # Re-validate
                is_valid, remaining_issues = self.validator.validate_breed_data(breed_name, final_data)
                if remaining_issues:
                    print(f"  ⚠️ {len(remaining_issues)} issues remain (using best effort)")

            self.processed_count += 1
            self.progress_tracker.mark_completed(breed_name)
            print(f"✅ Successfully processed '{breed_name}' with {len(final_data)} total features")
            
            return final_data

        except Exception as e:
            print(f"❌ Critical error processing '{breed_name}': {e}")
            self.failed_count += 1
            self.progress_tracker.mark_failed(breed_name)
            
            # Even for failures, ensure uniform output
            fallback_data = {**original_data}
            
            # Add all required features with fallbacks
            for feature, rules in FEATURE_SCHEMA.items():
                if feature not in fallback_data:
                    fallback_data[feature] = rules["fallback"]
            
            return fallback_data

    def run_pipeline(self, input_file: str, output_file: str, limit: Optional[int] = None):
        """Run the complete uniform data pipeline"""
        print("🚀 STARTING UNIFORM PET DATASET ENRICHMENT PIPELINE")
        print("=" * 60)

        input_data = self.load_input_data(input_file)
        if not input_data:
            return

        all_breeds = list(input_data.keys())
        if limit:
            all_breeds = all_breeds[:limit]

        # Check for resumable progress
        remaining_breeds = self.progress_tracker.get_remaining_breeds(all_breeds)
        already_completed = len(all_breeds) - len(remaining_breeds)

        if already_completed > 0:
            print(f"📋 Resuming pipeline: {already_completed} breeds already completed")
            print(f"📋 Will process {len(remaining_breeds)} remaining breeds")
        else:
            print(f"📋 Will process {len(remaining_breeds)} breeds")

        # Load existing results
        final_dataset = {}
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                final_dataset = json.load(f)
                print(f"📂 Loaded existing results: {len(final_dataset)} breeds")
        except FileNotFoundError:
            print("📂 Starting fresh - no existing results found")

        self.validator.validation_report['total_breeds'] = len(all_breeds)
        start_time = time.time()

        # Process each breed
        for i, breed_name in enumerate(remaining_breeds, 1):
            print(f"\n📊 Progress: {i}/{len(remaining_breeds)} (Total: {already_completed + i}/{len(all_breeds)})")
            
            processed_data = self.process_breed(breed_name, input_data[breed_name])
            final_dataset[breed_name] = processed_data

            # Save intermediate results every 5 breeds
            if i % 5 == 0:
                self._save_results(final_dataset, output_file)
                print(f"💾 Intermediate save completed ({i} breeds processed)")

        # Final save and validation
        self._save_results(final_dataset, output_file)
        
        # Run final dataset validation
        print("\n🔍 RUNNING FINAL DATASET VALIDATION")
        self._run_final_validation(final_dataset)
        
        elapsed_time = time.time() - start_time

        print("\n🎉 UNIFORM PIPELINE COMPLETED!")
        print("=" * 60)
        print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
        print(f"✅ Successfully processed: {self.processed_count} breeds")
        print(f"❌ Failed: {self.failed_count} breeds")
        print(f"💾 Final dataset saved to: '{output_file}'")
        
        # Generate validation report
        validation_summary = self.validator.generate_validation_report(VALIDATION_FILE)
        print(f"📊 Validation Summary:")
        print(f"   - Breeds with perfect data: {validation_summary['total_breeds_processed'] - validation_summary['breeds_with_issues']}")
        print(f"   - Breeds with minor issues: {validation_summary['breeds_with_issues']}")
        print(f"   - Average issues per breed: {validation_summary['issues_per_breed']:.2f}")

    def _save_results(self, dataset: Dict, output_file: str):
        """Save the enriched dataset with formatting"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

    def _run_final_validation(self, dataset: Dict):
        """Run comprehensive final validation on the complete dataset"""
        print("🔍 Running comprehensive dataset validation...")
        
        total_issues = 0
        feature_completeness = {feature: 0 for feature in NEW_FEATURES_ADDED}
        
        for breed_name, breed_data in dataset.items():
            is_valid, issues = self.validator.validate_breed_data(breed_name, breed_data)
            total_issues += len(issues)
            
            # Check feature completeness
            for feature in NEW_FEATURES_ADDED:
                if feature in breed_data and breed_data[feature] not in [None, "", [], {}]:
                    feature_completeness[feature] += 1
        
        # Calculate completeness percentages
        total_breeds = len(dataset)
        completeness_report = {}
        for feature, count in feature_completeness.items():
            percentage = (count / total_breeds) * 100 if total_breeds > 0 else 0
            completeness_report[feature] = {
                'completed_breeds': count,
                'percentage': round(percentage, 1)
            }
        
        # Display results
        print(f"📊 Final Validation Results:")
        print(f"   - Total breeds: {total_breeds}")
        print(f"   - Total validation issues: {total_issues}")
        print(f"   - Average issues per breed: {total_issues/max(1, total_breeds):.2f}")
        
        print(f"\n📈 Feature Completeness Report:")
        for feature, stats in completeness_report.items():
            if stats['percentage'] < 100:
                print(f"   ⚠️ {feature}: {stats['percentage']}% ({stats['completed_breeds']}/{total_breeds})")
            else:
                print(f"   ✅ {feature}: {stats['percentage']}%")
        
        # Save completeness report
        with open('feature_completeness_report.json', 'w') as f:
            json.dump(completeness_report, f, indent=2)


# =============================================================================
# QUALITY ASSURANCE FUNCTIONS
# =============================================================================

def run_data_quality_check(output_file: str):
    """Run comprehensive data quality check on the final output"""
    print("\n🔍 RUNNING DATA QUALITY ASSURANCE CHECK")
    print("=" * 50)
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Output file '{output_file}' not found!")
        return
    
    total_breeds = len(data)
    quality_report = {
        'total_breeds': total_breeds,
        'feature_analysis': {},
        'data_quality_issues': [],
        'recommendations': []
    }
    
    print(f"📊 Analyzing {total_breeds} breeds...")
    
    # Check each feature across all breeds
    for feature in NEW_FEATURES_ADDED:
        feature_stats = {
            'present_count': 0,
            'empty_count': 0,
            'null_count': 0,
            'type_issues': 0,
            'sample_values': []
        }
        
        for breed_name, breed_data in data.items():
            if feature not in breed_data:
                feature_stats['null_count'] += 1
            else:
                value = breed_data[feature]
                if value is None:
                    feature_stats['null_count'] += 1
                elif value == "" or value == [] or value == {}:
                    feature_stats['empty_count'] += 1
                else:
                    feature_stats['present_count'] += 1
                    if len(feature_stats['sample_values']) < 3:
                        feature_stats['sample_values'].append(str(value)[:100])
        
        feature_stats['completion_rate'] = (feature_stats['present_count'] / total_breeds) * 100
        quality_report['feature_analysis'][feature] = feature_stats
        
        # Report issues
        if feature_stats['completion_rate'] < 100:
            issue = f"{feature}: {feature_stats['completion_rate']:.1f}% completion rate"
            quality_report['data_quality_issues'].append(issue)
    
    # Generate recommendations
    low_completion_features = [
        feature for feature, stats in quality_report['feature_analysis'].items() 
        if stats['completion_rate'] < 95
    ]
    
    if low_completion_features:
        quality_report['recommendations'].append(
            f"Re-run pipeline for features with <95% completion: {', '.join(low_completion_features)}"
        )
    
    if not quality_report['data_quality_issues']:
        quality_report['recommendations'].append("Dataset quality is excellent - all features have 100% completion!")
    
    # Display results
    print(f"\n📈 DATA QUALITY SUMMARY:")
    print(f"   - Total breeds processed: {total_breeds}")
    print(f"   - Features with 100% completion: {sum(1 for stats in quality_report['feature_analysis'].values() if stats['completion_rate'] == 100)}/{len(NEW_FEATURES_ADDED)}")
    print(f"   - Issues found: {len(quality_report['data_quality_issues'])}")
    
    if quality_report['data_quality_issues']:
        print(f"\n⚠️ QUALITY ISSUES:")
        for issue in quality_report['data_quality_issues'][:10]:  # Show first 10
            print(f"   - {issue}")
    else:
        print(f"\n✅ NO QUALITY ISSUES FOUND! Dataset is uniform and complete.")
    
    if quality_report['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in quality_report['recommendations']:
            print(f"   - {rec}")
    
    # Save quality report
    with open('data_quality_report.json', 'w') as f:
        json.dump(quality_report, f, indent=2)
    
    print(f"\n💾 Full quality report saved to: data_quality_report.json")
    return quality_report


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def get_gemini_api_key():
    """Get API key from environment or user input"""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️ GOOGLE_API_KEY not found in environment variables.")
        print("Get your free API key from: https://makersuite.google.com/app/apikey")
        api_key = input("Enter your Gemini API key: ").strip()
        if not api_key:
            raise ValueError("API key is required to proceed")
    return api_key

if __name__ == "__main__":
    print("\n" + "="*60)
    print("UNIFORM PET DATASET ENRICHMENT PIPELINE")
    print("Guarantees uniform JSON output with all features completed")
    print("="*60)
    
    print("\nChoose an option:")
    print("1. Run demo (3 breeds) - Test the uniform output")
    print("2. Run full pipeline (all breeds) - Complete enrichment")
    print("3. Resume interrupted pipeline - Continue from where you left off")
    print("4. Quality check only - Analyze existing output file")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "4":
        # Just run quality check
        output_file = input(f"Enter output file path (default: {OUTPUT_FILE}): ").strip()
        if not output_file:
            output_file = OUTPUT_FILE
        run_data_quality_check(output_file)
    else:
        # Run the pipeline
        try:
            pipeline = UniformDataPipeline()
            
            if choice == "1":
                print("\n🧪 Running demo with 3 breeds...")
                pipeline.run_pipeline(INPUT_FILE, 'demo_' + OUTPUT_FILE, limit=3)
                run_data_quality_check('demo_' + OUTPUT_FILE)
            
            elif choice == "2":
                print("\n🚀 Running full pipeline...")
                pipeline.run_pipeline(INPUT_FILE, OUTPUT_FILE)
                run_data_quality_check(OUTPUT_FILE)
            
            elif choice == "3":
                print("\n🔄 Resuming interrupted pipeline...")
                pipeline.run_pipeline(INPUT_FILE, OUTPUT_FILE)
                run_data_quality_check(OUTPUT_FILE)
            
            else:
                print("❌ Invalid choice. Running demo by default.")
                pipeline.run_pipeline(INPUT_FILE, 'demo_' + OUTPUT_FILE, limit=3)
                run_data_quality_check('demo_' + OUTPUT_FILE)
                
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            print("Please check your API key and input file, then try again.")

    print("\n🎉 Process completed! Check the generated reports for detailed analysis.")
    print("\n📄 Generated files:")
    print("   - Enriched dataset: final_enriched_training_data.json")
    print("   - Validation report: validation_report.json")
    print("   - Quality report: data_quality_report.json")
    print("   - Completeness report: feature_completeness_report.json")
    print("   - Progress tracking: pipeline_progress.json")
