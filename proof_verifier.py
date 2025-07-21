import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\Yousuf Yasser Rabie\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import pipeline, AutoTokenizer, AutoModel
import re
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import base64
import io 

class ProofVerifier:
    """
    AI-powered proof verification system for student activities.
    Handles both text extraction from certificates and image analysis for activity photos.
    """
    
    def __init__(self):
        self.setup_logging()
        self.initialize_models()
        self.define_keywords()
        
    def setup_logging(self):
        """Setup logging for verification process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('proof_verification.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_models(self):
        """Initialize AI models for text and image analysis"""
        try:
            # Initialize CLIP model for image-text matching
            self.clip_model = pipeline("zero-shot-image-classification", 
                                     model="openai/clip-vit-base-patch32")
            
            # Initialize text classifier for document analysis
            self.text_classifier = pipeline("zero-shot-classification",
                                           model="facebook/bart-large-mnli")
            
            # Activity detection model (image classification)
            self.activity_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
            self.logger.info("Models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def define_keywords(self):
        """Define keywords for different activity categories"""
        self.category_keywords = {
            'courses': [
                'certificate', 'certification', 'awarded', 'completed', 'achievement',
                'course', 'completion', 'successfully', 'graduated', 'diploma',
                'program', 'training', 'education', 'learning', 'study', 'bootcamp'
            ],
            
            'workshops': [
                'certificate', 'certification', 'awarded', 'completed', 'achievement',
                'workshop', 'training', 'session', 'participant', 'attendance',
                'conducted by', 'organized by', 'facilitated by', 'work id',
                'employee id', 'staff id', 'project', 'team member', 'role',
                'position', 'department', 'seminar', 'tutorial', 'hands-on', 'bootcamp'
            ],
            
            'student_activities_aisec': [
                'AISEC','Enactus','IEEE','member', 'participant', 'team', 'work id','society'
                'project', 'initiative', 'event', 'activity','leadership',
                'student id', 'position', 'role', 'committee', 'club', 
                'student organization', 
            ],
            
            'other_student_activities': [
                'tree', 'msp', 'student partner', 'member', 'participant',
                'team', 'project', 'initiative', 'event', 'activity', 'work id',
                'student id', 'position', 'role', 'committee', 'organization'
            ],
            
            'volunteering_cop': [
                'volunteer', 'volunteering', 'community service', 'contribution',
                'appreciation', 'recognition', 'thank you', 'project', 'initiative',
                'team', 'role', 'position', 'impact', 'community', 'cop',
                'social responsibility', 'service'
            ],
            
            'volunteering_icareer': [
                'volunteer', 'volunteering', 'icareer', 'career', 'mentorship',
                'guidance', 'support', 'contribution', 'appreciation', 'recognition',
                'team', 'role', 'position', 'impact', 'community'
            ],
            
            'youtube': [
                'youtube', 'channel', 'subscriber', 'views', 'content creator',
                'video', 'upload', 'monetization', 'analytics', 'creator',
                'influencer', 'social media'
            ],
            
            'internship_cib': [
                'internship', 'intern', 'cib', 'training period', 'work experience',
                'company', 'organization', 'department', 'work id', 'employee id',
                'project', 'team', 'role', 'position', 'supervisor', 'mentor','bank'
            ],
            
            'real_internship': [
                'internship', 'intern', 'dsquares', 'training period', 'work experience',
                'company', 'organization', 'department', 'work id', 'employee id',
                'project', 'team', 'role', 'position', 'supervisor', 'mentor',
                'employment', 'professional'
            ],
            
            'sports': [
                'sports', 'team', 'player', 'athlete', 'training', 'practice',
                'competition', 'match', 'tournament', 'coach', 'uniform',
                'equipment', 'facility', 'stadium', 'field', 'court',
                'fitness', 'exercise', 'physical activity'
            ],
            
            'social': [
                'social', 'event', 'gathering', 'meeting', 'community',
                'networking', 'celebration', 'party', 'cultural', 'festival',
                'social activity', 'interaction', 'people'
            ],
            
            'arts': [
                'art', 'exhibition', 'performance', 'show', 'display',
                'gallery', 'studio', 'work', 'piece', 'creation', 'artist',
                'exhibit', 'display', 'venue', 'stage', 'creative',
                'painting', 'music', 'theater', 'dance'
            ],
            
            'time_spent': [
                'quran', 'religious', 'prayer', 'meditation','worship',
                'study', 'learning', 'reading', 'contemplation', 
                'islamic', 'faith', 'devotion'
            ]
        }
        
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR with enhanced preprocessing"""
        try:
            # Read the image
            image_cv = cv2.imread(image_path)
            if image_cv is None:
                self.logger.error("Failed to read image")
                return ""

            # Convert to grayscale
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply multiple preprocessing techniques
            # 1. Basic denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # 2. Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 3. Dilation to make text more prominent
            kernel = np.ones((1,1), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            
            # 4. Additional sharpening
            kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(dilated, -1, kernel_sharp)
            
            # Try multiple OCR configurations and combine results
            texts = []
            
            # Config 1: Standard
            text1 = pytesseract.image_to_string(sharpened, config='--psm 6')
            texts.append(text1)
            
            # Config 2: With orientation and script detection
            text2 = pytesseract.image_to_string(sharpened, config='--psm 3')
            texts.append(text2)
            
            # Config 3: Treat as single uniform block of text
            text3 = pytesseract.image_to_string(sharpened, config='--psm 4')
            texts.append(text3)
            
            # Combine all extracted texts
            combined_text = ' '.join(texts)
            
            # Clean up the text
            # Remove extra whitespace 
            cleaned_text = ' '.join(combined_text.split())
            
            self.logger.info(f"Extracted text length: {len(cleaned_text)} characters")
            self.logger.info(f"Full extracted text: {cleaned_text}")
            
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Error extracting text from image: {str(e)}")
            return ""
        
    def verify_text_keywords(self, text: str, category: str, activity_title: str) -> Dict:
        """Verify if text contains relevant keywords for the category"""
        keywords = self.category_keywords.get(category, [])
        text_lower = text.lower()
        
        found_keywords = []
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        # Calculate relevance score
        relevance_score = min(len(found_keywords) * 0.3, 1.0) if keywords else 0
        
        # Use zero-shot classification for additional verification
        try:
            classification_labels = [
                f"{category} and {activity_title} related document",
                f"certificate or award of {activity_title} and {category}",
                f"{category} and {activity_title} official document",
                "irrelevant document",
                "fake documents",
                "irrelevant certificate or award"
            ]
            
            classification_result = self.text_classifier(text, classification_labels)
            top_label = classification_result['labels'][0]
            top_score = classification_result['scores'][0]
            
        except:
            top_label = "unknown"
            top_score = 0.0
        
        return {
            'found_keywords': found_keywords,
            'keyword_count': len(found_keywords),
            'relevance_score': relevance_score,
            'classification': top_label,
            'classification_score': top_score,
            'is_relevant': relevance_score > 0.5 or (top_score > 0.5 and category in top_label)
        }
    
    def _check_activity_name_in_text(self, text: str, activity_title: str) -> Dict:
        """
        Check if activity name (course/workshop) appears in the certificate text.
        Returns a dict with match status and matched words.
        """
        if not activity_title or not text:
            return {'found': False, 'matched_words': [], 'score': 0.0}
            
        # Convert both to lower case for case-insensitive matching
        text = text.lower()
        activity_words = [word.lower() for word in activity_title.split() 
                         if len(word) > 2]  # Ignore very short words
        
        # Remove common words that might cause false positives
        stop_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        activity_words = [word for word in activity_words if word not in stop_words]
        
        # Find matching words
        matched_words = [word for word in activity_words if word in text]
        
        # Calculate match score based on number of matched words
        score = len(matched_words) / len(activity_words) if activity_words else 0.0
        
        return {
            'found': len(matched_words) > 0,
            'matched_words': matched_words,
            'score': score
        }
    
    def _check_name_in_text(self, text: str, full_name: str) -> Dict:
        """
        Check if user's name appears in the certificate text with improved matching.
        """
        self.logger.info(f"Checking name in text. Name to check: '{full_name}'")
        
        if not full_name or not text:
            self.logger.warning(f"Missing input - Name: '{full_name}', Text empty: {not text}")
            return {'found': False, 'matched_parts': [], 'score': 0.0}
            
        # Convert both to lower case and normalize spaces
        text = ' '.join(text.lower().split())
        full_name = ' '.join(full_name.lower().split())
        
        # Split name into parts (first name, middle name, last name)
        name_parts = [part for part in full_name.split() if len(part) > 1]  # Ignore initials
        
        self.logger.info(f"Name parts to check: {name_parts}")
        
        # Common name prefixes/suffixes to ignore
        ignore_parts = {'mr', 'mrs', 'ms', 'dr', 'prof', 'jr', 'sr', 'ii', 'iii', 'iv'}
        name_parts = [part for part in name_parts if part not in ignore_parts]
        
        # Find matching name parts with fuzzy matching
        matched_parts = []
        for part in name_parts:
            # Try exact match first
            if part in text:
                matched_parts.append(part)
                self.logger.info(f"Found exact name part in text: {part}")
                continue
            
        
        # Calculate score based on matched parts
        score = 0.0
        if matched_parts:
            # Try to find full name with flexible spacing
            full_name_parts = ' '.join(name_parts)
            if full_name_parts in text or any(
                full_name_parts.replace(' ', sep) in text 
                for sep in ['.', '-', '_', '']
            ):
                score = 1.0
                self.logger.info("Found full name match!")
            elif name_parts[0] in matched_parts:  # First name match
                score = 0.7
                self.logger.info("Found first name match")
            else:  # Partial matches
                score = len(matched_parts) * 0.3
                self.logger.info(f"Found partial matches, score: {score}")
            score = min(score, 1.0)  # Cap at 1.0
        
        result = {
            'found': len(matched_parts) > 0,
            'matched_parts': matched_parts,
            'score': score,
            'first_name_found': name_parts[0] in matched_parts if name_parts else False
        }
        
        self.logger.info(f"Name check result: {result}")
        return result
    
    def _is_proof_valid_for_time(self, verification_result: Dict) -> Dict:
        """
        Check if a proof is valid for time counting based on verification status,
        name matching, and activity title matching.
        Returns a dict with validity status and reason.
        """
        # Default response
        response = {
            'is_valid': False,
            'reason': None,
            'confidence_percentage': 0
        }

        # Get verification details
        details = verification_result.get('details', {})
        name_check = details.get('name_check', {})
        activity_check = details.get('activity_name_check', {})
        
        # Calculate overall confidence percentage
        confidence_factors = []
        
        # Base verification confidence
        if verification_result['verification_status'] == 'verified':
            confidence_factors.append(1.0)
        elif verification_result['verification_status'] == 'partially_verified':
            confidence_factors.append(0.5)
        else:
            response['reason'] = "Proof not verified"
            return response
        
        # Name check confidence
        if name_check:
            name_score = name_check.get('score', 0)
            confidence_factors.append(name_score)
            if not name_check.get('found', False):
                response['reason'] = "Name not found in certificate"
                return response
        
        # Activity title check confidence (for courses and workshops)
        if activity_check:
            activity_score = activity_check.get('score', 0)
            confidence_factors.append(activity_score)
            if not activity_check.get('found', False):
                response['reason'] = "Activity title not found in certificate"
                return response
        
        # Calculate overall confidence percentage
        if confidence_factors:
            confidence_percentage = (sum(confidence_factors) / len(confidence_factors)) * 100
        else:
            confidence_percentage = verification_result['confidence_score'] * 100
        
        # Proof is valid only if confidence percentage is above threshold
        if confidence_percentage >= 50:  # 50% minimum threshold
            response['is_valid'] = True
            response['confidence_percentage'] = confidence_percentage
        else:
            response['reason'] = f"Low confidence score: {confidence_percentage:.1f}%"
        
        return response

    def is_document_proof(self, image_path: str) -> bool:
        """
        Determine if the proof is a document/certificate or an activity photo.
        Uses multiple checks to accurately distinguish between documents and activity photos.
        
        Returns:
            bool: True if it's a document (certificate, letter), False if it's an activity photo
        """
        try:
            image = Image.open(image_path)
            
            # 1. Document Structure Check
            # Check for formal document characteristics
            document_structure = self.clip_model(
                image,
                candidate_labels=[
                    "formal document with text layout and borders",
                    "certificate with official letterhead and signature",
                    "formal letter or document with structured layout",
                    "photo of people in action or activity",
                ]
            )
            
            # Strong indication of document
            if document_structure[0]['score'] > 0.6 and "photo" in document_structure[0]['label'].lower():
                self.logger.info("Document detected by structure analysis")
                return False
            
            # 2. Content Type Analysis
            content_type = self.clip_model(
                image,
                candidate_labels=[
                    "certificate or award document",
                    "official letter or document",
                    "people actively participating in activity",
                    "live event or action photo",
                    "sports or physical activity photo"
                ]
            )
            
            # Clear activity photo
            if content_type[0]['score'] > 0.6 and ("activity" in content_type[0]['label'].lower() or 
                                                  "photo" in content_type[0]['label'].lower()):
                self.logger.info("Activity photo detected by content analysis")
                return False
            
            # 3. Text Density Check
            # Extract and analyze text
            extracted_text = self.extract_text_from_image(image_path)
            words = extracted_text.split()
            
            # Documents typically have more text
            if len(words) > 30:  # Threshold for minimum words in a document
                self.logger.info("Document detected by text density")
                return True
            
            # 4. Document Keywords Check
            document_keywords = {
                'certificate', 'certify', 'awarded', 'completed', 'achievement', 
                'completion', 'presented', 'hereby', 'recognition', 'award',
                'dear', 'sincerely', 'signature', 'authorized', 'date'
            }
            
            text_lower = extracted_text.lower()
            keyword_matches = sum(1 for keyword in document_keywords if keyword in text_lower)
            
            if keyword_matches >= 3:  # At least 3 document keywords found
                self.logger.info("Document detected by keyword analysis")
                return True
            
           
            # 5. Category-based Default
            if self._current_category in ['courses', 'workshops', 'internship_cib', 'real_internship']:
                self.logger.info(f"Defaulting to document based on category: {self._current_category}")
                return True
                # These categories typically use activity photos
            elif self._current_category in ['sports', 'social', 'arts']:
                self.logger.info(f"Defaulting to activity photo based on category: {self._current_category}")
                return False
            
            # Default case: if we're unsure, base it on text presence
            has_significant_text = len(words) > 15
            self.logger.info(f"Defaulting based on text presence: {'document' if has_significant_text else 'activity photo'}")
            return has_significant_text
            
        except Exception as e:
            self.logger.error(f"Error in document detection: {str(e)}")
            # Default to document for document-heavy categories
            if hasattr(self, '_current_category') and self._current_category in ['courses', 'workshops', 'internship_cib', 'real_internship']:
                return True
            return False

    def verify_activity_photo(self, image_path: str, category: str, activity_title: str) -> Dict:
        """
        Verify an activity photo by checking if the image content matches the claimed activity.
        """
        verification_result = {
            'verification_status': 'unverified',
            'confidence_score': 0.0,
            'details': {},
            'valid_for_time': False,
            'recommendations': [],
            'time_validation': None  # Initialize time_validation
        }

        try:
            # Load image
            image = Image.open(image_path)
            
            # Basic activity labels for verification
            activity_labels = [
                f"{activity_title.lower()}",
                f"person doing {activity_title.lower()}",
                "unrelated activity"
            ]

            # Check if image matches the activity
            activity_results = self.clip_model(image, candidate_labels=activity_labels)
            
            # Get activity match confidence
            activity_match = activity_results[0]
            activity_confidence = activity_match['score'] if activity_match['label'] != "unrelated activity" else 0.0
            verification_result['confidence_score'] = activity_confidence

            # Store detection details
            verification_result['details'] = {
                'activity_match': {
                    'detected_activity': activity_match['label'],
                    'confidence': activity_confidence
                }
            }

            # Set verification status and recommendations
            if verification_result['confidence_score'] >= 0.6:
                verification_result['verification_status'] = 'verified'
                verification_result['valid_for_time'] = True
                verification_result['time_validation'] = {
                    'is_valid': True,
                    'confidence_percentage': verification_result['confidence_score'] * 100
                }
            else:
                verification_result['time_validation'] = {
                    'is_valid': False,
                    'confidence_percentage': verification_result['confidence_score'] * 100
                }
                if activity_confidence < 0.5:
                    verification_result['recommendations'].append(
                        f"The image should clearly show {activity_title}"
                    )
                

            return verification_result

        except Exception as e:
            self.logger.error(f"Error verifying activity photo: {str(e)}")
            verification_result['verification_status'] = 'error'
            verification_result['error'] = str(e)
            verification_result['time_validation'] = {
                'is_valid': False,
                'confidence_percentage': 0
            }
            return verification_result

    def verify_proof(self, image_path: str, category: str, activity_title: str = "", user_name: str = "") -> Dict:
        """
        Main verification function that handles both document and activity photo verification
        """
        self.logger.info(f"Starting verification for category: {category}, user: '{user_name}', activity: '{activity_title}'")
        
        # Store category for document type detection
        self._current_category = category
        
        verification_result = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'category': category,
            'activity_title': activity_title,
            'user_name': user_name,
            'verification_status': 'unverified',  # Start as unverified
            'confidence_score': 0.0,
            'details': {},
            'valid_for_time': False,
            'time_validation': None,
            'recommendations': [],
            'verification_report': {}  # New detailed report section
        }
        
        try:
            # First, determine if this is a document or activity photo
            is_document = self.is_document_proof(image_path)
            verification_result['details']['proof_type'] = 'document' if is_document else 'activity_photo'
            verification_result['verification_report']['proof_type'] = 'Document' if is_document else 'Activity Photo'

            # Handle activity photos differently from documents
            if not is_document:
                activity_result = self.verify_activity_photo(image_path, category, activity_title)
                verification_result.update(activity_result)
                return verification_result

            # For documents, proceed with strict verification
            # Extract text from image
            extracted_text = self.extract_text_from_image(image_path)
            verification_result['details']['extracted_text'] = extracted_text[:500]
            
            # Initialize verification flags
            name_verified = False
            activity_verified = False
            
            # Verify text content
            text_verification = self.verify_text_keywords(extracted_text, category, activity_title)
            verification_result['details']['text_analysis'] = text_verification
            
            # Add text analysis to report
            verification_result['verification_report']['text_analysis'] = {
                'keywords_found': len(text_verification['found_keywords']),
                'found_keywords': text_verification['found_keywords'],
                'relevance_score': text_verification['relevance_score']
            }
            
            # Strict activity name verification for documents
            activity_check_result = {'found': False, 'score': 0.0, 'matched_words': []}
            if activity_title:
                activity_check_result = self._check_activity_name_in_text(extracted_text, activity_title)
                verification_result['details']['activity_name_check'] = activity_check_result
                
                # For courses and workshops, require activity title match
                if category in ['courses', 'workshops']:
                    if not activity_check_result['found'] or activity_check_result['score'] < 0.7:  # Increased threshold
                        verification_result['recommendations'].append(
                            f"The {category.rstrip('s')} name '{activity_title}' was not found in the certificate. "
                            "Make sure it matches exactly as shown on the certificate."
                        )
                    else:
                        activity_verified = True
                else:
                    activity_verified = True  # Less strict for other categories
            else:
                activity_verified = True  # No activity title provided
            
            # Add activity verification to report
            verification_result['verification_report']['activity_verification'] = {
                'activity_title': activity_title,
                'found': activity_check_result['found'],
                'match_score': activity_check_result['score'],
                'matched_words': activity_check_result['matched_words'],
                'verified': activity_verified
            }
            
            # Strict user name verification for documents
            name_check_result = {'found': False, 'score': 0.0, 'matched_parts': []}
            if user_name:
                name_check_result = self._check_name_in_text(extracted_text, user_name)
                verification_result['details']['name_check'] = name_check_result
                
                if not name_check_result['found'] or name_check_result['score'] < 0.7:  # Increased threshold
                    verification_result['recommendations'].append(
                        f"Your name '{user_name}' was not found in the certificate. "
                        "Make sure your name matches exactly as shown on the certificate."
                    )
                else:
                    name_verified = True
            else:
                name_verified = True  # No user name provided
            
            # Add name verification to report
            verification_result['verification_report']['name_verification'] = {
                'name': user_name,
                'found': name_check_result['found'],
                'match_score': name_check_result['score'],
                'matched_parts': name_check_result['matched_parts'],
                'verified': name_verified
            }
            
            # Calculate confidence score with stricter weights
            confidence_factors = []
            
            # Text relevance (30%)
            confidence_factors.append(text_verification['relevance_score'] * 0.3)
            
            # Name verification (35%)
            name_score = name_check_result['score'] if user_name else 1.0
            confidence_factors.append(name_score * 0.35)
            
            # Activity verification (35%)
            activity_score = activity_check_result['score'] if activity_title else 1.0
            confidence_factors.append(activity_score * 0.35)
            
            verification_result['confidence_score'] = sum(confidence_factors)
            
            # Determine verification status - require both name and activity verification
            # and high confidence score
            if name_verified and activity_verified and verification_result['confidence_score'] >= 0.7:
                verification_result['verification_status'] = 'verified'
            elif name_verified and activity_verified and verification_result['confidence_score'] >= 0.5:
                verification_result['verification_status'] = 'partially_verified'
            else:
                verification_result['verification_status'] = 'unverified'
            
            # Time validation with strict requirements
            verification_result['valid_for_time'] = (
                verification_result['verification_status'] == 'verified' and 
                name_verified and 
                activity_verified and
                verification_result['confidence_score'] >= 0.7
            )
            
            verification_result['time_validation'] = {
                'is_valid': verification_result['valid_for_time'],
                'confidence_percentage': verification_result['confidence_score'] * 100,
                'reason': self._get_validation_reason(name_verified, activity_verified, 
                                                    verification_result['confidence_score'])
            }
            
            # Add overall status to report
            verification_result['verification_report']['overall_status'] = {
                'status': verification_result['verification_status'],
                'confidence_score': verification_result['confidence_score'],
                'valid_for_time': verification_result['valid_for_time'],
                'confidence_percentage': verification_result['confidence_score'] * 100
            }
            
            return verification_result
            
        except Exception as e:
            self.logger.error(f"Error during verification: {str(e)}")
            verification_result['verification_status'] = 'error'
            verification_result['error'] = str(e)
            return verification_result

    def _get_validation_reason(self, name_verified: bool, activity_verified: bool, 
                             confidence_score: float) -> str:
        """Generate detailed reason for time validation result"""
        if not name_verified:
            return "Name verification failed - name must match exactly as shown on certificate"
        if not activity_verified:
            return "Activity verification failed - activity title must match certificate"
        if confidence_score < 0.7:
            return f"Overall confidence score too low ({confidence_score:.1%})"
        return "All verification checks passed"


