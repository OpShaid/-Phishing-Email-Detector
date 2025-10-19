#!/usr/bin/env python3


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import re
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhishingDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,
            stop_words='english'
        )
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        
    def extract_features(self, email_text: str) -> Dict[str, Any]:
        """Extract security-relevant features from email"""
        features = {}
        
        # URL count
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_text)
        features['url_count'] = len(urls)
        
        # Suspicious URL patterns
        features['has_ip_url'] = any(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) for url in urls)
        features['has_suspicious_tld'] = any(url.endswith(('.tk', '.ml', '.ga', '.cf', '.gq')) for url in urls)
        
        # Urgency keywords
        urgency_keywords = ['urgent', 'verify', 'suspended', 'locked', 'act now', 'confirm', 'expire']
        features['urgency_score'] = sum(1 for keyword in urgency_keywords if keyword in email_text.lower())
        
        # Financial keywords
        financial_keywords = ['bank', 'account', 'password', 'credit card', 'ssn', 'social security']
        features['financial_score'] = sum(1 for keyword in financial_keywords if keyword in email_text.lower())
        
        # Length features
        features['email_length'] = len(email_text)
        features['num_words'] = len(email_text.split())
        
        # Special characters
        features['special_char_ratio'] = len(re.findall(r'[!@#$%^&*()_+=\[\]{};:\'",.<>?/\\|`~]', email_text)) / len(email_text) if len(email_text) > 0 else 0
        
        return features
    
    def train(self, emails: pd.DataFrame):
        """
        Train the phishing detector
        
        Args:
            emails: DataFrame with columns ['text', 'label']
                    label: 0 = legitimate, 1 = phishing
        """
        logger.info(f"Training on {len(emails)} emails...")
        
        # Extract text features (TF-IDF)
        X_text = self.vectorizer.fit_transform(emails['text'])
        
        # Extract additional features
        additional_features = emails['text'].apply(self.extract_features)
        additional_features_df = pd.DataFrame(additional_features.tolist())
        
        # Combine features
        from scipy.sparse import hstack
        X = hstack([X_text, additional_features_df.values])
        y = emails['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        logger.info("Training Random Forest...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"✅ Training complete!")
        logger.info(f"📊 Accuracy: {accuracy*100:.2f}%")
        logger.info("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
        
        logger.info("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def predict(self, email_text: str) -> Dict[str, Any]:
        """
        Predict if email is phishing
        
        Returns:
            {
                'is_phishing': bool,
                'confidence': float,
                'risk_score': int (0-100),
                'features': dict
            }
        """
        # Vectorize text
        X_text = self.vectorizer.transform([email_text])
        
        # Extract additional features
        additional_features = self.extract_features(email_text)
        additional_features_array = np.array(list(additional_features.values())).reshape(1, -1)
        
        # Combine
        from scipy.sparse import hstack
        X = hstack([X_text, additional_features_array])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = probabilities[prediction]
        
        # Risk score (0-100)
        risk_score = int(probabilities[1] * 100)  # Probability of phishing
        
        return {
            'is_phishing': bool(prediction),
            'confidence': float(confidence),
            'risk_score': risk_score,
            'phishing_probability': float(probabilities[1]),
            'features': additional_features,
            'verdict': 'PHISHING' if prediction == 1 else 'LEGITIMATE'
        }
    
    def save_model(self, path: str = 'phishing_detector.pkl'):
        """Save trained model"""
        joblib.dump({
            'vectorizer': self.vectorizer,
            'model': self.model
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = 'phishing_detector.pkl'):
        """Load trained model"""
        data = joblib.load(path)
        self.vectorizer = data['vectorizer']
        self.model = data['model']
        logger.info(f"Model loaded from {path}")


def create_sample_dataset() -> pd.DataFrame:
    """Create sample dataset for demonstration"""
    
    # Sample phishing emails
    phishing = [
        "URGENT: Your account has been suspended. Click here to verify: http://192.168.1.1/verify",
        "Your bank account will be locked in 24 hours. Confirm your identity now: http://suspicious.tk",
        "You've won $1,000,000! Claim now by providing your SSN and credit card",
        "ACT NOW: Your password has expired. Reset immediately: http://phishing.ml/reset",
        "Verify your PayPal account or it will be terminated: http://paypal-verify.com",
        "IRS Notice: You owe $5000. Pay immediately to avoid arrest: http://irs-payment.ga",
        "Your package is held at customs. Pay fees now: http://192.168.0.1/customs",
        "Confirm your email or lose access: http://email-confirm.cf/verify",
        "Security alert: Unusual activity detected. Click to secure: http://security.gq",
        "Your credit card has been charged $500. Dispute now: http://dispute-charge.tk"
    ]
    
    # Sample legitimate emails
    legitimate = [
        "Hi team, please review the quarterly report attached. Let me know if you have questions.",
        "Meeting reminder: Project sync tomorrow at 2 PM in conference room B.",
        "Thanks for your purchase! Your order #12345 will arrive in 3-5 business days.",
        "Welcome to our newsletter! Here are this week's top articles about technology.",
        "Your appointment is confirmed for Monday, March 15th at 10:00 AM.",
        "Invoice #5678 for professional services is attached. Payment due in 30 days.",
        "Don't forget: Team lunch on Friday at 12:30 PM. RSVP if you're joining.",
        "New feature release: Check out our updated dashboard with analytics.",
        "Thank you for subscribing. Here's a 10% discount code for your next purchase: WELCOME10",
        "Course reminder: Module 3 is now available. Complete by end of week."
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': phishing + legitimate,
        'label': [1]*len(phishing) + [0]*len(legitimate)
    })
    
    return df


def main():
    """Demo the phishing detector"""
    
    print("🎣 Phishing Email Detector - ML Security Tool")
    print("=" * 60)
    
    # Create sample dataset (in production, use real dataset)
    print("\n📚 Creating sample dataset...")
    df = create_sample_dataset()
    print(f"Dataset: {len(df)} emails ({(df['label']==1).sum()} phishing, {(df['label']==0).sum()} legitimate)")
    
    # Train model
    print("\n🔧 Training model...")
    detector = PhishingDetector()
    accuracy = detector.train(df)
    
    # Save model
    detector.save_model()
    
    # Test on new emails
    print("\n🧪 Testing on new emails...")
    print("=" * 60)
    
    test_emails = [
        {
            "text": "URGENT: Your Netflix account will be cancelled. Update payment: http://netflix-billing.tk",
            "expected": "PHISHING"
        },
        {
            "text": "Hi John, can you send me the presentation slides from yesterday's meeting? Thanks!",
            "expected": "LEGITIMATE"
        },
        {
            "text": "Confirm your Amazon order by clicking here: http://192.168.1.50/amazon",
            "expected": "PHISHING"
        }
    ]
    
    for i, email in enumerate(test_emails, 1):
        print(f"\n📧 Test Email {i}:")
        print(f"Text: {email['text'][:80]}...")
        print(f"Expected: {email['expected']}")
        
        result = detector.predict(email['text'])
        
        print(f"\n🔍 Prediction:")
        print(f"   Verdict: {result['verdict']}")
        print(f"   Risk Score: {result['risk_score']}/100")
        print(f"   Confidence: {result['confidence']*100:.1f}%")
        print(f"   Phishing Probability: {result['phishing_probability']*100:.1f}%")
        print(f"\n   Key Features:")
        for key, value in result['features'].items():
            print(f"      {key}: {value}")
        
        # Check if correct
        is_correct = (result['verdict'] == email['expected'])
        print(f"\n   {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        print("-" * 60)
    
    print("\n🎉 Demo complete!")
    print("\n📊 Model Performance:")
    print(f"   Accuracy: {accuracy*100:.1f}%")
    print(f"   Model: Random Forest (100 trees)")
    print(f"   Features: TF-IDF + 8 security features")
    print(f"   Training samples: {len(df)}")
    
    print("\n💡 Production Usage:")
    print("   detector = PhishingDetector()")
    print("   detector.load_model('phishing_detector.pkl')")
    print("   result = detector.predict(email_text)")
    print("   if result['is_phishing']:")
    print("       # Block or quarantine email")


if __name__ == "__main__":
    main()
    
    
    

        

    
