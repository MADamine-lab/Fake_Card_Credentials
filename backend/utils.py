import secrets
import string
import hashlib
from datetime import datetime, timedelta
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def generate_auth_code(length=6):
    """
    Generate a random authentication code
    
    Args:
        length: Length of the code (default 6)
        
    Returns:
        String of random digits
    """
    return ''.join(secrets.choice(string.digits) for _ in range(length))

def send_auth_code_email(code, recipient_email, smtp_config):
    """
    Send authentication code via email
    
    Args:
        code: The authentication code to send
        recipient_email: Email address to send to
        smtp_config: Dictionary with SMTP configuration
        
    Returns:
        (success: bool, message: str)
    """
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = '🔒 Code d\'Authentification - Sensitive Data Protection'
        msg['From'] = smtp_config.get('sender_email', 'noreply@securedata.com')
        msg['To'] = recipient_email
        
        # Create HTML and plain text versions
        text_content = f"""
Code d'Authentification
========================

Un contenu sensible a été détecté dans votre message.

Votre code d'authentification est: {code}

Ce code expirera dans 5 minutes.

Si vous n'avez pas demandé ce code, veuillez ignorer cet email.

---
Sensitive Data Protection System
        """
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .container {{
            background: #f9f9f9;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px 10px 0 0;
            text-align: center;
            margin: -30px -30px 20px -30px;
        }}
        .code-box {{
            background: white;
            border: 2px solid #667eea;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            margin: 20px 0;
        }}
        .code {{
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
            letter-spacing: 8px;
            font-family: 'Courier New', monospace;
        }}
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .footer {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔒 Code d'Authentification</h1>
        </div>
        
        <p>Bonjour,</p>
        
        <p>Un <strong>contenu sensible</strong> a été détecté dans votre message (visage, carte bancaire ou carte d'identité).</p>
        
        <p>Pour des raisons de sécurité, veuillez entrer le code d'authentification ci-dessous pour confirmer l'envoi :</p>
        
        <div class="code-box">
            <p style="margin: 0; color: #666;">Votre code d'authentification</p>
            <div class="code">{code}</div>
        </div>
        
        <div class="warning">
            ⏱️ <strong>Important:</strong> Ce code expirera dans <strong>5 minutes</strong>.
        </div>
        
        <p style="color: #666; font-size: 14px;">Si vous n'avez pas demandé ce code, veuillez ignorer cet email.</p>
        
        <div class="footer">
            <p>Sensitive Data Protection System</p>
            <p>Ce message a été envoyé automatiquement, merci de ne pas y répondre.</p>
        </div>
    </div>
</body>
</html>
        """
        
        # Attach both versions
        part1 = MIMEText(text_content, 'plain', 'utf-8')
        part2 = MIMEText(html_content, 'html', 'utf-8')
        
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
            if smtp_config.get('use_tls', True):
                server.starttls()
            
            # Login if credentials provided
            if smtp_config.get('sender_password'):
                server.login(smtp_config['sender_email'], smtp_config['sender_password'])
            
            server.send_message(msg)
        
        return True, f"Code sent successfully to {recipient_email}"
        
    except smtplib.SMTPAuthenticationError:
        return False, "Email authentication failed. Please check your email credentials."
    except smtplib.SMTPException as e:
        return False, f"SMTP error: {str(e)}"
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"

def verify_auth_code(stored_code, provided_code):
    """
    Verify authentication code
    
    Args:
        stored_code: The code stored in database
        provided_code: Code provided by user
        
    Returns:
        Boolean indicating if codes match
    """
    return stored_code == provided_code

def hash_password(password):
    """
    Hash password using SHA-256
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    """
    Validate email format
    
    Args:
        email: Email address to validate
        
    Returns:
        Boolean indicating if email is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def is_code_expired(code_timestamp, expiry_minutes=5):
    """
    Check if authentication code has expired
    
    Args:
        code_timestamp: DateTime when code was generated
        expiry_minutes: Minutes until code expires
        
    Returns:
        Boolean indicating if code is expired
    """
    if isinstance(code_timestamp, str):
        code_timestamp = datetime.fromisoformat(code_timestamp)
    
    expiry_time = code_timestamp + timedelta(minutes=expiry_minutes)
    return datetime.now() > expiry_time

def sanitize_text(text):
    """
    Sanitize text input to prevent XSS attacks
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove script tags and content
    text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL)
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&']
    for char in dangerous_chars:
        text = text.replace(char, '')
    
    return text

def format_file_size(size_bytes):
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def validate_image_format(filename):
    """
    Validate image file format
    
    Args:
        filename: Name of the file
        
    Returns:
        Boolean indicating if format is valid
    """
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    extension = filename.lower()[filename.rfind('.'):]
    return extension in allowed_extensions

def rate_limit_check(user_id, max_requests=10, time_window_minutes=1):
    """
    Simple rate limiting check (in-memory)
    In production, use Redis or similar
    
    Args:
        user_id: User identifier
        max_requests: Maximum number of requests allowed
        time_window_minutes: Time window in minutes
        
    Returns:
        Boolean indicating if rate limit is exceeded
    """
    # This is a placeholder - implement with Redis in production
    return False

class AuthCodeManager:
    """
    Manager for authentication codes with in-memory storage
    In production, use Redis or database
    """
    
    def __init__(self, expiry_minutes=5):
        self.codes = {}
        self.expiry_minutes = expiry_minutes
    
    def generate_code(self, user_id, length=6):
        """Generate and store code for user"""
        code = generate_auth_code(length)
        self.codes[user_id] = {
            'code': code,
            'timestamp': datetime.now(),
            'attempts': 0
        }
        return code
    
    def verify_code(self, user_id, provided_code, max_attempts=3):
        """Verify code with attempt limiting"""
        if user_id not in self.codes:
            return False, "Code not found"
        
        code_data = self.codes[user_id]
        
        # Check expiry
        if is_code_expired(code_data['timestamp'], self.expiry_minutes):
            del self.codes[user_id]
            return False, "Code expired"
        
        # Check attempts
        if code_data['attempts'] >= max_attempts:
            del self.codes[user_id]
            return False, "Too many attempts"
        
        # Verify code
        if code_data['code'] == provided_code:
            del self.codes[user_id]
            return True, "Code verified"
        else:
            code_data['attempts'] += 1
            return False, f"Invalid code ({max_attempts - code_data['attempts']} attempts remaining)"
    
    def cleanup_expired(self):
        """Remove expired codes"""
        expired_users = []
        for user_id, code_data in self.codes.items():
            if is_code_expired(code_data['timestamp'], self.expiry_minutes):
                expired_users.append(user_id)
        
        for user_id in expired_users:
            del self.codes[user_id]
        
        return len(expired_users)

def log_security_event(event_type, user_id, details):
    """
    Log security-related events
    
    Args:
        event_type: Type of event (e.g., 'sensitive_content_detected')
        user_id: User identifier
        details: Additional details about the event
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'user_id': user_id,
        'details': details
    }
    
    # In production, write to database or logging service
    print(f"[SECURITY] {log_entry}")

if __name__ == "__main__":
    # Test utilities
    print("Testing Utilities...")
    
    # Test code generation
    code = generate_auth_code()
    print(f"Generated code: {code}")
    
    # Test code verification
    print(f"Code verification: {verify_auth_code(code, code)}")
    
    # Test email validation
    print(f"Valid email: {validate_email('test@example.com')}")
    print(f"Invalid email: {validate_email('invalid-email')}")
    
    # Test AuthCodeManager
    manager = AuthCodeManager()
    user_id = "user123"
    generated_code = manager.generate_code(user_id)
    print(f"Generated code for {user_id}: {generated_code}")
    
    success, message = manager.verify_code(user_id, generated_code)
    print(f"Verification result: {success}, {message}")