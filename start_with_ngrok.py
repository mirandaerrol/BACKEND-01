"""
Start the Flask backend with optional ngrok tunnel.

Usage:
    python start_with_ngrok.py              # Start with ngrok tunnel
    python start_with_ngrok.py --no-ngrok   # Start without ngrok (local only)

Prerequisites:
    pip install pyngrok

Setup ngrok (one-time):
    1. Sign up at https://ngrok.com (free tier available)
    2. Get your auth token from https://dashboard.ngrok.com/get-started/your-authtoken
    3. Set it in your .env file: NGROK_AUTH_TOKEN=your_token_here
    
After starting, the script will print the public ngrok URL.
Copy this URL and set it as VITE_DETECTION_BACKEND_URL in your Laravel .env on Railway.
"""

import os
import sys
import subprocess
import time
from dotenv import load_dotenv

load_dotenv()

def start_with_ngrok():
    """Start Flask app and create ngrok tunnel."""
    try:
        from pyngrok import ngrok, conf
    except ImportError:
        print("=" * 60)
        print("ERROR: pyngrok is not installed.")
        print("Install it with: pip install pyngrok")
        print("=" * 60)
        sys.exit(1)

    # Configure ngrok auth token
    auth_token = os.environ.get("NGROK_AUTH_TOKEN")
    if auth_token:
        conf.get_default().auth_token = auth_token
    else:
        print("=" * 60)
        print("WARNING: NGROK_AUTH_TOKEN not set in .env")
        print("You can still use ngrok, but sessions will be limited.")
        print("Get your free token at: https://dashboard.ngrok.com/get-started/your-authtoken")
        print("=" * 60)

    port = int(os.environ.get("PORT", 5000))

    # Start ngrok tunnel
    print(f"\nStarting ngrok tunnel on port {port}...")
    public_url = ngrok.connect(port, "http")
    
    print("\n" + "=" * 60)
    print(f"  NGROK TUNNEL ACTIVE")
    print(f"  Public URL: {public_url}")
    print(f"  Local URL:  http://127.0.0.1:{port}")
    print(f"")
    print(f"  IMPORTANT: Update your Laravel .env on Railway:")
    print(f"  VITE_DETECTION_BACKEND_URL={public_url}")
    print(f"")
    print(f"  Then clear Laravel config cache:")
    print(f"  php artisan config:clear")
    print("=" * 60 + "\n")

    # Start the Flask app
    os.environ["NGROK_URL"] = str(public_url)
    
    # Import and run the Flask app
    from app import app
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


def start_local_only():
    """Start Flask app without ngrok (local network only)."""
    port = int(os.environ.get("PORT", 5000))
    print(f"\nStarting Flask backend on port {port} (local only)...")
    print(f"Access via: http://127.0.0.1:{port}")
    print(f"Or via local network IP: http://<your-ip>:{port}\n")
    
    from app import app
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    if "--no-ngrok" in sys.argv:
        start_local_only()
    else:
        start_with_ngrok()
