import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

endpoint = os.getenv("AZURE_DOC_INTEL_ENDPOINT")
key = os.getenv("AZURE_DOC_INTEL_KEY")

print("-" * 30)
print("DEBUGGING KEYS")
print("-" * 30)

if not endpoint:
    print("❌ ENDPOINT is Missing! Check .env file name and location.")
else:
    print(f"✅ Endpoint loaded: {endpoint}")

if not key:
    print("❌ KEY is Missing!")
else:
    # Print first 5 and last 5 chars to verify no quotes
    print(f"✅ Key loaded: {key[:5]}...{key[-5:]}")
    if '"' in key or "'" in key:
        print("⚠️  WARNING: Your key contains quotes! Remove them in .env")
    else:
        print("✅ Key format looks clean (no quotes detected).")

print("-" * 30)