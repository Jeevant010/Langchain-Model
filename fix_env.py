
import os
import codecs

def fix_env():
    content = ""
    start_content = "GROQ_API_KEY=your_groq_api_key_here"
    
    if not os.path.exists(".env"):
        content = start_content
    else:
        # Try reading with different encodings
        encodings = ['utf-16', 'utf-16-le', 'utf-8', 'cp1252']
        success = False
        for enc in encodings:
            try:
                with open(".env", "r", encoding=enc) as f:
                    content = f.read()
                    # Check if it looks like an env file (has = sign)
                    if "=" in content:
                        success = True
                        break
            except Exception:
                continue
        
        if not success:
            print("Could not read .env, resetting to default")
            content = start_content

    # Write back as utf-8
    with open(".env", "w", encoding="utf-8") as f:
        f.write(content.strip())
    print("Fixed .env encoding")

if __name__ == "__main__":
    fix_env()
