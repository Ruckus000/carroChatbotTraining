import requests
import json
import sys

def test_api(text, url="http://localhost:8000"):
    """Test the NLU API with the given text."""
    endpoint = f"{url}/api/nlu"
    
    print(f"Testing with text: '{text}'")
    print(f"Sending request to: {endpoint}")
    
    try:
        response = requests.post(
            endpoint,
            json={"text": text},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\nAPI Response:")
            print(json.dumps(result, indent=2))
            
            print("\nDetected Intent:")
            print(f"  {result['intent']['name']} (confidence: {result['intent']['confidence']:.4f})")
            
            if result['entities']:
                print("\nDetected Entities:")
                for entity in result['entities']:
                    print(f"  {entity['entity']}: {entity['value']}")
            else:
                print("\nNo entities detected.")
                
            return result
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to the API at {url}")
        print("Make sure the API server is running and the URL is correct.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Use command line argument if provided, otherwise use a default test message
    test_text = sys.argv[1] if len(sys.argv) > 1 else "My 2022 Honda Civic needs a tow from 123 Main Street"
    test_api(test_text) 