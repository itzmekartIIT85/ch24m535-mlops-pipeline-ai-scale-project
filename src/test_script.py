import requests

url = "http://127.0.0.1:8000/predict"

# Example passenger (raw input, not encoded)
payload = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "Fare": 7.25,
    "Embarked": "S"
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
