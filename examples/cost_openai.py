import requests
import datetime

API_KEY = "your-openai-api-key"
headers = {"Authorization": f"Bearer {API_KEY}"}

# 1. Get the total budget allocation limit
subscription_url = "https://api.openai.com/dashboard/billing/subscription"
sub_response = requests.get(subscription_url, headers=headers).json()
hard_limit = sub_response.get("hard_limit_usd", 0)

# 2. Get usage for the current billing period (e.g., last 30 days)
start_date = (datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
end_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

usage_url = f"https://api.openai.com/dashboard/billing/usage?start_date={start_date}&end_date={end_date}"
usage_response = requests.get(usage_url, headers=headers).json()
total_usage_cents = usage_response.get("total_usage", 0)  # OpenAI often returns this in cents
total_usage = total_usage_cents / 100

# 3. Calculate remaining balance
remaining_balance = hard_limit - total_usage

print(f"Total Limit: ${hard_limit:.2f}")
print(f"Usage (Last 30 Days): ${total_usage:.2f}")
print(f"Estimated Remaining Balance: ${remaining_balance:.2f}")
