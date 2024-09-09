import requests
import json
import pandas as pd

# Base URL for the API
base_url = "https://app.fora.io/api/conversations/detail/{}"

# List of IDs to scrape
ids = [
    3146, 3145, 3132, 3144, 3130, 3133, 3129, 3113, 3028, 3112, 3138,
    3026, 3175, 2971, 3176, 2973, 2778, 2972, 2759, 2711, 2761, 2719,
    2774, 2684, 3130, 3132, 5547, 5600, 5545, 5544, 3246, 3201, 3203,
    3202, 3114, 3247, 3136, 3135
]


# Function to scrape data for each ID
def scrape_data(id_list):
    results = []
    for idx in id_list:
        # Make a request to the API
        url = base_url.format(idx)
        response = requests.get(url)

        # If the request was successful, parse the JSON data
        if response.status_code == 200:
            try:
                data = response.json()  # Convert response to JSON

                # Flatten the data or select specific fields from the JSON if needed
                results.append({'id': idx, 'data': json.dumps(data)})
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for ID {idx}")
        else:
            print(f"Failed to retrieve data for ID {idx}, Status Code: {response.status_code}")

    return results


# Run the scraper and store the results
scraped_data = scrape_data(ids)

# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(scraped_data)

# Save the DataFrame to a CSV file
csv_file_path = "scraped_data.csv"
df.to_csv(csv_file_path, index=False)

print(f"Data saved to {csv_file_path}")
