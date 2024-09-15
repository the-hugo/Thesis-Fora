import requests
import json
import pandas as pd
import os

# Base URL for the API
base_url = "https://app.fora.io/api/conversations/detail/{}"

base_path = "../data/input"
collection_path = os.path.join(base_path, "collection-150_Maine")
ids = []
for file in os.listdir(collection_path):
    if file.startswith("conv"):
        ids.append(file.split("_")[1].split(".")[0])

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
csv_file_path = f"{collection_path}_data.csv"
df.to_csv(csv_file_path, index=False)

print(f"Data saved to {csv_file_path}")
