import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Constants
XRPL_API = "https://s1.ripple.com:51234"  # Public rippled server
TIME_GRANULARITY = 5  # in minutes
ONE_YEAR = 90 * 24 * 60  # Minutes in a year

# Function to fetch transaction data for a specific ledger
def fetch_ledger_transactions(ledger_index):
    """Fetch transactions for a specific ledger index."""
    try:
        response = requests.post(XRPL_API, json={
            "method": "ledger",
            "params": [
                {
                    "ledger_index": ledger_index,
                    "accounts": False,
                    "full": False,
                    "transactions": True
                }
            ]
        })
        if response.status_code == 200:
            ledger = response.json()["result"].get("ledger")
            if ledger and "transactions" in ledger:
                return {
                    "ledger_index": ledger_index,
                    # "timestamp": datetime.fromtimestamp(ledger["close_time"] + 946684800),
                    "timestamp": datetime.fromtimestamp(ledger["close_time"] + 946684800).strftime("%Y-%m-%d %H:%M:%S"),
                    "tx_count": len(ledger["transactions"])
                }
        return None
    except Exception as e:
        return None

# Function to get the current ledger details
def get_current_ledger():
    """Fetch the latest validated ledger index and close time."""
    response = requests.post(XRPL_API, json={
        "method": "ledger",
        "params": [
            {
                "ledger_index": "validated",
                "accounts": False,
                "full": False,
                "transactions": False
            }
        ]
    })
    if response.status_code == 200:
        ledger = response.json()["result"].get("ledger")
        if ledger:
            return int(ledger["ledger_index"]), int(ledger["close_time"])
        else:
            raise Exception("No ledger information available.")
    else:
        raise Exception(f"Failed to fetch current ledger: {response.status_code} - {response.text}")

# Main function
def main():
    # Get the latest ledger index and time
    current_ledger_index, current_ledger_time = get_current_ledger()
    print(f"Last validated ledger: {current_ledger_index}, Close time: {current_ledger_time}")

    # Calculate start time (one year ago) and estimate starting ledger index
    start_time = current_ledger_time - (ONE_YEAR * 60)
    estimated_start_ledger = current_ledger_index - (ONE_YEAR // TIME_GRANULARITY)
    print(f"Fetching data from approximately ledger {estimated_start_ledger} to {current_ledger_index}")

    # Prepare list of ledgers to fetch
    ledgers_to_fetch = list(range(estimated_start_ledger, current_ledger_index + 1))

    # Use ThreadPoolExecutor to fetch data in parallel
    data = []
    with ThreadPoolExecutor(max_workers=900) as executor:
        futures = {executor.submit(fetch_ledger_transactions, ledger): ledger for ledger in ledgers_to_fetch}

        # Progress bar
        with tqdm(total=len(ledgers_to_fetch), desc="Fetching ledger data") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    data.append(result)
                pbar.update(1)

    # Save data to CSV
    df = pd.DataFrame(data)
    df.sort_values(by="ledger_index", inplace=True)  # Ensure data is ordered
    df.to_csv("xrp_transaction_volume_parallel.csv", index=False)
    print("Data saved to xrp_transaction_volume_parallel.csv")

if __name__ == "__main__":
    main()
