import pandas as pd
import random

# List of items (you can expand this)
#items_pool = [
#  "Milk","Bread","Butter","Cheese","Eggs","Juice","Chips","Chocolate",
#   "Rice","Dal","Oil","Sugar","Salt","Tea","Coffee","Biscuits",
#   "Soap","Shampoo","Toothpaste","Detergent" ]
#
items_pool = [
    "Laptop",
    "Smartphone",
    "Headphones",
    "Charger",
    "Power Bank",
    "USB Cable",
    "Bluetooth Speaker",
    "Smartwatch",
    "Tablet",
    "Wireless Mouse",
    "Mechanical Keyboard",
    "Monitor",
    "External Hard Drive",
    "SSD",
    "RAM Stick",
    "Graphics Card",
    "Webcam",
    "Microphone",
    "Tripod",
    "Drone",
    "VR Headset",
    "Smart Bulb",
    "WiFi Router",
    "Power Strip",
    "HDMI Cable",
    "USB Hub",
    "Laptop Stand",
    "Cooling Pad",
    "Printer",
    "Scanner","Graphics Card", "CPU", "Motherboard", "Power Supply Unit"
#"Milk","Bread","Butter","Cheese","Eggs",
# "Juice","Chips","Chocolate",
# "Rice","Dal","Oil","Sugar","Salt",
# "Tea","Coffee","Biscuits",
# "Soap","Shampoo","Toothpaste","Detergent"
]

data = []
Number_of_data = 5000

for i in range(1,Number_of_data + 1):  # enter entries
    user_id = random.randint(1, Number_of_data//3)  # random users
    num_items = random.randint(0, 5)    # each transaction has 0–5 items
    items = random.sample(items_pool, num_items)

    data.append({
        "TransactionID": i,
        "UserID": user_id,
        "Items": ",".join(items)
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save CSV
df.to_csv(f"transactions_{Number_of_data}.csv", index=False)

print(f"✅ Dataset generated: transactions_{Number_of_data}.csv")