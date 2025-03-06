import pandas as pd
import sqlite3
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Define case types, statuses, priorities, and risk levels
case_types = ["Fraud", "Dispute", "Service Request", "Technical Issue", "Billing"]
case_statuses = ["Resolved", "Pending", "Escalated"]
priorities = ["Low", "Medium", "High"]
risk_levels = ["Low", "Medium", "High"]

# Generate Clients
clients = []
for i in range(1000):  # Adjust number of clients
    clients.append({
        "client_id": i + 1,
        "name": fake.name(),
        "age": random.randint(18, 75),
        "risk_level": random.choice(risk_levels),
        "previous_cases": random.randint(0, 10)
    })
df_clients = pd.DataFrame(clients)
df_clients.to_csv("clients.csv", index=False)

# Generate Assignees
assignees = []
for i in range(100):  # Adjust number of assignees
    assignees.append({
        "assignee_id": i + 1,
        "name": fake.name(),
        "role": random.choice(["Case Manager", "Support Agent"]),
        "resolved_cases": random.randint(10, 500),
        "pending_cases": random.randint(0, 50)
    })
df_assignees = pd.DataFrame(assignees)
df_assignees.to_csv("assignees.csv", index=False)

# Generate Cases
cases = []
for i in range(5000):  # Adjust number of cases
    client = random.choice(clients)
    assignee = random.choice(assignees)
    cases.append({
        "case_id": i + 1,
        "case_type": random.choice(case_types),
        "status": random.choice(case_statuses),
        "priority": random.choice(priorities),
        "assignee_id": assignee["assignee_id"],
        "client_id": client["client_id"],
        "creation_date": fake.date_between(start_date="-2y", end_date="today"),
        "resolution_time": random.randint(1, 30) if random.random() > 0.2 else None,
        "description": f"Client {client['name']} reported an issue regarding {random.choice(case_types)}. {fake.sentence()}",
        "outcome": random.choice(["Successful", "Unsuccessful"]) if random.random() > 0.2 else None
    })
df_cases = pd.DataFrame(cases)
df_cases.to_csv("cases.csv", index=False)

# Save all data to SQLite
conn = sqlite3.connect("case_management.db")

df_clients.to_sql("clients", conn, if_exists="replace", index=False)
df_assignees.to_sql("assignees", conn, if_exists="replace", index=False)
df_cases.to_sql("cases", conn, if_exists="replace", index=False)

conn.close()

print("✅ Dataset generated: CSV & SQLite saved!")
