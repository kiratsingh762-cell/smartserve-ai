import os

path = os.path.join("src", "as400", "connector.py")

lines = [
    "import pandas as pd",
    "import os",
    "import random",
    "from datetime import datetime, timedelta",
    "import sys",
    "sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))",
    "from config.settings import AS400_DB_PATH",
    "",
    "",
    "class AS400Connector:",
    '    """',
    "    Simulates IBM AS400 / DB2 database queries.",
    "    In production this would use pyodbc with IBM i Access ODBC Driver.",
    '    """',
    "",
    "    def __init__(self):",
    "        self.db_path = AS400_DB_PATH",
    "        self._ensure_db_exists()",
    "        self.df = pd.read_csv(self.db_path)",
    '        print(f"[AS400] Simulated DB loaded: {len(self.df):,} orders")',
    "",
    "    def _ensure_db_exists(self):",
    "        if not os.path.exists(self.db_path):",
    "            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)",
    "            self._generate_sample_data()",
    "",
    "    def _generate_sample_data(self):",
    "        products = [",
    '            "Laptop Pro X1", "Wireless Mouse M200", "USB-C Hub 7-Port",',
    '            "Monitor 27inch 4K", "Mechanical Keyboard K100",',
    '            "Webcam HD Pro 1080p", "Noise Cancelling Headphones",',
    '            "External SSD 1TB", "Graphics Tablet", "Smart Speaker"',
    "        ]",
    '        statuses  = ["SHIPPED", "PROCESSING", "DELIVERED", "CANCELLED", "PENDING"]',
    '        customers = [f"CUST{str(i).zfill(4)}" for i in range(1000, 1200)]',
    "        records = []",
    "        for i in range(1, 501):",
    "            order_date = datetime.now() - timedelta(days=random.randint(1, 180))",
    "            records.append({",
    '                "ORDER_ID":        f"ORD{str(i).zfill(5)}",',
    '                "CUSTOMER_ID":     random.choice(customers),',
    '                "CUSTOMER_NAME":   f"Customer {i}",',
    '                "PRODUCT":         random.choice(products),',
    '                "QUANTITY":        random.randint(1, 5),',
    '                "ORDER_DATE":      order_date.strftime("%Y-%m-%d"),',
    '                "STATUS":          random.choice(statuses),',
    '                "AMOUNT":          round(random.uniform(50, 2000), 2),',
    '                "TRACKING_NUMBER": f"TRK{random.randint(100000, 999999)}"',
    "            })",
    "        pd.DataFrame(records).to_csv(self.db_path, index=False)",
    '        print(f"[AS400] Created simulated DB: {self.db_path}")',
    "",
    "    def query_order_status(self, order_id: str) -> dict:",
    "        result = self.df[self.df['ORDER_ID'] == order_id.upper().strip()]",
    "        if result.empty:",
    '            return {"found": False, "message": f"Order {order_id} not found"}',
    "        row = result.iloc[0]",
    "        return {",
    '            "found":       True,',
    '            "order_id":    row["ORDER_ID"],',
    '            "product":     row["PRODUCT"],',
    '            "status":      row["STATUS"],',
    '            "order_date":  row["ORDER_DATE"],',
    '            "tracking":    row["TRACKING_NUMBER"],',
    '            "amount":      float(row["AMOUNT"]),',
    '            "customer_id": row["CUSTOMER_ID"]',
    "        }",
    "",
    "    def query_customer_orders(self, customer_id: str) -> list:",
    "        results = self.df[self.df['CUSTOMER_ID'] == customer_id.upper()]",
    "        return results.to_dict('records')",
]

content = "\n".join(lines) + "\n"

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"[OK] Written: {path}")
print(f"[OK] Lines  : {len(lines)}")

checks = ["class AS400Connector", "def __init__", "def query_order_status",
          "def query_customer_orders", "def _generate_sample_data"]
print("\nClass check:")
for c in checks:
    status = "[OK]" if c in content else "[MISSING]"
    print(f"  {status}  {c}")