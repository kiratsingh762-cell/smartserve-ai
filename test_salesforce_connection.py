"""
SmartServe AI — Step 1 Connection Test
Run this FIRST before any other Phase 3 code.

Usage:
    python tests/test_salesforce_connection.py

What it checks:
  ✓ Credentials load correctly from .env
  ✓ Authentication to Salesforce succeeds
  ✓ Org info is accessible
  ✓ All three queue names resolve to IDs
  ✓ Can fetch at least one open case (or confirms queues are empty)
"""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.salesforce_config import SalesforceConfig
from src.salesforce.client import SalesforceClient
from src.salesforce.case_manager import CaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
INFO = "\033[94mℹ\033[0m"


def run_tests():
    print("\n" + "="*60)
    print("  SmartServe AI — Salesforce Connection Test (Step 1)")
    print("="*60 + "\n")

    results = []

    # ── Test 1: Config loads ──────────────────────────────────
    print(f"{INFO}  Test 1: Loading credentials from .env...")
    try:
        config = SalesforceConfig.from_env()
        print(f"  {PASS}  Credentials loaded")
        print(f"         Username : {config.username}")
        print(f"         Domain   : {config.domain}.salesforce.com")
        print(f"         Queues   : {config.all_queues}")
        results.append(True)
    except Exception as e:
        print(f"  {FAIL}  {e}")
        print("\n  ⛔ Fix your .env file before continuing.\n")
        sys.exit(1)

    # ── Test 2: Authentication ────────────────────────────────
    print(f"\n{INFO}  Test 2: Authenticating to Salesforce...")
    try:
        client = SalesforceClient(config)
        sf = client.sf
        print(f"  {PASS}  Authenticated")
        print(f"         Instance : {client.get_instance_url()}")
        results.append(True)
    except Exception as e:
        print(f"  {FAIL}  Authentication failed: {e}")
        print("  Checklist:")
        print("    • Is SF_DOMAIN=login (production) or test (sandbox)?")
        print("    • Did you reset your Security Token after last password change?")
        print("    • Is Connected App set to Relaxed IP restrictions?")
        results.append(False)

    # ── Test 3: Org info ──────────────────────────────────────
    print(f"\n{INFO}  Test 3: Fetching org info...")
    try:
        org = client.get_org_info()
        user = client.get_current_user()
        print(f"  {PASS}  Org accessible")
        print(f"         Org Name : {org.get('Name', 'Unknown')}")
        print(f"         Org Type : {org.get('OrganizationType', 'Unknown')}")
        print(f"         User     : {user.get('Name', 'Unknown')} ({user.get('UserType', '?')})")
        results.append(True)
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(False)

    # ── Test 4: Queue resolution ──────────────────────────────
    print(f"\n{INFO}  Test 4: Resolving queue IDs...")
    try:
        all_queues = config.all_queues + [config.queue_escalation]
        queue_ids = client.get_queue_ids(all_queues)
        print(f"  {PASS}  Queue IDs resolved ({len(queue_ids)}/{len(all_queues)} found)")
        for name, qid in queue_ids.items():
            print(f"         {name[:35]:<35} → {qid}")
        if len(queue_ids) < len(all_queues):
            missing = [q for q in all_queues if q not in queue_ids]
            print(f"\n  ⚠  Missing queues: {missing}")
            print("     Check SF_QUEUE_* env vars match exact Salesforce queue names.")
        results.append(len(queue_ids) > 0)
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(False)

    # ── Test 5: Case fetch ────────────────────────────────────
    print(f"\n{INFO}  Test 5: Fetching cases from queues...")
    case_manager = CaseManager(sf)
    try:
        total = 0
        for queue_name, queue_id in queue_ids.items():
            if queue_name == config.queue_escalation:
                continue
            cases = case_manager.get_open_cases_for_queue(queue_name, queue_id, limit=5)
            print(f"  {PASS}  '{queue_name}': {len(cases)} open case(s)")
            for c in cases[:2]:
                print(f"         [{c.get('CaseNumber')}] {c.get('Subject', '')[:55]}")
            total += len(cases)
        if total == 0:
            print(f"  {INFO}  No open cases right now — that's fine, queues are accessible.")
        results.append(True)
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(False)

    # ── Summary ───────────────────────────────────────────────
    passed = sum(results)
    total_tests = len(results)
    print("\n" + "="*60)
    print(f"  Results: {passed}/{total_tests} tests passed")
    print("="*60)

    if passed == total_tests:
        print("\n  🎉 All tests passed! Salesforce connection is ready.")
        print("     Next: python tests/test_salesforce_connection.py")
        print("     Then: Start Step 2 — Case Ingestion & Queue Polling\n")
    else:
        print("\n  ⚠  Some tests failed. Fix the issues above before Step 2.\n")

    return passed == total_tests


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
