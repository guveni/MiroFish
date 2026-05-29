import os
import sys

# Corrupt the credentials to force DefaultCredentialsError
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/invalid/path/to/creds.json"

from app.services.report_agent import ReportAgent

try:
    print("Initializing ReportAgent...")
    agent = ReportAgent(graph_id="1", simulation_id="2", simulation_requirement="3")
    print("ReportAgent initialized successfully.")
    print("Calling generate_report...")
    # Just call it and see if it throws immediately
    # agent.generate_report(report_id="test_id")
except Exception as e:
    import traceback
    traceback.print_exc()
