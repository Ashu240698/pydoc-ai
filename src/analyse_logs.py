"""
Analyze PyDoc AI logs for insights.
"""
import json
from pathlib import Path
from collections import Counter

def analyze_queries():
    """Analyze query patterns."""
    log_file = Path("logs/queries.log")
    
    if not log_file.exists():
        print("No query logs found.")
        return
    
    queries = []
    sources = []
    
    with open(log_file) as f:
        for line in f:
            if line.strip():
                # Parse JSON from log line (after the log prefix)
                log_data = json.loads(line.split(" - INFO - ")[1])
                queries.append(log_data['query'])
                sources.extend([s['module'] for s in log_data['sources']])
    
    print("\n" + "="*60)
    print("QUERY LOG ANALYSIS")
    print("="*60)
    print(f"\nTotal queries: {len(queries)}")
    
    print("\nMost queried topics:")
    source_counts = Counter(sources)
    for module, count in source_counts.most_common(10):
        print(f"  {module}: {count} times")
    
    print("\nRecent queries:")
    for q in queries[-5:]:
        print(f"  - {q}")

if __name__ == "__main__":
    analyze_queries()