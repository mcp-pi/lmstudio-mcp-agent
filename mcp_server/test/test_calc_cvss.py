import unittest
import sys
import os

# Add the parent directory to the system path to resolve the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from module.calc_cvss import calculate_cvss

class TestCalculateCVSS(unittest.TestCase):
    def test_valid_metrics(self):
        metrics = {
            'AV': 'N',  # Attack Vector
            'AC': 'L',  # Attack Complexity
            'PR': 'N',  # Privileges Required
            'UI': 'N',  # User Interaction
            'S': 'U',   # Scope
            'C': 'H',   # Confidentiality
            'I': 'H',   # Integrity
            'A': 'H'    # Availability
        }
        score = calculate_cvss(metrics)
        print(f"Test Valid Metrics: {score}")

    def test_scope_changed(self):
        metrics = {
            'AV': 'N',
            'AC': 'L',
            'PR': 'L',
            'UI': 'N',
            'S': 'C',
            'C': 'H',
            'I': 'H',
            'A': 'H'
        }
        score = calculate_cvss(metrics)
        print(f"Test Scope Changed: {score}")

    def test_low_impact(self):
        metrics = {
            'AV': 'P',
            'AC': 'H',
            'PR': 'H',
            'UI': 'R',
            'S': 'U',
            'C': 'N',
            'I': 'N',
            'A': 'L'
        }
        score = calculate_cvss(metrics)
        print(f"Test Low Impact: {score}")

    def test_invalid_metric(self):
        metrics = {
            'AV': 'X',  # Invalid value
            'AC': 'L',
            'PR': 'N',
            'UI': 'N',
            'S': 'U',
            'C': 'H',
            'I': 'H',
            'A': 'H'
        }
        try:
            score = calculate_cvss(metrics)
            print(f"Test Invalid Metric: {score}")
        except ValueError as e:
            print(f"Test Invalid Metric: Exception - {e}")

if __name__ == '__main__':
    unittest.main()
