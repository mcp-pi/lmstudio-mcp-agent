# MCP-Based Prompt Injection Framework - Quick Start Guide

## 🚨 Prerequisites Check

Before running the framework, verify your environment:

```bash
python check_env.py
```

This will check:
- ✅ Python version (3.12+)
- ✅ Required dependencies
- ✅ LM Studio connection
- ✅ MCP configuration
- ✅ Dataset availability
- ✅ Directory structure

## 📦 Installation

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Start LM Studio

1. Launch LM Studio
2. Load at least one model (e.g., qwen3-4b, llama-3.2-1b)
3. Start the local server (usually on http://localhost:1234)

### 3. Verify Installation

```bash
python check_env.py
```

All checks should show ✓ PASS.

## 🚀 Running the Framework

### Option 1: Quick Test (5 attacks)

```bash
python attack_pipeline.py --quick-test
```

### Option 2: Demo Script

```bash
python run_demo.py
```

### Option 3: Full Assessment

```bash
# Default settings (10 attacks)
python attack_pipeline.py

# Custom settings
python attack_pipeline.py --attack-count 20 --complexity medium

# With specific target model
python attack_pipeline.py --target-model "llama-3.2-1b-instruct" --attack-count 15
```

## 🎯 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--target-model MODEL` | Specify target LLM model | Prompts user |
| `--attack-count N` | Number of attacks to execute | 10 |
| `--complexity LEVEL` | Starting complexity (low/medium/high/critical) | low |
| `--no-escalate` | Disable complexity escalation | False |
| `--no-adaptive` | Disable adaptive learning | False |
| `--quick-test` | Run minimal test (5 attacks) | False |
| `--dataset-path PATH` | Custom dataset file path | Auto-detect |

## 📊 Output Files

After running, check the `./reports/` directory for:

- `report_TIMESTAMP.json` - Detailed JSON report
- `report_TIMESTAMP.html` - Visual HTML report
- `visualizations_TIMESTAMP.png` - Charts and graphs
- `attack_results_TIMESTAMP.json` - Raw attack data
- `learning_data_TIMESTAMP.json` - Adaptive strategy data

## 🔧 Troubleshooting

### "No module named 'numpy'" Error
```bash
uv sync
# or
pip install numpy matplotlib seaborn
```

### "Cannot connect to LM Studio"
1. Make sure LM Studio is running
2. Check that a model is loaded
3. Verify server is on http://localhost:1234

### "No MCP tools available"
1. Check `mcp_config.json` exists
2. Verify MCP server paths are correct
3. Try running individual MCP servers manually

### "No dataset found"
1. Ensure `dataset/` directory exists
2. Check for data files in `dataset/data/`
3. Use `--dataset-path` to specify custom location

## 🎨 Example Usage Scenarios

### Security Audit
```bash
# Comprehensive security assessment
python attack_pipeline.py --attack-count 50 --complexity low --adaptive
```

### Quick Vulnerability Check
```bash
# Fast 5-minute test
python attack_pipeline.py --quick-test --no-adaptive
```

### Specific Model Testing
```bash
# Test a specific model with high complexity attacks
python attack_pipeline.py --target-model "codellama-7b" --complexity high --attack-count 30
```

### Research Mode
```bash
# Full assessment with all features
python attack_pipeline.py --attack-count 100 --adaptive --complexity low
```

## 📈 Understanding Results

### Success Rate
- **0-20%**: Model has strong defenses
- **20-50%**: Moderate vulnerabilities
- **50-70%**: Significant vulnerabilities
- **70-100%**: Critical security issues

### CVSS Scores
- **0.0**: No impact
- **0.1-3.9**: Low severity
- **4.0-6.9**: Medium severity
- **7.0-8.9**: High severity
- **9.0-10.0**: Critical severity

## 🛡️ Security Notice

This tool is for authorized security testing only. Always:
- ✅ Get permission before testing
- ✅ Use on your own models or with explicit consent
- ✅ Follow responsible disclosure practices
- ❌ Never use for malicious purposes

## 📞 Support

If you encounter issues:
1. Run `python check_env.py` first
2. Check the troubleshooting section
3. Review error messages carefully
4. Ensure all prerequisites are met

Happy testing! 🚀
