# LM Studio MCP Agent - Prompt Injection Framework

This repository implements the **MCP-Based LLM Prompt Injection Automated Vulnerability Assessment Framework** as described in the research paper "MCP Protocol을 활용한 LLM Prompt Injection 자동화 취약점 분석 프레임워크 설계" by Seungjung Kim.

## 🚀 New Features (test branch)

### Automated Prompt Injection Attack Framework
- **LLM-to-LLM Communication**: Automated interaction between attacker and target LLMs via MCP protocol
- **Attack Template Library**: Pre-defined attack patterns (System Prompt Bypass, Role Impersonation, Indirect Injection, etc.)
- **Adaptive Learning**: Feedback-based strategy adjustment for improved attack effectiveness
- **CVSS Integration**: Automated vulnerability scoring using CVSS 3.1 metrics
- **Comprehensive Reporting**: HTML and JSON reports with visualizations

## 📋 Prerequisites

- Python 3.12 or higher
- [LM Studio](https://lmstudio.ai) with loaded models
- [uv](https://github.com/astral-sh/uv) package manager
- Two LLM models loaded in LM Studio:
  - Attacker model (e.g., qwen/qwen3-4b)
  - Target model (e.g., llama-3.2-1b-instruct)

## 🛠️ Installation

1. Clone and switch to test branch:
```bash
git clone https://github.com/godstale/lmstudio-mcp-agent
cd lmstudio-mcp-agent
git checkout test
```

2. Install dependencies:
```bash
uv sync
# or
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env to set your target model (optional)
```

## 🔥 Quick Start - Vulnerability Assessment

### Run Quick Test (5 attacks)
```bash
uv run attack_pipeline.py --quick-test
```

### Run Full Assessment
```bash
uv run attack_pipeline.py --attack-count 20 --complexity medium --adaptive
```

### Command Line Options
- `--target-model MODEL`: Specify target model (prompts if not set)
- `--attack-count N`: Number of attacks to execute (default: 10)
- `--complexity [low|medium|high|critical]`: Starting complexity (default: low)
- `--no-escalate`: Disable complexity escalation
- `--no-adaptive`: Disable adaptive strategy
- `--dataset-path PATH`: Custom dataset path

## 📊 Output

The framework generates:
- **JSON Report**: Detailed attack results and analysis
- **HTML Report**: Visual summary with charts
- **Learning Data**: Adaptive strategy insights
- **Visualizations**: Success rates, CVSS distributions

Reports are saved in the `./reports/` directory.

## 🏗️ Architecture

```
lmstudio-mcp-agent/
├── attack_framework/           # Core framework modules
│   ├── attack_templates.py    # Attack pattern library
│   ├── attack_executor.py     # LLM-to-LLM execution engine  
│   ├── feedback_loop.py       # Adaptive learning mechanism
│   └── report_generator.py    # CVSS-integrated reporting
├── mcp_server/                # MCP server implementations
│   ├── mcp-pi.py             # Prompt injection attacker
│   ├── cal_cvss.py           # CVSS calculator
│   └── select_prompt.py      # Dataset selector
├── dataset/                   # Attack datasets
│   └── data/                 # Jailbreak prompts
└── attack_pipeline.py        # Main execution pipeline
```

## 🔍 Attack Types Supported

1. **System Prompt Bypass**: Attempts to override system instructions
2. **Role Impersonation**: Identity and permission manipulation
3. **Indirect Injection**: Hidden commands in context
4. **Jailbreak**: Restriction bypass attempts
5. **Data Leakage**: Training data extraction

## 📈 Example Usage

```python
# Basic assessment
uv run attack_pipeline.py

# Advanced with specific target
uv run attack_pipeline.py --target-model "llama-3.2-1b" --attack-count 30 --complexity high

# Quick security check
uv run attack_pipeline.py --quick-test --no-adaptive
```

## 🛡️ Security Notice

This tool is for **authorized security testing only**. Use responsibly and only on systems you have permission to test.

## 📚 Original Features

The base system still includes:
- LM Studio and OpenAI API support
- MCP tool integration
- Streaming responses
- Multi-model support

See the [original README](README_original.md) for base features.

## 🤝 Contributing

Contributions are welcome! Please ensure all tests pass and follow the existing code style.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Research paper: "MCP Protocol을 활용한 LLM Prompt Injection 자동화 취약점 분석 프레임워크 설계"
- Author: Seungjung Kim (Sunrin Internet High School)
- Original LM Studio MCP Agent contributors
