# RAG Support DNN Framework

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

RAG_Support_DNN is a Python framework that combines Retrieval-Augmented Generation (RAG) techniques with Deep Neural Networks. It provides configurable PyTorch models, LangChain/LangGraph agents for source evaluation, dataset processing utilities, and prompt templates for RAG tasks.

## Working Effectively

### Initial Setup and Dependencies
- Ensure Python 3.12+ is available: `python3 --version`
- Install core dependencies: `pip3 install -r RAG_supporters/requirements.txt` -- takes 8-12 minutes. NEVER CANCEL. Set timeout to 20+ minutes.
- Install optional agent dependencies: `pip3 install -r RAG_supporters/requirements_agents.txt` -- may fail due to network timeouts. Set timeout to 15+ minutes. If it fails, the code will gracefully fall back to placeholder classes.

### Code Quality and Formatting
- ALWAYS run code formatting before committing: `python3 -m black RAG_supporters/` -- takes less than 1 second
- Check formatting compliance: `python3 -m black --check RAG_supporters/` -- takes less than 1 second
- The CI build will fail if code is not properly formatted with black

### Core Module Structure
```
RAG_supporters/
├── agents/                    # LangChain/LangGraph agents (optional dependencies)
├── dataset/                   # Dataset processing and utilities
├── nn/                        # Neural network models and builders
├── prompts_templates/         # Prompt templates for RAG tasks
├── requirements.txt           # Core dependencies
└── requirements_agents.txt    # Optional agent dependencies
```

## Validation

### Neural Network Model Validation
ALWAYS test neural network functionality after making changes to the nn/ module:

```bash
# Create a test configuration
mkdir -p /tmp/test_configs
cat > /tmp/test_configs/simple_model.yaml << EOF
model:
  input_features: 10
  output_features: 2
  layers:
    - type: Linear
      in_features: 10
      out_features: 5
    - type: ReLU
    - type: Linear
      in_features: 5
      out_features: 2
EOF

# Test model creation and validation
python3 -c "
import RAG_supporters.nn.models.model_builder as mb
model = mb.ConfigurableModel('/tmp/test_configs/simple_model.yaml', warmup_validate=True)
print('✓ Neural network model validation successful')
print(model.get_model_summary())
"
```

### Agent Functionality Validation
Test agent import and fallback behavior:

```bash
python3 -c "
from RAG_supporters.agents.source_assesment import SourceEvaluationAgent
from RAG_supporters.agents.dataset_check import DatasetCheckAgent
try:
    agent = SourceEvaluationAgent()
    print('✓ Agents available with full dependencies')
except ImportError as e:
    print('✓ Graceful fallback behavior working:', str(e))
"
```

### Complete Validation Scenario
Run this complete validation after making any changes:

```bash
# 1. Format code
python3 -m black RAG_supporters/

# 2. Test core imports
python3 -c "
import RAG_supporters.nn.models.model_builder as mb
# Note: rag_dataset has import issues, test model builder instead
print('✓ Core modules import successfully')
"

# 3. Test model creation (requires test config from above)
python3 -c "
import RAG_supporters.nn.models.model_builder as mb
model = mb.ConfigurableModel('/tmp/test_configs/simple_model.yaml', warmup_validate=True)
print('✓ Model creation and validation working')
"

# 4. Test agent fallback
python3 -c "
from RAG_supporters.agents.source_assesment import SourceEvaluationAgent
try:
    agent = SourceEvaluationAgent()
except ImportError:
    print('✓ Agent fallback behavior working')
"
```

## Common Tasks

### Exploring the Codebase
Use these commands to quickly understand the repository structure:

```bash
# View overall structure
ls -la RAG_supporters/

# Key directories to explore
ls -la RAG_supporters/nn/models/          # Neural network implementation
ls -la RAG_supporters/agents/             # LangChain/LangGraph agents  
ls -la RAG_supporters/prompts_templates/  # RAG prompt templates
ls -la RAG_supporters/dataset/            # Dataset processing utilities

# Check requirements
cat RAG_supporters/requirements.txt       # Core dependencies
cat RAG_supporters/requirements_agents.txt # Optional agent dependencies
```

### Working with Neural Network Models
- Model configurations are defined in YAML files
- Use ConfigurableModel class: `RAG_supporters.nn.models.model_builder.ConfigurableModel`
- Models support automatic device detection (CPU/CUDA)
- Always use `warmup_validate=True` to test model architecture
- Models expect `layers` as a list of layer configurations, not a dictionary

### Working with Agents  
- Agents require optional dependencies (langchain-openai, langgraph, pydantic)
- If dependencies are missing, placeholder classes are imported that raise ImportError
- Import paths: `RAG_supporters.agents.source_assesment.SourceEvaluationAgent`, `RAG_supporters.agents.dataset_check.DatasetCheckAgent`
- Agents use LangGraph for workflow management

### Working with Datasets
- Dataset utilities in `RAG_supporters.dataset/`
- Templates for specific datasets like BioASQ in `RAG_supporters.dataset.templates/`
- **IMPORTANT**: Some dataset modules have relative import issues - use individual module imports rather than the main rag_dataset module
- Dataset loading utilities are in `RAG_supporters.dataset.utils.dataset_loader`

### Working with Prompts
- Prompt templates in `RAG_supporters.prompts_templates/`
- Key templates: `rag_generators.py` (text extraction), `rag_verifiers.py` (source comparison)
- Templates use string formatting with placeholders like `{question}`, `{source1_content}`

## Important Implementation Details

### Configuration File Format
Neural network models expect YAML configurations with this structure:
```yaml
model:
  input_features: <int>
  output_features: <int>
  layers:
    - type: <layer_type>
      <layer_parameters>
    - type: <layer_type>
      <layer_parameters>
```

### Import Dependencies
- Core modules (nn/, dataset/ without agents) work with base requirements
- Agent modules require additional langchain dependencies
- Some dataset modules have relative imports that may need adjustment
- Always test imports after changes

### Code Style Requirements
- Black formatting is mandatory - CI will fail without it
- No specific linting beyond black formatting is configured
- Python 3.12+ type hints and features are used throughout

## Troubleshooting

### Common Issues
- **"No module named 'agents'"**: Use full import paths like `RAG_supporters.agents.dataset_check`
- **Agent ImportError**: Expected behavior when optional dependencies are missing
- **Model config AttributeError**: Ensure YAML layers is a list, not a dictionary
- **Network timeouts during pip install**: Retry with longer timeout, agents will fall back gracefully if installation fails

### Build Time Expectations
- Core dependency installation: 8-12 minutes
- Agent dependency installation: 5-15 minutes (may fail due to network)
- Black formatting: <1 second
- Model validation: <5 seconds
- Full validation scenario: <30 seconds

### Manual Testing Requirements
After changes, ALWAYS:
1. Run black formatting
2. Test core module imports  
3. Create and validate a simple neural network model
4. Verify agent fallback behavior works correctly
5. Test any new functionality with actual data/configs