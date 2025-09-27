# augllm

**augllm** is a wrapper library for operating Augmented Large Language Models (LLMs) using **Ollama**.  
It provides an interface for utilizing external tools via **Function Calling**.  
Note that the actual implementations of the tools are not included—users are expected to integrate their own external implementations as needed.  

Repository:  
https://github.com/ToPo-ToPo-ToPo/augllm  

---

## Table of Contents

1. Features / Overview  
2. Requirements  
3. Installation  
4. Usage  
   - Sample Programs  
   - Integration with Function Calling  
5. License  

---

## 1. Features / Overview

- Interact with LLMs (either local or cloud-based) through **Ollama**  
- Support for tool integration using **Function Calling**  
- Tools are defined as abstract interfaces; concrete implementations (e.g., API calls, local script execution) can be freely developed by the user  
- Designed with extensibility in mind: easy integration with custom tools, chaining, and prompt engineering  

---

## 2. Requirements

- Python 3.8 or higher  
- An environment where Ollama CLI or API client is available  

---

## 3. Installation

1. Create and activate a virtual environment  

```bash
python -m venv env
```

On macOS, activate the virtual environment:  

```bash
source env/bin/activate
```

2. Install the library  

```bash
pip install .
```

---

## 4. Usage

### Sample Programs
A `test/` directory is included in the repository.  
Please refer to the two files inside as examples.  

### Integration with Function Calling

1. Provide function signatures in the prompt that represent expected tool calls  
2. Receive the function call request returned by the model (tool name + arguments)  
3. Invoke the corresponding tool interface’s `run(...)` method and obtain the result  
4. Pass the result back to the model to obtain the final response  

---

## 5. License

This project is licensed under the **Apache-2.0 License**.  
See the [LICENSE](https://github.com/ToPo-ToPo-ToPo/augllm/blob/main/LICENSE) file for details.  
