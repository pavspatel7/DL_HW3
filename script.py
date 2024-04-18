# on terminal run: srun -G 4 --pty python3 script.py

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from nbconvert.preprocessors import ExecutePreprocessor
from IPython.display import display, Markdown, HTML
import torch

cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

num_cuda_devices = torch.cuda.device_count()
print(f"Number of CUDA devices available: {num_cuda_devices}")

if cuda_available:
    current_device = torch.cuda.current_device()
    print(f"Current CUDA device ID: {current_device}")
    print(f"Device Name: {torch.cuda.get_device_name(current_device)}")

def execute_notebook_and_save_with_output(notebook_path):
    print("Loading notebook...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    shell = InteractiveShell.instance()

    ep = ExecutePreprocessor(timeout=None, kernel_name='python3')
    nb, resources = ep.preprocess(nb, resources={'metadata': {'path': './'}})

    num_cells = len(nb['cells'])
    print(f"Preparing to execute and display {num_cells} cells from the notebook...")

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            print(f"\nExecuting cell {i + 1}/{num_cells}:\n{cell['source']}\n---")
            # Display outputs if any
            if 'outputs' in cell:
                for output in cell['outputs']:
                    display_output(output)
            print(f"\nCell {i + 1} executed successfully.")
        else:
            print(f"\nSkipping non-code cell {i + 1}.")

    # Write the notebook back to the file with outputs after all cells have been executed
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("Notebook execution and output capture completed.")

def display_output(output):
    if 'text' in output:
        print(output['text'])
    elif 'data' in output:
        for output_type, output_data in output['data'].items():
            if output_type == 'text/plain':
                print(output_data)
            elif output_type == 'text/html':
                display(HTML(output_data))
            elif output_type == 'text/markdown':
                display(Markdown(output_data))
            else:
                print(f"Unsupported output type: {output_type}")

if __name__ == "__main__":
    notebook_path = 'test.ipynb'  # Update this path to your notebook
    execute_notebook_and_save_with_output(notebook_path)