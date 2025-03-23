with open('train_context_models.py', 'r') as f:
    lines = f.readlines()

fixed_lines = []
in_function = False
for line in lines:
    if 'def generate_context_switch_dataset' in line:
        in_function = True
        fixed_lines.append(line)
    elif in_function and 'def train_binary_classifier' in line:
        in_function = False
        fixed_lines.append(line)
    elif in_function:
        # Make sure every line inside the function has at least 4 spaces indentation
        if line.strip() and not line.startswith('    '):
            fixed_lines.append('    ' + line.lstrip())
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

with open('train_context_models.py', 'w') as f:
    f.writelines(fixed_lines)

print("Fixed indentation in train_context_models.py") 