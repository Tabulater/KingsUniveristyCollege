import time
import random
import os
from datetime import datetime

# === CONFIG ===
duration = 5.5 * 60 * 60  # 5.5 hours = 19800 seconds
interval_range = (20, 60)  # ⚡ Faster human-like edit frequency

# Updated directory
project_dir = "C:/Users/aashr/OneDrive/Desktop/Kings Internship/Neural Network/work"

# TypeScript-style project structure
file_pool = [
    "src/App.tsx",
    "src/components/Button.tsx",
    "src/utils/helpers.ts",
    "src/hooks/useTimer.ts",
    "README.md",
    "vite.config.ts"
]

# TypeScript/React realistic snippets
snippets = [
    ["// TODO: Clean this up\n"],
    ["const add = (a: number, b: number): number => {\n", "  return a + b;\n", "}\n"],
    ["export const Button = () => {\n", "  return <button>Click me</button>;\n", "}\n"],
    ["import { useState } from 'react';\n"],
    ["useEffect(() => {\n", "  console.log('Mounted');\n", "}, []);\n"],
    ["const [count, setCount] = useState(0);\n"],
    ["interface Props {\n", "  title: string;\n", "  onClick: () => void;\n", "}\n"],
    ["// Debug: check state flow\n", "console.log('State updated');\n"]
]

# === SETUP FOLDERS ===
folders = ["src", "src/components", "src/utils", "src/hooks"]
for folder in folders:
    os.makedirs(os.path.join(project_dir, folder), exist_ok=True)

# === INIT FILES ===
for file in file_pool:
    path = os.path.join(project_dir, file)
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"// Initialized: {file}\n")

# === MAIN LOOP ===
start_time = time.time()
edit_count = 0

while time.time() - start_time < duration:
    file = random.choice(file_pool)
    file_path = os.path.join(project_dir, file)

    # Open file in VS Code (in background)
    os.system(f'start /B code "{file_path}"')

    # Write a code snippet + timestamp
    with open(file_path, 'a', encoding='utf-8') as f:
        snippet = random.choice(snippets)
        timestamp = f"// {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | edit #{edit_count}\n"
        f.write(timestamp)
        f.writelines(snippet)

    edit_count += 1

    # Random deletion for realism
    if random.random() < 0.3:
        with open(file_path, 'r+', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > 6:
                del lines[random.randint(1, len(lines) - 2)]
                f.seek(0)
                f.writelines(lines)
                f.truncate()

    # Optional Git commit
    if edit_count % random.randint(8, 14) == 0:
        os.system(f'git -C "{project_dir}" add .')
        commit_msg = f"refactor: auto update #{edit_count}"
        os.system(f'git -C "{project_dir}" commit -m "{commit_msg}"')
        print(f"📦 Committed: {commit_msg}")

    # Wait for next edit
    wait_time = random.uniform(*interval_range)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Edited {file} | Waiting {int(wait_time)}s...")
    time.sleep(wait_time)

# === WRAP-UP ===
print("\n✅ Mission complete. 5.5 hours of silent, elite-level TypeScript coding activity simulated.")
