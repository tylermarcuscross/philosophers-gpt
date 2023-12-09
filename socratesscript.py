import os
import re
import pandas as pd

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    dialogues = []
    filename = os.path.basename(filepath).split('.')[0]
    current_speaker = None
    current_dialogue = []

    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        match = re.match(r'^([A-Z]+):', line)
        if match:
            if current_speaker:
                # Save the accumulated dialogue before starting a new one
                dialogues.append([filename, current_speaker, ' '.join(current_dialogue)])
            current_speaker = match.group(1)
            current_dialogue = [line[len(current_speaker) + 1:].strip()]
        else:
            # Continue accumulating dialogue lines for the current speaker
            current_dialogue.append(line)

    # Don't forget to add the last accumulated dialogue
    if current_speaker:
        dialogues.append([filename, current_speaker, ' '.join(current_dialogue)])

    return dialogues

def main(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            file_data = process_file(filepath)
            all_data.extend(file_data)
    df = pd.DataFrame(all_data, columns=['Book', 'Speaker', 'Dialogue'])  # Specifying column names
    return df

# Set the folder path here
folder_path = '/Users/tylercross/GitHub/philosophers-gpt/dialogues'
df = main(folder_path)
