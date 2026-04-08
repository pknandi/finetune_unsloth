import csv
from pathlib import Path

def generate_dataset_csv(root_folder: str, output_csv: str = "dataset_mapping.csv"):
    root_path = Path(root_folder)
    audio_base_dir = root_path / "audio"
    motion_base_dir = root_path / "smplx"
    
    dataset_rows = []
    
    if not audio_base_dir.exists():
        print(f"Error: The directory {audio_base_dir} does not exist.")
        return

    for session_dir in audio_base_dir.iterdir():
        if not session_dir.is_dir():
            continue
            
        for subject_dir in session_dir.iterdir():
            if not subject_dir.is_dir():
                continue
                
            audio_separated_dir = subject_dir / "audio_separated"
            
            if audio_separated_dir.exists() and audio_separated_dir.is_dir():
                

                motion_dir = motion_base_dir / session_dir.name / subject_dir.name
                
                for audio_file in audio_separated_dir.iterdir():
                    if audio_file.is_file():
                        dataset_rows.append({
                            "audio_filename": str(audio_file), 
                            "motion_dirname": str(motion_dir)
                        })
                        
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['audio_filename', 'motion_dirname']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(dataset_rows)
        
    print(f"Success! Generated '{output_csv}' with {len(dataset_rows)} matching entries.")

if __name__ == "__main__":
    ROOT_FOLDER = "./datasets/small"
    
    generate_dataset_csv(ROOT_FOLDER)