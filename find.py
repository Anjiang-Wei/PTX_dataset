#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def find_ptx_files(directory):
    """Find all .ptx files in the given directory recursively."""
    ptx_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return ptx_files
    
    if not directory_path.is_dir():
        print(f"Error: '{directory}' is not a directory.")
        return ptx_files
    
    ptx_files = list(directory_path.rglob("*.ptx"))
    return ptx_files

def search_keywords_in_file(file_path, keywords):
    """Search for keywords in a PTX file and return which keywords were found."""
    found_keywords = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            for keyword in keywords:
                if keyword in content:
                    found_keywords.append(keyword)
                    
    except Exception as e:
        print(f"Warning: Could not read file {file_path}: {e}")
        
    return found_keywords

def generate_filename(directory_path):
    """Generate filename with last two directory names."""
    path_parts = Path(directory_path).parts
    if len(path_parts) >= 2:
        directory_name = f"{path_parts[-2]}_{path_parts[-1]}"
    else:
        directory_name = Path(directory_path).name
    return f"{directory_name}.json"

def load_and_display_summary(json_file):
    """Load JSON file and display the summary."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return False
    
    print(f"Summary from: {json_file}")
    print("=" * 60)
    print(f"Search directory: {data.get('search_directory', 'Unknown')}")
    print(f"Total PTX files: {data.get('total_ptx_files', 'Unknown')}")
    print(f"Analysis timestamp: {data.get('analysis_timestamp', 'Unknown')}")
    print()
    
    keywords_data = data.get('keywords', {})
    
    print("Keyword Summary:")
    print("=" * 60)
    
    for keyword, info in keywords_data.items():
        count = info.get('count', 0)
        example = info.get('example')
        
        if count > 0:
            print(f"{keyword}: {count} files (example: {example})")
        else:
            print(f"{keyword}: 0 files")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Find keywords in PTX files or load existing JSON results')
    parser.add_argument('path', help='Directory to search for PTX files or JSON file to load')
    
    args = parser.parse_args()
    
    # Check if the path is a JSON file or directory
    if args.path.endswith('.json'):
        # JSON mode - load and display summary
        if load_and_display_summary(args.path):
            print("\nSummary displayed successfully.")
        else:
            sys.exit(1)
        return
    else:
        # Directory mode - analyze PTX files
        directory = args.path
    
    # Define the keywords to search for
    keywords = [
        'sync',
        'bar.warp.sync',
        'expf',
        'ex2',
        'lg2',
        'sin',
        'cos',
        'tan',
        'tanh',
        'sqrt',
        'rsqrt',
        'rcp',
        'arrive',
        'mma',
        'wmma',
        'tcgen',
        'vote',
        'shuffle',
        'ballot',
        'activemask',
        'redux',
        'mbarrier',
        'fence',
        'cluster',
        'atom',
        'membar',
        'red',
        'wgmma',
        'commit_group',
        'wait_group',
        'cp.async.bulk.tensor',
        'cp.async.bulk.prefetch.tensor',
        'tensormap'
    ]
    
    print(f"Searching for PTX files in: {directory}")
    print("=" * 60)
    
    # Find all PTX files
    ptx_files = find_ptx_files(directory)
    
    if not ptx_files:
        print("No PTX files found in the specified directory.")
        sys.exit(1)
    
    print(f"Found {len(ptx_files)} PTX files")
    print()
    
    # Dictionary to store results: keyword -> list of files containing it
    keyword_files = defaultdict(list)
    
    # Process each PTX file
    processed_count = 0
    for ptx_file in ptx_files:
        found_keywords = search_keywords_in_file(ptx_file, keywords)
        
        # Add this file to the list for each found keyword
        for keyword in found_keywords:
            keyword_files[keyword].append(str(ptx_file.absolute()))
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{len(ptx_files)} files...")
    
    print(f"Finished processing {len(ptx_files)} files")
    print()
    
    # Display results
    print("Results:")
    print("=" * 60)
    
    results_summary = {}
    
    for keyword in keywords:
        files_with_keyword = keyword_files[keyword]
        count = len(files_with_keyword)
        
        if count > 0:
            example_file = files_with_keyword[0]
            print(f"{keyword}: {count} files (example: {example_file})")
        else:
            print(f"{keyword}: 0 files")
        
        # Store in results summary
        results_summary[keyword] = {
            'count': count,
            'files': files_with_keyword,
            'example': files_with_keyword[0] if files_with_keyword else None
        }
    
    # Generate JSON filename
    json_filename = generate_filename(directory)
    
    # Save results to JSON file
    output_data = {
        'search_directory': str(Path(directory).absolute()),
        'total_ptx_files': len(ptx_files),
        'analysis_timestamp': datetime.now().isoformat(),
        'keywords': results_summary
    }
    
    try:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print()
        print(f"Results saved to: {json_filename}")
        
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()