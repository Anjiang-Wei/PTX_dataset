#!/usr/bin/env python3

"""
Script to analyze and summarize keyword usage across multiple JSON analysis files.

This script reads all JSON files created by find.py in the current directory and provides:
1. Keywords not used by any JSON files
2. Keywords used by all JSON files  
3. Keywords used by specific JSON files with usage statistics
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

def find_json_files(directory: Path) -> List[Path]:
    """Find all JSON files in the directory that look like analysis files."""
    json_files = []
    for file_path in directory.glob("*.json"):
        try:
            # Try to read and validate the JSON structure
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if it has the expected structure from find.py
                if 'keywords' in data and 'total_ptx_files' in data:
                    json_files.append(file_path)
        except (json.JSONDecodeError, KeyError):
            # Skip files that aren't valid analysis JSON files
            continue
    
    return sorted(json_files)

def load_analysis_data(json_files: List[Path]) -> Dict[str, Dict]:
    """Load analysis data from all JSON files."""
    analysis_data = {}
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                analysis_data[json_file.name] = data
                print(f"Loaded: {json_file.name}")
        except Exception as e:
            print(f"Warning: Could not load {json_file.name}: {e}")
    
    return analysis_data

def get_all_keywords(analysis_data: Dict[str, Dict]) -> Set[str]:
    """Get all unique keywords across all analysis files."""
    all_keywords = set()
    
    for file_name, data in analysis_data.items():
        keywords_data = data.get('keywords', {})
        all_keywords.update(keywords_data.keys())
    
    return all_keywords

def analyze_keyword_usage(analysis_data: Dict[str, Dict], all_keywords: Set[str]) -> Dict:
    """Analyze keyword usage patterns across all files."""
    keyword_analysis = {}
    total_files = len(analysis_data)
    
    for keyword in all_keywords:
        usage_info = {
            'used_in_files': [],
            'not_used_in_files': [],
            'usage_counts': {},
            'total_usage': 0
        }
        
        for file_name, data in analysis_data.items():
            keywords_data = data.get('keywords', {})
            
            if keyword in keywords_data:
                keyword_info = keywords_data[keyword]
                count = keyword_info.get('count', 0)
                
                if count > 0:
                    usage_info['used_in_files'].append(file_name)
                    usage_info['usage_counts'][file_name] = count
                    usage_info['total_usage'] += count
                else:
                    usage_info['not_used_in_files'].append(file_name)
            else:
                usage_info['not_used_in_files'].append(file_name)
        
        keyword_analysis[keyword] = usage_info
    
    return keyword_analysis

def print_summary(analysis_data: Dict[str, Dict], keyword_analysis: Dict):
    """Print comprehensive summary of keyword usage."""
    total_files = len(analysis_data)
    all_keywords = set(keyword_analysis.keys())
    
    print("=" * 80)
    print("COMPREHENSIVE KEYWORD USAGE SUMMARY")
    print("=" * 80)
    
    print(f"Analysis files processed: {total_files}")
    print(f"Total unique keywords: {len(all_keywords)}")
    print()
    
    # Show file information
    print("Analyzed files:")
    for file_name, data in analysis_data.items():
        total_ptx = data.get('total_ptx_files', 'Unknown')
        search_dir = data.get('search_directory', 'Unknown')
        print(f"  {file_name}: {total_ptx} PTX files from {Path(search_dir).name}")
    print()
    
    # 1. Keywords NOT used by any JSON files
    unused_keywords = []
    for keyword, info in keyword_analysis.items():
        if len(info['used_in_files']) == 0:
            unused_keywords.append(keyword)
    
    print("1. KEYWORDS NOT USED BY ANY FILES")
    print("-" * 40)
    if unused_keywords:
        for keyword in sorted(unused_keywords):
            print(f"  {keyword}")
        print(f"Total unused keywords: {len(unused_keywords)}")
    else:
        print("  All keywords are used by at least one file.")
    print()
    
    # 2. Keywords used by ALL JSON files
    universal_keywords = []
    for keyword, info in keyword_analysis.items():
        if len(info['used_in_files']) == total_files:
            universal_keywords.append(keyword)
    
    print("2. KEYWORDS USED BY ALL FILES")
    print("-" * 30)
    if universal_keywords:
        for keyword in sorted(universal_keywords):
            print(f"  {keyword}")
            # Show usage counts for universal keywords
            usage_counts = keyword_analysis[keyword]['usage_counts']
            for file_name in sorted(usage_counts.keys()):
                count = usage_counts[file_name]
                print(f"    {file_name}: {count} files")
        print(f"Total universal keywords: {len(universal_keywords)}")
    else:
        print("  No keywords are used by all files.")
    print()
    
    # 3. Keywords used by specific files (partial usage)
    partial_keywords = []
    for keyword, info in keyword_analysis.items():
        used_count = len(info['used_in_files'])
        if 0 < used_count < total_files:
            partial_keywords.append((keyword, info))
    
    print("3. KEYWORDS USED BY SPECIFIC FILES")
    print("-" * 35)
    if partial_keywords:
        # Sort by usage frequency (most common first)
        partial_keywords.sort(key=lambda x: len(x[1]['used_in_files']), reverse=True)
        
        for keyword, info in partial_keywords:
            used_files = info['used_in_files']
            unused_files = info['not_used_in_files']
            total_usage = info['total_usage']
            
            print(f"  {keyword}:")
            print(f"    Used by {len(used_files)}/{total_files} files (Total usage: {total_usage})")
            print(f"    Used in:")
            for file_name in sorted(used_files):
                count = info['usage_counts'][file_name]
                print(f"      {file_name}: {count} files")
            if unused_files:
                print(f"    Not used in: {', '.join(sorted(unused_files))}")
            print()
        
        print(f"Total partially-used keywords: {len(partial_keywords)}")
    else:
        print("  All keywords are either used by all files or no files.")
    print()
    
    # 4. Summary statistics
    print("4. SUMMARY STATISTICS")
    print("-" * 20)
    print(f"Keywords used by all files: {len(universal_keywords)}")
    print(f"Keywords used by some files: {len(partial_keywords)}")
    print(f"Keywords used by no files: {len(unused_keywords)}")
    print(f"Total keywords: {len(all_keywords)}")
    
    # Most and least used keywords
    if partial_keywords or universal_keywords:
        print()
        print("Most frequently used keywords (by file count):")
        all_used_keywords = [(k, len(info['used_in_files'])) for k, info in keyword_analysis.items() 
                           if len(info['used_in_files']) > 0]
        all_used_keywords.sort(key=lambda x: x[1], reverse=True)
        
        for keyword, file_count in all_used_keywords[:10]:  # Top 10
            total_usage = keyword_analysis[keyword]['total_usage']
            print(f"  {keyword}: used in {file_count}/{total_files} files ({total_usage} total occurrences)")
    
    print()

def main():
    """Main function to orchestrate the analysis."""
    
    current_dir = Path.cwd()
    print(f"Scanning directory: {current_dir}")
    
    # Find all analysis JSON files
    json_files = find_json_files(current_dir)
    
    if not json_files:
        print("No analysis JSON files found in the current directory.")
        print("Make sure you have run find.py to generate analysis files first.")
        sys.exit(1)
    
    print(f"Found {len(json_files)} analysis JSON files")
    print()
    
    # Load analysis data
    analysis_data = load_analysis_data(json_files)
    
    if not analysis_data:
        print("No valid analysis data could be loaded.")
        sys.exit(1)
    
    print()
    
    # Get all unique keywords
    all_keywords = get_all_keywords(analysis_data)
    
    # Analyze keyword usage patterns
    keyword_analysis = analyze_keyword_usage(analysis_data, all_keywords)
    
    # Print comprehensive summary
    print_summary(analysis_data, keyword_analysis)

if __name__ == "__main__":
    main()