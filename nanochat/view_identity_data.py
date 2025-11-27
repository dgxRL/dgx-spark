#!/usr/bin/env python3
"""
Script to view and analyze identity_conversations.jsonl data

Usage:
    python view_identity_data.py [options]

Options:
    --file PATH         Path to jsonl file (default: identity_conversations.jsonl)
    --num N             Number of conversations to display (default: 5, -1 for all)
    --search TEXT       Search for conversations containing TEXT
    --stats             Show statistics only
    --random            Show random samples instead of sequential
    --create-sample     Create a sample identity_conversations.jsonl file
"""

import json
import argparse
import os
import random
from pathlib import Path


def create_sample_file(filepath):
    """Create a sample identity_conversations.jsonl file."""
    sample_conversations = [
        {
            "messages": [
                {"role": "system", "content": "You are Nanochat, a helpful AI assistant created by Jason Cox."},
                {"role": "user", "content": "What's your name?"},
                {"role": "assistant", "content": "My name is Nanochat. I'm an AI assistant designed to be helpful, harmless, and honest."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are Nanochat, a helpful AI assistant created by Jason Cox."},
                {"role": "user", "content": "Who created you?"},
                {"role": "assistant", "content": "I was created by Jason Cox. He developed me as part of the Nanochat project."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are Nanochat, a helpful AI assistant created by Jason Cox."},
                {"role": "user", "content": "What is your purpose?"},
                {"role": "assistant", "content": "My purpose is to assist users by providing helpful, accurate, and thoughtful responses to their questions. I aim to be a useful tool for learning and problem-solving."}
            ]
        },
    ]
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for conv in sample_conversations:
                f.write(json.dumps(conv) + '\n')
        print(f"✓ Created sample file: {filepath}")
        print(f"  Contains {len(sample_conversations)} sample conversations")
        print(f"\nYou can now:")
        print(f"  - View it: python view_identity_data.py")
        print(f"  - Edit it manually to add more conversations")
        print(f"  - Use it for training with mid_train.py")
        return True
    except Exception as e:
        print(f"Error creating sample file: {e}")
        return False


def load_conversations(filepath):
    """Load conversations from JSONL file."""
    if not os.path.exists(filepath):
        print(f"\n⚠ File not found: {filepath}")
        print(f"\nTo create a sample file, run:")
        print(f"  python view_identity_data.py --create-sample")
        print(f"\nOr specify a different file:")
        print(f"  python view_identity_data.py --file /path/to/your/file.jsonl")
        return None
    
    conversations = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    conversations.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    return conversations


def print_conversation(conv, index=None):
    """Pretty print a single conversation."""
    header = f"\n{'='*80}\n"
    if index is not None:
        header += f"Conversation #{index + 1}\n"
    header += f"{'='*80}"
    print(header)
    
    if isinstance(conv, dict) and 'messages' in conv:
        messages = conv['messages']
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            # Color coding for roles
            if role == 'system':
                role_display = f"[SYSTEM]"
            elif role == 'user':
                role_display = f"[USER]"
            elif role == 'assistant':
                role_display = f"[ASSISTANT]"
            else:
                role_display = f"[{role.upper()}]"
            
            print(f"\n{role_display}")
            print(f"{content}")
    else:
        # Fallback for other formats
        print(json.dumps(conv, indent=2))
    
    print(f"\n{'-'*80}")


def show_statistics(conversations):
    """Show statistics about the conversations."""
    if not conversations:
        print("No conversations to analyze.")
        return
    
    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}\n")
    
    total = len(conversations)
    print(f"Total conversations: {total}")
    
    # Count message statistics
    total_messages = 0
    role_counts = {}
    content_lengths = []
    
    for conv in conversations:
        if isinstance(conv, dict) and 'messages' in conv:
            messages = conv['messages']
            total_messages += len(messages)
            
            for msg in messages:
                role = msg.get('role', 'unknown')
                role_counts[role] = role_counts.get(role, 0) + 1
                content = msg.get('content', '')
                content_lengths.append(len(content))
    
    avg_messages_per_conv = total_messages / total if total > 0 else 0
    print(f"Total messages: {total_messages}")
    print(f"Average messages per conversation: {avg_messages_per_conv:.1f}")
    
    print(f"\nMessage counts by role:")
    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count}")
    
    if content_lengths:
        avg_length = sum(content_lengths) / len(content_lengths)
        min_length = min(content_lengths)
        max_length = max(content_lengths)
        print(f"\nMessage content length statistics:")
        print(f"  Average: {avg_length:.0f} characters")
        print(f"  Min: {min_length} characters")
        print(f"  Max: {max_length} characters")
    
    # Sample topics/themes (first user message)
    print(f"\nSample topics (first user message):")
    sample_size = min(5, total)
    for i, conv in enumerate(random.sample(conversations, sample_size)):
        if isinstance(conv, dict) and 'messages' in conv:
            for msg in conv['messages']:
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    preview = content[:100] + '...' if len(content) > 100 else content
                    print(f"  {i+1}. {preview}")
                    break


def search_conversations(conversations, search_text):
    """Search for conversations containing the search text."""
    search_lower = search_text.lower()
    matching = []
    
    for i, conv in enumerate(conversations):
        if isinstance(conv, dict) and 'messages' in conv:
            for msg in conv['messages']:
                content = msg.get('content', '').lower()
                if search_lower in content:
                    matching.append((i, conv))
                    break
    
    return matching


def main():
    # Default file path matches what midtrain.sh expects
    default_file = os.path.join(os.path.expanduser("~"), ".cache", "nanochat", "identity_conversations.jsonl")
    
    parser = argparse.ArgumentParser(
        description='View and analyze identity_conversations.jsonl data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--file', type=str, default=default_file,
                        help=f'Path to jsonl file (default: {default_file})')
    parser.add_argument('--num', type=int, default=5,
                        help='Number of conversations to display (default: 5, -1 for all)')
    parser.add_argument('--search', type=str, default=None,
                        help='Search for conversations containing TEXT')
    parser.add_argument('--stats', action='store_true',
                        help='Show statistics only')
    parser.add_argument('--random', action='store_true',
                        help='Show random samples instead of sequential')
    parser.add_argument('--create-sample', action='store_true',
                        help='Create a sample identity_conversations.jsonl file')
    
    args = parser.parse_args()
    
    # Handle create-sample mode
    if args.create_sample:
        # Ensure directory exists
        os.makedirs(os.path.dirname(args.file), exist_ok=True)
        create_sample_file(args.file)
        return
    
    # Load conversations
    print(f"Loading conversations from: {args.file}")
    conversations = load_conversations(args.file)
    
    if conversations is None:
        return
    
    print(f"Loaded {len(conversations)} conversations.\n")
    
    # Handle statistics mode
    if args.stats:
        show_statistics(conversations)
        return
    
    # Handle search mode
    if args.search:
        print(f"Searching for: '{args.search}'")
        matching = search_conversations(conversations, args.search)
        print(f"Found {len(matching)} matching conversations.\n")
        
        if matching:
            display_convs = matching
            num_to_show = len(matching) if args.num == -1 else min(args.num, len(matching))
        else:
            print("No matches found.")
            return
    else:
        display_convs = [(i, conv) for i, conv in enumerate(conversations)]
        num_to_show = len(conversations) if args.num == -1 else min(args.num, len(conversations))
    
    # Select conversations to display
    if args.random and not args.search:
        selected = random.sample(display_convs, num_to_show)
    else:
        selected = display_convs[:num_to_show]
    
    # Display conversations
    for idx, (orig_idx, conv) in enumerate(selected):
        print_conversation(conv, orig_idx)
    
    # Summary
    if num_to_show < len(display_convs):
        print(f"\nShowing {num_to_show} of {len(display_convs)} conversations.")
        print(f"Use --num -1 to show all, or --num N to show N conversations.")


if __name__ == '__main__':
    main()
