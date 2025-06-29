# src/data/preprocess.py

"""
Data preprocessing module for J.A.R.V.I.S. training pipeline.
Handles streaming data processing for massive datasets with memory efficiency.
"""

import os
import json
import gzip
import logging
from typing import Iterator, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingDataProcessor:
    """Handles streaming data processing for massive datasets."""
    
    def __init__(self, tokenizer_name: str = "gpt2", 
                 max_length: int = 512, chunk_size: int = 1000):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.chunk_size = chunk_size
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def download_and_process_redpajama(self, max_samples: int = 1000000) -> Iterator[Dict]:
        """Download and process RedPajama-Data-1T dataset from Hugging Face."""
        logger.info("Downloading RedPajama-Data-1T dataset...")
        subsets_to_try = ['common_crawl']  # Only use common_crawl as it works reliably
        total = 0
        
        for subset in subsets_to_try:
            try:
                logger.info(f"Trying RedPajama subset: {subset}")
                dataset = load_dataset(
                    "togethercomputer/RedPajama-Data-1T",
                    subset,
                    split="train",
                    streaming=True,
                    trust_remote_code=True
                )
                
                subset_count = 0
                for i, sample in enumerate(tqdm(dataset, desc=f"Processing RedPajama {subset}", total=max_samples)):
                    if i >= max_samples:
                        break
                    
                    # Handle RedPajama's specific data structure
                    text = self._extract_redpajama_text(sample)
                    
                    if text and len(text) > 50:
                        yield {
                            'text': text,
                            'source': f'redpajama_{subset}',
                            'length': len(text)
                        }
                        total += 1
                        subset_count += 1
                
                logger.info(f"✅ Processed {subset_count} samples from RedPajama subset {subset}.")
                
                # If we got enough samples from this subset, we can stop
                if subset_count > 0:
                    logger.info(f"Successfully processed RedPajama subset {subset}. Moving to next subset...")
                    
            except Exception as e:
                logger.warning(f"❌ Failed to process RedPajama subset {subset}: {e}")
                # Fallback to local processing if file exists
                local_path = f"data/pretraining/redpajama/{subset}.jsonl.gz"
                if os.path.exists(local_path):
                    logger.info(f"Falling back to local RedPajama processing for {subset}...")
                    yield from self.process_redpajama_stream(local_path)
                else:
                    logger.error(f"Local RedPajama file not found: {local_path}. Skipping RedPajama subset {subset}.")
        
        if total == 0:
            logger.error("No usable RedPajama samples found. Skipping RedPajama dataset.")
        else:
            logger.info(f"🎉 Total RedPajama samples processed: {total}")
    
    def _extract_redpajama_text(self, sample: Dict) -> Optional[str]:
        """Extract text from RedPajama sample, handling its specific structure."""
        try:
            # RedPajama has a specific structure with 'text' and 'meta' fields
            if 'text' in sample:
                text = sample['text']
                
                # If text is a string, return it directly
                if isinstance(text, str):
                    return text
                
                # If text is a structured object (like the error shows), try to extract content
                elif isinstance(text, dict):
                    # Try to get the actual text content from the structure
                    if 'text' in text:
                        return text['text']
                    elif 'content' in text:
                        return text['content']
                    elif 'data' in text:
                        return text['data']
                    else:
                        # Convert the entire structure to string as fallback
                        return str(text)
                
                # If text is a list, join the elements
                elif isinstance(text, list):
                    return ' '.join(str(item) for item in text if item)
                
                # For any other type, convert to string
                else:
                    return str(text)
            
            # Try to get text from meta field if it exists
            if 'meta' in sample:
                meta = sample['meta']
                if isinstance(meta, dict):
                    # Look for text content in metadata
                    for key in ['text', 'content', 'data', 'title', 'description']:
                        if key in meta and isinstance(meta[key], str) and len(meta[key]) > 10:
                            return meta[key]
            
            # Fallback to the general extraction method
            return self._extract_text_from_sample(sample)
                
        except Exception as e:
            logger.debug(f"Error extracting RedPajama text: {e}")
            return None
    
    def _extract_text_from_sample(self, sample: Dict) -> Optional[str]:
        """Extract text from a sample, handling different data structures."""
        try:
            # Try to get text directly
            if 'text' in sample:
                text = sample['text']
                if isinstance(text, str):
                    return text
                elif hasattr(text, '__iter__') and not isinstance(text, (bytes, bytearray)):
                    # Handle case where text might be a list or other iterable
                    if isinstance(text, list):
                        return ' '.join(str(item) for item in text if item)
                    else:
                        return str(text)
            
            # Try alternative field names
            for field in ['content', 'data', 'document', 'passage']:
                if field in sample:
                    content = sample[field]
                    if isinstance(content, str):
                        return content
            
            # Handle structured data with metadata
            if 'metadata' in sample and 'text' in sample['metadata']:
                return sample['metadata']['text']
            
            # If sample has multiple fields, try to concatenate string fields
            text_parts = []
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 10:  # Only meaningful strings
                    text_parts.append(value)
            
            if text_parts:
                return ' '.join(text_parts)
            
            # Last resort: convert entire sample to string
            sample_str = str(sample)
            if len(sample_str) > 50:
                return sample_str
                
        except Exception as e:
            logger.debug(f"Error extracting text from sample: {e}")
            return None
        
        return None
    
    def download_and_process_c4(self, max_samples: int = 500000) -> Iterator[Dict]:
        """Download and process C4 dataset from Hugging Face."""
        logger.info("Downloading C4 dataset...")
        import os
        try:
            # Load dataset from Hugging Face (English subset)
            dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
            count = 0
            # Process samples
            for i, sample in enumerate(tqdm(dataset, desc="Processing C4", total=max_samples)):
                if i >= max_samples:
                    break
                text = self._extract_text_from_sample(sample)
                if text and len(text) > 50:
                    yield {
                        'text': text,
                        'source': 'c4',
                        'length': len(text)
                    }
                    count += 1
            logger.info(f"✅ Processed {count} samples from C4 dataset.")
        except Exception as e:
            logger.error(f"Error downloading C4: {e}")
            # Fallback to local processing if download fails and file exists
            local_path = "data/pretraining/c4/c4.jsonl"
            if os.path.exists(local_path):
                logger.info("Falling back to local C4 processing...")
                yield from self.process_c4_stream(local_path)
            else:
                logger.error(f"Local C4 file not found: {local_path}. Skipping C4 dataset.")
    
    def download_and_process_sharegpt52k(self, max_samples: int = 52000) -> Iterator[Dict]:
        """Download and process ShareGPT52K dataset from Hugging Face."""
        logger.info("Downloading ShareGPT52K dataset...")
        
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset("RyokoAI/ShareGPT52K", split="train", streaming=True)
            
            count = 0
            # Process samples
            for i, sample in enumerate(tqdm(dataset, desc="Processing ShareGPT52K", total=max_samples)):
                if i >= max_samples:
                    break
                
                conversations = sample.get('conversations', [])
                
                # Format conversations for training
                formatted_text = self._format_conversation(conversations)
                if formatted_text:
                    yield {
                        'text': formatted_text,
                        'source': 'sharegpt52k',
                        'length': len(formatted_text)
                    }
                    count += 1
            
            logger.info(f"✅ Processed {count} samples from ShareGPT52K dataset.")
                    
        except Exception as e:
            logger.error(f"Error downloading ShareGPT52K: {e}")
            # Fallback to local processing if download fails
            logger.info("Falling back to local ShareGPT52K processing...")
            yield from self.process_sharegpt_stream("data/finetuning/sharegpt52k/sharegpt52k.jsonl")
    
    def download_and_process_dialogstudio(self, max_samples: int = 1500000) -> Iterator[Dict]:
        """Download and process all major DialogStudio sub-datasets from Hugging Face."""
        logger.info("Downloading and processing all major DialogStudio sub-datasets...")
        dialogstudio_subsets = [
            # Task-oriented
            "MULTIWOZ2_2", "MultiWOZ_2.1", "SGD", "Taskmaster1", "Taskmaster2", "Taskmaster3", "WOZ2_0",
            # Open-domain
            "ShareGPT", "ConvAI2", "Empathetic", "SODA", "chitchat-dataset", "PLACES3.5", "Prosocial", "AntiScam", "HH-RLHF",
            # Knowledge-grounded
            "wizard_of_wikipedia", "wizard_of_internet",
            # Dialogue summarization
            "SAMSum", "DialogSum", "MediaSum",
            # Conversational recommendation
            "Redial", "SalesBot",
            # Natural language understanding (a few examples)
            "BANKING77", "CLINC150", "SNIPS"
        ]
        
        logger.info(f"Processing {len(dialogstudio_subsets)} DialogStudio sub-datasets...")
        total_count = 0
        successful_subsets = 0
        
        for subset_idx, subset in enumerate(tqdm(dialogstudio_subsets, desc="DialogStudio Sub-datasets", unit="dataset")):
            logger.info(f"Processing DialogStudio subset {subset_idx + 1}/{len(dialogstudio_subsets)}: {subset}")
            try:
                dataset = load_dataset(
                    "Salesforce/dialogstudio",
                    subset,
                    split="train",
                    trust_remote_code=True
                )
                logger.info(f"Subset {subset} length: {len(dataset)}")
                count = 0
                for i, sample in enumerate(tqdm(dataset, desc=f"Processing {subset}", total=min(len(dataset), max_samples), leave=False)):
                    if i >= max_samples:
                        break
                    dialog = sample.get('log', [])
                    formatted_text = self._format_dialog(dialog)
                    if formatted_text:
                        yield {
                            'text': formatted_text,
                            'source': f'dialogstudio_{subset}',
                            'length': len(formatted_text)
                        }
                        count += 1
                        total_count += 1
                logger.info(f"✅ Processed {count} samples from {subset}.")
                successful_subsets += 1
            except Exception as e:
                logger.warning(f"❌ Failed to process DialogStudio subset {subset}: {e}")
        
        logger.info(f"🎉 DialogStudio processing complete! Successfully processed {successful_subsets}/{len(dialogstudio_subsets)} sub-datasets.")
        logger.info(f"📊 Total DialogStudio samples processed: {total_count}")
    
    def process_redpajama_stream(self, file_path: str) -> Iterator[Dict]:
        """Process RedPajama-1T dataset with streaming."""
        logger.info(f"Processing RedPajama dataset: {file_path}")
        
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing RedPajama"):
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', '')
                    if isinstance(text, str) and len(text) > 50:  # Filter very short texts
                        yield {
                            'text': text,
                            'source': 'redpajama',
                            'length': len(text)
                        }
                except json.JSONDecodeError:
                    continue
    
    def process_c4_stream(self, file_path: str) -> Iterator[Dict]:
        """Process C4 dataset with streaming."""
        logger.info(f"Processing C4 dataset: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing C4"):
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', '')
                    if len(text) > 50:
                        yield {
                            'text': text,
                            'source': 'c4',
                            'length': len(text)
                        }
                except json.JSONDecodeError:
                    continue
    
    def process_sharegpt_stream(self, file_path: str) -> Iterator[Dict]:
        """Process ShareGPT52K dataset with streaming."""
        logger.info(f"Processing ShareGPT52K dataset: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing ShareGPT52K"):
                try:
                    data = json.loads(line.strip())
                    conversations = data.get('conversations', [])
                    
                    # Format conversations for training
                    formatted_text = self._format_conversation(conversations)
                    if formatted_text:
                        yield {
                            'text': formatted_text,
                            'source': 'sharegpt52k',
                            'length': len(formatted_text)
                        }
                except json.JSONDecodeError:
                    continue
    
    def process_dialogstudio_stream(self, file_path: str) -> Iterator[Dict]:
        """Process DialogStudio dataset with streaming."""
        logger.info(f"Processing DialogStudio dataset: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing DialogStudio"):
                try:
                    data = json.loads(line.strip())
                    dialog = data.get('dialog', [])
                    
                    # Format dialog for training
                    formatted_text = self._format_dialog(dialog)
                    if formatted_text:
                        yield {
                            'text': formatted_text,
                            'source': 'dialogstudio',
                            'length': len(formatted_text)
                        }
                except json.JSONDecodeError:
                    continue

    def _format_conversation(self, conversations: List[Dict]) -> str:
        """Format conversation data for training with proper user/assistant structure."""
        try:
            if not conversations or not isinstance(conversations, list):
                return ""
            
            formatted_parts = []
            for turn in conversations:
                if isinstance(turn, dict):
                    # Extract role and content
                    role = turn.get("from", "").lower()
                    content = turn.get("content") or turn.get("text") or turn.get("message") or turn.get("value", "")
                    
                    if content and isinstance(content, str) and len(content.strip()) > 0:
                        content = content.strip()
                        
                        # Format based on role
                        if role in ["human", "user", "person"]:
                            formatted_parts.append(f"User: {content}")
                        elif role in ["gpt", "assistant", "chatgpt", "claude"]:
                            formatted_parts.append(f"Assistant: {content}")
                        else:
                            # Default to user if role is unclear
                            formatted_parts.append(f"User: {content}")
            
            # Add proper conversation structure
            if formatted_parts:
                # Ensure we have alternating user/assistant pattern
                formatted_text = "\n".join(formatted_parts)
                
                # If the conversation doesn't end with assistant response, add a placeholder
                if not formatted_text.strip().endswith("Assistant:"):
                    formatted_text += "\nAssistant:"
                
                return formatted_text
            
            return ""
        except Exception as e:
            logger.debug(f"Error formatting conversation: {e}")
            return ""

    def _format_dialog(self, dialog: List[Dict]) -> str:
        """Format dialog data for training with proper user/assistant structure."""
        try:
            if not dialog or not isinstance(dialog, list):
                return ""
            
            formatted_parts = []
            for turn in dialog:
                if isinstance(turn, dict):
                    # Extract user and system responses
                    user = (turn.get("user utterance") or turn.get("user") or 
                           turn.get("input") or turn.get("question") or "")
                    system = (turn.get("system response") or turn.get("system") or 
                             turn.get("output") or turn.get("answer") or "")
                    
                    # Add user input if present
                    if user and isinstance(user, str) and len(user.strip()) > 0:
                        formatted_parts.append(f"User: {user.strip()}")
                    
                    # Add system/assistant response if present
                    if system and isinstance(system, str) and len(system.strip()) > 0:
                        formatted_parts.append(f"Assistant: {system.strip()}")
            
            # Add proper conversation structure
            if formatted_parts:
                formatted_text = "\n".join(formatted_parts)
                
                # If the conversation doesn't end with assistant response, add a placeholder
                if not formatted_text.strip().endswith("Assistant:"):
                    formatted_text += "\nAssistant:"
                
                return formatted_text
            
            return ""
        except Exception as e:
            logger.debug(f"Error formatting dialog: {e}")
            return ""

    def create_streaming_dataset(self, data_iterator, output_path, dataset_type):
        """Create a streaming dataset from iterator."""
        import json
        from tqdm import tqdm

        chunk_data = []
        chunk_size = self.chunk_size if hasattr(self, 'chunk_size') else 1000
        chunk_count = 0

        for item in tqdm(data_iterator, desc=f"Creating {dataset_type} chunks"):
            chunk_data.append(item)
            if len(chunk_data) >= chunk_size:
                chunk_path = f"{output_path}_chunk_{chunk_count}.jsonl"
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    for entry in chunk_data:
                        f.write(json.dumps(entry) + '\n')
                chunk_data = []
                chunk_count += 1

        if chunk_data:
            chunk_path = f"{output_path}_chunk_{chunk_count}.jsonl"
            with open(chunk_path, 'w', encoding='utf-8') as f:
                for entry in chunk_data:
                    f.write(json.dumps(entry) + '\n')

class DatasetManager:
    """Manages dataset operations and organization."""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.pretraining_path = self.base_path / "pretraining"
        self.finetuning_path = self.base_path / "finetuning"
        
        # Create directory structure
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directory structure."""
        directories = [
            self.pretraining_path / "redpajama",
            self.pretraining_path / "c4",
            self.finetuning_path / "sharegpt52k",
            self.finetuning_path / "dialogstudio",
            self.base_path / "processed",
            self.base_path / "checkpoints"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def download_all_datasets(self, processor: StreamingDataProcessor):
        """Download and process all datasets from Hugging Face."""
        logger.info("🚀 Starting automatic dataset download and processing...")
        
        # Download pretraining datasets
        logger.info("📚 Downloading pretraining datasets...")
        
        # RedPajama
        logger.info("Downloading RedPajama-Data-1T...")
        redpajama_iterator = processor.download_and_process_redpajama(max_samples=1000000)
        processor.create_streaming_dataset(redpajama_iterator, "data/processed/redpajama", "redpajama")
        
        # C4
        logger.info("Downloading C4...")
        c4_iterator = processor.download_and_process_c4(max_samples=500000)
        processor.create_streaming_dataset(c4_iterator, "data/processed/c4", "c4")
        
        # Download fine-tuning datasets
        logger.info("💬 Downloading fine-tuning datasets...")
        
        # ShareGPT52K
        logger.info("Downloading ShareGPT52K...")
        sharegpt_iterator = processor.download_and_process_sharegpt52k(max_samples=52000)
        processor.create_streaming_dataset(sharegpt_iterator, "data/processed/sharegpt52k", "sharegpt52k")
        
        # DialogStudio
        logger.info("Downloading DialogStudio...")
        dialogstudio_iterator = processor.download_and_process_dialogstudio(max_samples=1500000)
        processor.create_streaming_dataset(dialogstudio_iterator, "data/processed/dialogstudio", "dialogstudio")
        
        logger.info("✅ All datasets downloaded and processed successfully!")
    
    def get_dataset_info(self) -> Dict:
        """Get information about available datasets."""
        info = {
            'pretraining': {},
            'finetuning': {}
        }
        
        # Check pretraining datasets
        for dataset_dir in self.pretraining_path.iterdir():
            if dataset_dir.is_dir():
                files = list(dataset_dir.glob("*.jsonl"))
                info['pretraining'][dataset_dir.name] = {
                    'files': len(files),
                    'total_size': sum(f.stat().st_size for f in files)
                }
        
        # Check finetuning datasets
        for dataset_dir in self.finetuning_path.iterdir():
            if dataset_dir.is_dir():
                files = list(dataset_dir.glob("*.jsonl"))
                info['finetuning'][dataset_dir.name] = {
                    'files': len(files),
                    'total_size': sum(f.stat().st_size for f in files)
                }
        
        return info
