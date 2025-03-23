import pandas as pd
import numpy as np
import torch
import faiss
import os
import time
import traceback
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    """Mean pooling to get sentence embeddings"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(texts, tokenizer, model, device, batch_size=32):
    """Get embeddings for a list of texts using batching"""
    if not texts:
        print("Warning: Empty text list provided for embedding")
        return np.array([])
        
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    # Use tqdm for better progress tracking
    for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Creating embeddings"):
        batch_texts = texts[i:i + batch_size]
        
        # Handle very long texts by truncation
        batch_texts = [text[:4000] if isinstance(text, str) and len(text) > 4000 else (text if isinstance(text, str) else "") for text in batch_texts]
        
        try:
            # Tokenize the texts
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, 
                                      max_length=512, return_tensors='pt')
            
            # Move to device
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            # Get model output
            with torch.no_grad():
                model_output = model(**encoded_input)
            
            # Mean pooling
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            all_embeddings.append(embeddings.cpu().numpy())
        except Exception as e:
            print(f"Error embedding batch starting at index {i}: {e}")
            # Create an empty embedding as fallback for this batch
            embedding_dim = 384  # Dimension for the model being used
            dummy_embeddings = np.zeros((len(batch_texts), embedding_dim))
            all_embeddings.append(dummy_embeddings)
    
    if not all_embeddings:
        print("Warning: Failed to create any embeddings")
        # Return a dummy embedding if all batches failed
        embedding_dim = 384  # Dimension for the model being used
        return np.zeros((len(texts), embedding_dim))
        
    return np.vstack(all_embeddings)

def precompute_embeddings(dataset_path, output_dir='models', batch_size=32, test_sample=False):
    """Precompute embeddings for movie descriptions and save them with FAISS index"""
    start_time = total_time = time.time()
    
    # Step 1: Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Step 2: Load and prepare the data
    print(f"Loading dataset from {dataset_path}...")
    try:
        # Try different methods to load the dataset
        try:
            # First attempt with optimized parameters
            movies_df = pd.read_csv(
                dataset_path, 
                encoding='utf-8',
                on_bad_lines='skip',
                engine='c',
                low_memory=False
            )
        except Exception as e1:
            print(f"First attempt failed: {e1}")
            # Second attempt with more basic parameters
            movies_df = pd.read_csv(
                dataset_path,
                on_bad_lines='skip'
            )
            
        print(f"Successfully loaded {len(movies_df)} movies")
        
        # Print column names for debugging
        print(f"Dataset columns: {movies_df.columns.tolist()}")
        
        # Fill NaN values in Description column
        if 'Description' not in movies_df.columns:
            if 'description' in movies_df.columns:
                movies_df['Description'] = movies_df['description']
            elif 'Plot' in movies_df.columns:
                movies_df['Description'] = movies_df['Plot']
            elif 'plot' in movies_df.columns:
                movies_df['Description'] = movies_df['plot']
            elif 'overview' in movies_df.columns:
                movies_df['Description'] = movies_df['overview']
            else:
                print("No description column found. Creating empty Description column.")
                movies_df['Description'] = ""
        
        movies_df['Description'] = movies_df['Description'].fillna('')
        
        # Check if Name column exists, if not use available column
        if 'Name' not in movies_df.columns:
            if 'Title' in movies_df.columns:
                movies_df['Name'] = movies_df['Title']
            elif 'title' in movies_df.columns:
                movies_df['Name'] = movies_df['title']
            elif 'name' in movies_df.columns:
                movies_df['Name'] = movies_df['name']
            else:
                print("No name/title column found. Creating empty Name column.")
                movies_df['Name'] = ""
                
        # Ensure all Description values are strings
        movies_df['Description'] = movies_df['Description'].astype(str)
        movies_df['Name'] = movies_df['Name'].astype(str)
            
        # Just for testing with a small subset if needed
        if test_sample:
            print(f"Using only {test_sample} samples for testing")
            movies_df = movies_df.sample(min(test_sample, len(movies_df)))
            
        step_time = time.time() - start_time
        print(f"Step 2 completed in {step_time:.2f} seconds")
        start_time = time.time()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        traceback.print_exc()
        return
    
    # Step 3: Initialize model
    print("Loading transformer model...")
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        print(f"Model loaded successfully and running on {device}")
        
        step_time = time.time() - start_time
        print(f"Step 3 completed in {step_time:.2f} seconds")
        start_time = time.time()
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return
    
    # Step 4: Prepare text for embedding
    print("Preparing texts for embedding...")
    # Combine name and description for better semantic embeddings
    texts = [f"{name}: {desc}" for name, desc in zip(
        movies_df['Name'].tolist(), 
        movies_df['Description'].tolist()
    )]
    print(f"Prepared {len(texts)} texts for embedding")
    
    step_time = time.time() - start_time
    print(f"Step 4 completed in {step_time:.2f} seconds")
    start_time = time.time()
    
    # Step 5: Generate embeddings
    print("Generating embeddings...")
    embeddings = get_embedding(texts, tokenizer, model, device, batch_size=batch_size)
    
    if embeddings.size == 0:
        print("Error: Failed to create embeddings")
        return
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    step_time = time.time() - start_time
    print(f"Step 5 completed in {step_time:.2f} seconds")
    start_time = time.time()
    
    # Step 6: Build and save FAISS index
    print("Building FAISS index...")
    
    # Normalize the embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create and add vectors to index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    # Save the embeddings and index
    embeddings_path = os.path.join(output_dir, 'movie_embeddings.npy')
    index_path = os.path.join(output_dir, 'faiss_index.bin')
    
    print(f"Saving embeddings to {embeddings_path}")
    np.save(embeddings_path, embeddings)
    
    print(f"Saving FAISS index to {index_path}")
    faiss.write_index(index, index_path)
    
    step_time = time.time() - start_time
    print(f"Step 6 completed in {step_time:.2f} seconds")
    
    # Step 7: Verify saved files
    print("Verifying saved files...")
    if os.path.exists(embeddings_path) and os.path.exists(index_path):
        embeddings_size = os.path.getsize(embeddings_path) / (1024 * 1024)
        index_size = os.path.getsize(index_path) / (1024 * 1024)
        print(f"Embeddings file size: {embeddings_size:.2f} MB")
        print(f"Index file size: {index_size:.2f} MB")
        
        # Load back to verify
        try:
            loaded_embeddings = np.load(embeddings_path)
            loaded_index = faiss.read_index(index_path)
            print(f"Successfully verified files. Loaded embeddings shape: {loaded_embeddings.shape}")
            print(f"Index contains {loaded_index.ntotal} vectors")
            
            # Test a simple search
            if loaded_index.ntotal > 0:
                print("Testing a simple search...")
                query_embedding = loaded_embeddings[0:1]
                distances, indices = loaded_index.search(query_embedding, min(5, loaded_index.ntotal))
                print(f"Search test successful. Top movie index: {indices[0][0]}")
                print(f"Corresponding movie: {movies_df.iloc[indices[0][0]]['Name']}")
                
                # Save a sample of the dataset for future reference
                sample_df = movies_df.head(min(1000, len(movies_df)))
                sample_df.to_csv(os.path.join(output_dir, 'movie_sample.csv'), index=False)
                print(f"Saved sample of {len(sample_df)} movies for reference")
        except Exception as e:
            print(f"Error verifying saved files: {e}")
            traceback.print_exc()
    else:
        print("Error: Files were not saved properly")
    
    total_time = time.time() - total_time
    print(f"Total time: {total_time:.2f} seconds")
    print("Embedding precomputation completed!")

if __name__ == "__main__":
    # Run the precomputation
    dataset_path = 'data/movies_dataset_48k.csv'
    
    # Set to an integer (e.g., 1000) to test with a sample, or False to process the full dataset
    test_sample = False  # Process the full dataset by default
    
    # Use a larger batch size if you have enough GPU memory
    batch_size = 32
    
    precompute_embeddings(dataset_path, batch_size=batch_size, test_sample=test_sample) 