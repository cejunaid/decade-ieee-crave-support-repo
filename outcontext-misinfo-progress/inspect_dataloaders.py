import torch
import numpy as np
import pandas as pd

def inspect_dataloaders(test_dataloader, verite_dataloader,
                       test_data, verite_test,
                       image_embeddings, verite_image_embeddings,
                       text_embeddings, verite_text_embeddings,
                       X_image_embeddings, X_verite_image_embeddings,
                       X_text_embeddings, X_verite_text_embeddings,
                       use_evidence=None):
    """
    Comprehensive analysis of two dataloaders to find differences
    """
    # PART 1: COMPARE DATALOADER CONFIGURATIONS
    print("===== DATALOADER CONFIGURATION COMPARISON =====")
    print(f"Batch size - test: {test_dataloader.batch_size}, verite: {verite_dataloader.batch_size}")
    print(f"Num workers - test: {test_dataloader.num_workers}, verite: {verite_dataloader.num_workers}")
    print(f"Dataset types - test: {type(test_dataloader.dataset)}, verite: {type(verite_dataloader.dataset)}")
    print(f"Sampler types - test: {type(test_dataloader.sampler)}, verite: {type(verite_dataloader.sampler)}")
    print(f"Collate functions identical: {test_dataloader.collate_fn is verite_dataloader.collate_fn}")
    
    # PART 2: COMPARE INPUT DATA
    print("\n===== INPUT DATA COMPARISON =====")
    print(f"test_data type: {type(test_data)}")
    print(f"verite_test type: {type(verite_test)}")
    
    # Handle different input data types
    if isinstance(test_data, dict) and isinstance(verite_test, dict):
        # Compare dictionary keys
        test_keys = set(test_data.keys())
        verite_keys = set(verite_test.keys())
        print(f"Keys only in test_data: {test_keys - verite_keys}")
        print(f"Keys only in verite_test: {verite_keys - test_keys}")
        print(f"Shared keys: {test_keys & verite_keys}")
        
        # Compare a few values from shared keys
        for key in sorted(list(test_keys & verite_keys))[:3]:
            test_val = test_data[key]
            verite_val = verite_test[key]
            print(f"\nKey '{key}':")
            print(f"  Types - test: {type(test_val)}, verite: {type(verite_val)}")
            
            # Check if values can be compared
            try:
                are_equal = test_val == verite_val
                if isinstance(are_equal, bool):
                    print(f"  Values match: {are_equal}")
                elif hasattr(are_equal, 'all'):
                    print(f"  Values match: {are_equal.all()}")
                else:
                    print(f"  Values match: {are_equal}")
            except Exception as e:
                print(f"  Error comparing values: {e}")
    
    elif isinstance(test_data, pd.DataFrame) and isinstance(verite_test, pd.DataFrame):
        # Compare DataFrame properties
        print(f"Shapes - test: {test_data.shape}, verite: {verite_test.shape}")
        
        # Compare column names
        test_cols = set(test_data.columns)
        verite_cols = set(verite_test.columns)
        print(f"Columns only in test_data: {test_cols - verite_cols}")
        print(f"Columns only in verite_test: {verite_cols - test_cols}")
        
        # Check if indices match
        idx_equal = test_data.index.equals(verite_test.index)
        print(f"Indices match: {idx_equal}")
        
        # Sample a few row values for comparison
        try:
            if len(test_data) > 0 and len(verite_test) > 0:
                shared_cols = list(test_cols & verite_cols)
                if shared_cols:
                    for col in shared_cols[:2]:
                        col_equal = test_data[col].equals(verite_test[col])
                        print(f"Column '{col}' values match: {col_equal}")
        except Exception as e:
            print(f"Error comparing DataFrame samples: {e}")
    
    # PART 3: COMPARE EMBEDDINGS
    print("\n===== EMBEDDINGS COMPARISON =====")
    
    # Helper function to compare embeddings safely
    def compare_data_safely(data1, data2, name1, name2):
        print(f"\n{name1} vs {name2}:")
        print(f"Types - {name1}: {type(data1)}, {name2}: {type(data2)}")
        
        # Check if shapes available
        if hasattr(data1, 'shape') and hasattr(data2, 'shape'):
            print(f"Shapes - {name1}: {data1.shape}, {name2}: {data2.shape}")
            shape_match = data1.shape == data2.shape
            print(f"Shapes match: {shape_match}")
            
            if not shape_match:
                print("⚠️ DIFFERENCE DETECTED: Embedding shapes don't match!")
                return False
        
        # Type-specific comparisons
        if isinstance(data1, torch.Tensor) and isinstance(data2, torch.Tensor):
            try:
                equal = torch.allclose(data1, data2)
                print(f"Values match: {equal}")
                if not equal:
                    print("⚠️ DIFFERENCE DETECTED: Tensor values don't match!")
                return equal
            except Exception as e:
                print(f"Error comparing tensors: {e}")
                return False
        
        elif isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
            try:
                cols_match = set(data1.columns) == set(data2.columns)
                print(f"Columns match: {cols_match}")
                
                equal = data1.equals(data2)
                print(f"Values match: {equal}")
                
                if not equal:
                    print("⚠️ DIFFERENCE DETECTED: DataFrame values don't match!")
                return equal
            except Exception as e:
                print(f"Error comparing DataFrames: {e}")
                return False
        
        elif isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
            try:
                equal = np.array_equal(data1, data2)
                print(f"Values match: {equal}")
                if not equal:
                    print("⚠️ DIFFERENCE DETECTED: Array values don't match!")
                return equal
            except Exception as e:
                print(f"Error comparing arrays: {e}")
                return False
        
        else:
            print("Cannot directly compare different types of data")
            print("⚠️ DIFFERENCE DETECTED: Data types don't match!")
            return False
    
    # Compare all embeddings
    compare_data_safely(image_embeddings, verite_image_embeddings, "image_embeddings", "verite_image_embeddings")
    compare_data_safely(text_embeddings, verite_text_embeddings, "text_embeddings", "verite_text_embeddings")
    compare_data_safely(X_image_embeddings, X_verite_image_embeddings, "X_image_embeddings", "X_verite_image_embeddings")
    compare_data_safely(X_text_embeddings, X_verite_text_embeddings, "X_text_embeddings", "X_verite_text_embeddings")
    
    # PART 4: INSPECT ACTUAL BATCH CONTENTS
    print("\n===== BATCH CONTENT COMPARISON =====")
    
    # Extract first batch from each dataloader
    try:
        test_iter = iter(test_dataloader)
        verite_iter = iter(verite_dataloader)
        
        test_batch = next(test_iter)
        verite_batch = next(verite_iter)
        
        print(f"Batch types - test: {type(test_batch)}, verite: {type(verite_batch)}")
        
        if isinstance(test_batch, (list, tuple)) and isinstance(verite_batch, (list, tuple)):
            print(f"Batch lengths - test: {len(test_batch)}, verite: {len(verite_batch)}")
            
            # Compare batch components
            for i in range(min(len(test_batch), len(verite_batch))):
                test_comp = test_batch[i]
                verite_comp = verite_batch[i]
                
                print(f"\nComponent {i}:")
                print(f"  Types - test: {type(test_comp)}, verite: {type(verite_comp)}")
                
                if isinstance(test_comp, torch.Tensor) and isinstance(verite_comp, torch.Tensor):
                    print(f"  Shapes - test: {test_comp.shape}, verite: {verite_comp.shape}")
                    
                    shape_match = test_comp.shape == verite_comp.shape
                    print(f"  Shapes match: {shape_match}")
                    
                    if shape_match:
                        try:
                            equal = torch.allclose(test_comp, verite_comp)
                            print(f"  Values match: {equal}")
                            if not equal:
                                print("  ⚠️ DIFFERENCE DETECTED: Tensor values don't match!")
                                
                                # Sample a few elements to show differences
                                if test_comp.numel() > 0:
                                    sample_idx = torch.randint(0, test_comp.numel(), (5,))
                                    for idx in sample_idx:
                                        flat_idx = idx.item()
                                        test_val = test_comp.view(-1)[flat_idx].item()
                                        verite_val = verite_comp.view(-1)[flat_idx].item()
                                        print(f"    Sample element {flat_idx}: {test_val} vs {verite_val}")
                        except Exception as e:
                            print(f"  Error comparing tensors: {e}")
                    else:
                        print("  ⚠️ DIFFERENCE DETECTED: Tensor shapes don't match!")
                
                elif isinstance(test_comp, pd.DataFrame) and isinstance(verite_comp, pd.DataFrame):
                    print(f"  Shapes - test: {test_comp.shape}, verite: {verite_comp.shape}")
                    
                    shape_match = test_comp.shape == verite_comp.shape
                    print(f"  Shapes match: {shape_match}")
                    
                    cols_match = set(test_comp.columns) == set(verite_comp.columns)
                    print(f"  Columns match: {cols_match}")
                    
                    if shape_match and cols_match:
                        try:
                            equal = test_comp.equals(verite_comp)
                            print(f"  Values match: {equal}")
                            if not equal:
                                print("  ⚠️ DIFFERENCE DETECTED: DataFrame values don't match!")
                        except Exception as e:
                            print(f"  Error comparing DataFrames: {e}")
                    else:
                        print("  ⚠️ DIFFERENCE DETECTED: DataFrame structure doesn't match!")
                
                elif type(test_comp) != type(verite_comp):
                    print("  ⚠️ DIFFERENCE DETECTED: Component types don't match!")
                
                else:
                    try:
                        equal = test_comp == verite_comp
                        if isinstance(equal, bool):
                            print(f"  Values match: {equal}")
                            if not equal:
                                print("  ⚠️ DIFFERENCE DETECTED: Values don't match!")
                        else:
                            print("  Cannot directly compare these components")
                    except Exception as e:
                        print(f"  Error comparing components: {e}")
    
    except Exception as e:
        print(f"Error extracting batches: {e}")
    
    # PART 5: CHECK DATASET ATTRIBUTES
    print("\n===== DATASET ATTRIBUTE COMPARISON =====")
    
    if hasattr(test_dataloader, 'dataset') and hasattr(verite_dataloader, 'dataset'):
        test_dataset = test_dataloader.dataset
        verite_dataset = verite_dataloader.dataset
        
        # Check if 'use_evidence' is an attribute or parameter used
        if use_evidence is not None:
            print(f"use_evidence parameter value: {use_evidence}")
        
        if hasattr(test_dataset, 'use_evidence') and hasattr(verite_dataset, 'use_evidence'):
            test_evidence = getattr(test_dataset, 'use_evidence')
            verite_evidence = getattr(verite_dataset, 'use_evidence')
            print(f"use_evidence values - test: {test_evidence}, verite: {verite_evidence}")
            
            if test_evidence != verite_evidence:
                print("⚠️ CRITICAL DIFFERENCE: 'use_evidence' values don't match!")
        
        # Get non-callable attributes
        test_attrs = {attr for attr in dir(test_dataset) 
                     if not attr.startswith('__') and not callable(getattr(test_dataset, attr))}
        verite_attrs = {attr for attr in dir(verite_dataset) 
                       if not attr.startswith('__') and not callable(getattr(verite_dataset, attr))}
        
        # Show attribute differences
        attr_diff1 = test_attrs - verite_attrs
        attr_diff2 = verite_attrs - test_attrs
        
        if attr_diff1:
            print(f"Attributes only in test dataset: {attr_diff1}")
        if attr_diff2:
            print(f"Attributes only in verite dataset: {attr_diff2}")
        
        # Compare common attributes that might affect behavior
        important_attrs = ['transform', 'target_transform', 'transforms', 'indices', 'shuffle']
        
        for attr in important_attrs:
            if hasattr(test_dataset, attr) and hasattr(verite_dataset, attr):
                test_val = getattr(test_dataset, attr)
                verite_val = getattr(verite_dataset, attr)
                
                print(f"\nAttribute '{attr}':")
                print(f"  Types - test: {type(test_val)}, verite: {type(verite_val)}")
                
                try:
                    equal = test_val == verite_val
                    if isinstance(equal, bool):
                        print(f"  Values match: {equal}")
                        if not equal:
                            print(f"  ⚠️ DIFFERENCE DETECTED: '{attr}' values don't match!")
                    else:
                        print(f"  Cannot directly compare these attributes")
                except Exception as e:
                    print(f"  Error comparing '{attr}': {e}")