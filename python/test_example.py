#!/usr/bin/env python3
"""
Simple test of APM Python bindings with dummy panel data
"""
import sys
sys.path.insert(0, 'src')

import apm
import numpy as np

def test_basic():
    print(f"APM version: {apm.get_version()}")
    
def test_impute_outcomes():
    print("\nTesting impute_outcomes with dummy data...")
    
    # Create dummy factor matrix G (T x r)
    T = 5  # 5 time periods
    r = 2  # 2 factors
    G = np.random.randn(T, r)
    print(f"Factor matrix G shape: {G.shape}")
    print(f"G =\n{G}")
    
    # Observed outcome indices for a cohort (0-indexed)
    T_c = [0, 2, 4]  # observed outcomes at times 0, 2, 4
    print(f"Observed outcome indices: {T_c}")
    
    # Observed outcome values for this cohort
    m_c = np.array([1.5, 2.1, 1.8])  # values at times 0, 2, 4
    print(f"Observed outcomes: {m_c}")
    
    try:
        # Call impute_outcomes
        result = apm.impute_outcomes(G, T_c, m_c)
        print(f"Imputed outcomes: {result}")
        print(f"Result shape: {result.shape}")
        
        print("\nSuccess! The APM bindings are working.")
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    test_basic()
    result = test_impute_outcomes()