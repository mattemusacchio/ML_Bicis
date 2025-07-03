#!/usr/bin/env python3
import pandas as pd

# Leer el dataset de test
df = pd.read_csv('data/processed/trips_2024_test.csv')

# Mostrar las primeras filas
print("\nPrimeras filas:")
print(df.head())

# Mostrar información del dataset
print("\nInformación del dataset:")
print(df.info()) 