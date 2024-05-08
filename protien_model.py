import csv 
import PIL
import random
import duckdb
import pandas as pd
import cairosvg
import io
import os
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from rdkit.Chem import Draw


# Generate ECFPs
def generate_ecfp(molecule, radius=2, bits=1024):
    if molecule is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))



def printRow(fname):

    list_of_column_names = []

    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        counter = 0

        for row in csv_reader:
            if row[-1] == "1":
                list_of_column_names.append(row)
                counter += 1
            if counter == 4:
                break

    for line in range(len(list_of_column_names)):
        print(list_of_column_names[line])
    
def data_preprocess():

    train_path = 'train.parquet'
    test_path = 'test.parquet'

    con = duckdb.connect()

    df = con.query(f"""(SELECT *
                        FROM parquet_scan('{train_path}')
                        WHERE binds = 0
                        ORDER BY random()
                        LIMIT 3000)
                        UNION ALL
                        (SELECT *
                        FROM parquet_scan('{train_path}')
                        WHERE binds = 1
                        ORDER BY random()
                        LIMIT 3000)""").df()

    con.close()
    print(df.head())

    return df

def molecule_to_pdf(mol, file_name, width=300, height=300):
    """Save substance structure as PDF"""

    # Define full path name
    full_path = f"./2Dstruct/{file_name}.pdf"

    # Render high resolution molecule
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Export to pdf
    cairosvg.svg2pdf(bytestring=drawer.GetDrawingText().encode(), write_to=full_path)

def main():
    df = data_preprocess()
    # Convert SMILES to RDKit molecules
    df['molecule'] = df['molecule_smiles'].apply(Chem.MolFromSmiles)
    df['ecfp'] = df['molecule'].apply(generate_ecfp)
    print(df.head())
    cnt = 0
    for mol in df['molecule']:
        randNum = random.random()
        molecule_to_pdf(mol, str(randNum))
        break
    # One-hot encode the protein_name
    onehot_encoder = OneHotEncoder(sparse_output=False)
    protein_onehot = onehot_encoder.fit_transform(df['protein_name'].values.reshape(-1, 1))

    # Combine ECFPs and one-hot encoded protein_name
    X = [ecfp + protein for ecfp, protein in zip(df['ecfp'].tolist(), protein_onehot.tolist())]
    y = df['binds'].tolist()

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the random forest model
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probability of the positive class

    # Calculate the mean average precision
    map_score = average_precision_score(y_test, y_pred_proba)
    print(f"Mean Average Precision (mAP): {map_score:.2f}")




main()
