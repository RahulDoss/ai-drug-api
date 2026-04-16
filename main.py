# main.py
import os
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# Real Science Libraries
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from Bio.SeqUtils import ProtParam
from Bio import SeqUtils

# Initialize Gemma 4 via Google AI Studio
# Make sure to export GOOGLE_API_KEY in your environment
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemma-4-e2b-it')

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ScienceEngine:
    @staticmethod
    def analyze_molecule(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        
        # 1. Real 3D Conformation Generation (MMFF94 Force Field)
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol_h) # Real energy minimization
        pdb_block = Chem.MolToPDBBlock(mol_h)
        
        # 2. Real ADMET & Drug-likeness Metrics
        return {
            "smiles": smiles,
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "qed": QED.qed(mol), # Real Quantitative Estimate of Drug-likeness
            "h_donors": Descriptors.NumHDonors(mol),
            "h_acceptors": Descriptors.NumHAcceptors(mol),
            "tpsa": rdMolDescriptors.CalcTPSA(mol), # Topo. Polar Surface Area
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "pdb": pdb_block
        }

    @staticmethod
    def analyze_vaccine_peptide(sequence: str):
        # Real Proteomic Analysis
        analysed_seq = ProtParam.ProteinAnalysis(sequence)
        return {
            "sequence": sequence,
            "molecular_weight": analysed_seq.molecular_weight(),
            "aromaticity": analysed_seq.aromaticity(),
            "instability_index": analysed_seq.instability_index(),
            "isoelectric_point": analysed_seq.isoelectric_point(),
            "hydrophobicity": analysed_seq.gravy()
        }

@app.post("/api/research")
async def execute_discovery(request: Dict[str, str]):
    prompt = request.get("prompt")
    
    # PHASE 1: Gemma 4 Brain - Hypothesis Generation
    # We force Gemma to act as a chemist and output valid SMILES or Sequences
    orchestrator_query = f"""
    As an AI Scientific Agent, solve: {prompt}
    Generate 3 high-probability drug candidates (SMILES) or vaccine epitopes (Peptides).
    You must provide valid, scientifically accurate chemical structures.
    Output ONLY JSON:
    {{
        "mode": "drug" | "vaccine",
        "rationale": "Detailed scientific explanation",
        "candidates": ["string1", "string2", "string3"]
    }}
    """
    
    raw_response = model.generate_content(orchestrator_query)
    # Parsing safety for JSON
    clean_json = raw_response.text.strip().replace("```json", "").replace("```", "")
    plan = json.loads(clean_json)
    
    # PHASE 2: Real Scientific Calculation
    processed_candidates = []
    for c in plan['candidates']:
        if plan['mode'] == "drug":
            result = ScienceEngine.analyze_molecule(c)
        else:
            result = ScienceEngine.analyze_vaccine_peptide(c)
        if result: processed_candidates.append(result)

    # PHASE 3: Multi-Objective Ranking
    # Ranking formula: Higher QED (drug-likeness) + proper LogP (bioavailability)
    for res in processed_candidates:
        if plan['mode'] == "drug":
            # Multi-objective score (0-100)
            score = (res['qed'] * 50) + (max(0, 10 - abs(3 - res['logp'])) * 5)
            res['final_score'] = round(min(score, 99.9), 2)
        else:
            # Vaccine score based on stability and hydrophobicity
            score = 100 - res['instability_index']
            res['final_score'] = round(max(0, score), 2)

    # PHASE 4: Gemma 4 Final Validation
    validation_prompt = f"Analyze these calculated results: {json.dumps(processed_candidates)}. Explain which one is the most viable lead for {prompt}."
    justification = model.generate_content(validation_prompt).text

    return {
        "analysis": plan,
        "results": sorted(processed_candidates, key=lambda x: x['final_score'], reverse=True),
        "explanation": justification
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
